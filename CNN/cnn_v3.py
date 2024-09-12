import gc
import os
import pickle
import random
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score as APS


class CFG:

    PREPROCESS = False
    EPOCHS = 50
    BATCH_SIZE = 4096
    LR = 1e-3
    WD = 0.05

    NBR_FOLDS = 15
    SELECTED_FOLDS = [0]

    SEED = 2024


import tensorflow as tf
from tensorflow.keras import layers, Model
from functools import partial
from tensorflow.keras.callbacks import Callback

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

set_seeds(seed=CFG.SEED)


try:
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
except:
    pass


if CFG.PREPROCESS:
    enc = {'l': 1, 'y': 2, '@': 3, '3': 4, 'H': 5, 'S': 6, 'F': 7, 'C': 8, 'r': 9, 's': 10, '/': 11, 'c': 12, 'o': 13,
           '+': 14, 'I': 15, '5': 16, '(': 17, '2': 18, ')': 19, '9': 20, 'i': 21, '#': 22, '6': 23, '8': 24, '4': 25, '=': 26,
           '1': 27, 'O': 28, '[': 29, 'D': 30, 'B': 31, ']': 32, 'N': 33, '7': 34, 'n': 35, '-': 36}
    train_raw = pd.read_parquet('kaggle/input/leash-BELKA/train.parquet')
    smiles = train_raw[train_raw['protein_name']=='BRD4']['molecule_smiles'].values
    assert (smiles!=train_raw[train_raw['protein_name']=='HSA']['molecule_smiles'].values).sum() == 0
    assert (smiles!=train_raw[train_raw['protein_name']=='sEH']['molecule_smiles'].values).sum() == 0
    def encode_smile(smile):
        tmp = [enc[i] for i in smile]
        tmp = tmp + [0]*(142-len(tmp))
        return np.array(tmp).astype(np.uint8)

    smiles_enc = joblib.Parallel(n_jobs=96)(joblib.delayed(encode_smile)(smile) for smile in tqdm(smiles))
    smiles_enc = np.stack(smiles_enc)
    train = pd.DataFrame(smiles_enc, columns = [f'enc{i}' for i in range(142)])
    train['bind1'] = train_raw[train_raw['protein_name']=='BRD4']['binds'].values
    train['bind2'] = train_raw[train_raw['protein_name']=='HSA']['binds'].values
    train['bind3'] = train_raw[train_raw['protein_name']=='sEH']['binds'].values
    train.to_parquet('train_enc.parquet')

    test_raw = pd.read_parquet('kaggle/input/leash-BELKA/test.parquet')
    smiles = test_raw['molecule_smiles'].values

    smiles_enc = joblib.Parallel(n_jobs=96)(joblib.delayed(encode_smile)(smile) for smile in tqdm(smiles))
    smiles_enc = np.stack(smiles_enc)
    test = pd.DataFrame(smiles_enc, columns = [f'enc{i}' for i in range(142)])
    test.to_parquet('test_enc.parquet')

else:
    train = pd.read_parquet('kaggle/input/belka-enc-dataset/train_enc.parquet')
    test = pd.read_parquet('kaggle/input/belka-enc-dataset/test_enc.parquet')
    

def my_model():
    with strategy.scope():
        INP_LEN = 142 # 128
        NUM_FILTERS = 64
        hidden_dim = 128

        inputs = tf.keras.layers.Input(shape=(INP_LEN,), dtype='int32')
        x = tf.keras.layers.Embedding(input_dim=36, output_dim=hidden_dim, input_length=INP_LEN, mask_zero = True)(inputs)
        x = tf.keras.layers.Conv1D(filters=NUM_FILTERS, kernel_size=19,  activation='silu', padding='valid',  strides=1)(x)
        x = tf.keras.layers.Conv1D(filters=NUM_FILTERS*2, kernel_size=9,  activation='silu', padding='valid',  strides=1)(x)
        x = tf.keras.layers.Conv1D(filters=NUM_FILTERS*3, kernel_size=3,  activation='silu', padding='valid',  strides=1)(x)
        x = tf.keras.layers.GlobalMaxPooling1D()(x)

        x = tf.keras.layers.Dense(1024, activation='silu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(1024, activation='silu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(512, activation='silu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)

        outputs = tf.keras.layers.Dense(3, activation='sigmoid')(x)

        model = tf.keras.models.Model(inputs = inputs, outputs = outputs)
        optimizer = tf.keras.optimizers.Adam(learning_rate=CFG.LR, weight_decay = CFG.WD)
        loss = 'binary_crossentropy'
        weighted_metrics = [tf.keras.metrics.AUC(curve='PR', name = 'avg_precision')]
        model.compile(
        loss=loss,
        optimizer=optimizer,
        weighted_metrics=weighted_metrics,
        )
        return model


class MeanAveragePrecisionCallback(Callback):
    def __init__(self, validation_data, class_names=['BRD4', 'HSA', 'sEH'], verbose=False):
        super().__init__()
        self.validation_data = validation_data
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data
        y_pred = self.model.predict(x_val, batch_size=8*CFG.BATCH_SIZE)
        
        # Ensure y_val and y_pred have the correct shape
        if y_val.ndim == 1:
            y_val = np.eye(self.num_classes)[y_val]
        if y_pred.ndim == 1:
            y_pred = np.column_stack((1 - y_pred, y_pred))  # For binary classification
        
        average_precisions = []
        for i in range(self.num_classes):
            ap = APS(y_val[:, i], y_pred[:, i], average = 'micro')
            average_precisions.append(ap)
            logs[f'val_map_{self.class_names[i]}'] = ap
        
        mAP = np.mean(average_precisions)
        logs['val_map'] = mAP

        if self.verbose:
            print(f'\nEpoch {epoch + 1}')
            for i, class_name in enumerate(self.class_names):
                print(f'  AP {class_name}: {average_precisions[i]:.4f}')
            print(f'  mAP: {mAP:.4f}')

EXP_NAME = 'cnn_v3'
    

FEATURES = [f'enc{i}' for i in range(142)]
TARGETS = ['bind1', 'bind2', 'bind3']

skf = StratifiedKFold(n_splits = CFG.NBR_FOLDS, shuffle = True, random_state = 42)

all_preds = []
for fold,(train_idx, valid_idx) in enumerate(skf.split(train, train[TARGETS].sum(1))):
    
    if fold not in CFG.SELECTED_FOLDS:
        continue;
    
    X_train = train.loc[train_idx, FEATURES]
    y_train = train.loc[train_idx, TARGETS]
    X_val = train.loc[valid_idx, FEATURES]
    y_val = train.loc[valid_idx, TARGETS]

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(CFG.BATCH_SIZE)
    valid_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(CFG.BATCH_SIZE)

    es = tf.keras.callbacks.EarlyStopping(patience=5, monitor="val_loss", mode='min', verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(monitor='val_loss', filepath=f"{EXP_NAME}_model.h5",
                                                        save_best_only=True, save_weights_only=True,
                                                    mode='min')
    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=5, verbose=1)
    model = my_model()
    history = model.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs=CFG.EPOCHS,
            callbacks=[checkpoint, reduce_lr_loss, es],
            batch_size=CFG.BATCH_SIZE,
            verbose=1,
        )
    model.load_weights(f"{EXP_NAME}_model.h5")
    oof = model.predict(X_val, batch_size = 2*CFG.BATCH_SIZE)
    print('fold :', fold, 'CV score =', APS(y_val, oof, average = 'micro'))
    
    preds = model.predict(test, batch_size = 2*CFG.BATCH_SIZE)
    all_preds.append(preds)

preds = np.mean(all_preds, 0)

# Convert oof to a DataFrame and assign column names
class_names = ['BRD4', 'HSA', 'sEH']
oof_df = pd.DataFrame(oof, columns=class_names)

scores = {}

# Iterate over each class and corresponding target
for class_name, target in zip(class_names, TARGETS):
    y_val_class = y_val[target]
    oof_class = oof_df[class_name]
    score = APS(y_val_class, oof_class, average='micro')
    scores[class_name] = score
    print(f'CV score for {class_name} =', score)

