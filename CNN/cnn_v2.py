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
    EPOCHS = 50 #20
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

train = pd.read_parquet('train_no_test_wide_encoded.parquet')
valid = pd.read_parquet('test_ensemble_wide_modified.parquet')
test = pd.read_parquet('test_enc.parquet')


def my_model():
    with strategy.scope():
        INP_LEN = 142
        NUM_FILTERS = 64
        hidden_dim = 128

        inputs = tf.keras.layers.Input(shape=(INP_LEN,), dtype='int32')
        x2 = tf.keras.layers.Embedding(input_dim=36, output_dim=hidden_dim, input_length=INP_LEN, mask_zero = True)(inputs)
        x = tf.keras.layers.Conv1D(filters=NUM_FILTERS, kernel_size=19,  activation='silu', padding='valid',  strides=1)(x2)
        x = tf.keras.layers.Conv1D(filters=NUM_FILTERS*2, kernel_size=9,  activation='silu', padding='valid',  strides=1)(x)
        x = tf.keras.layers.Conv1D(filters=NUM_FILTERS*3, kernel_size=3,  activation='silu', padding='valid',  strides=1)(x)
        
        x = tf.keras.layers.GlobalMaxPooling1D()(x)

        fw = tf.keras.layers.GRU(128, go_backwards=True)
        x_1 = fw(x2)

        x = tf.concat([x, x_1], 1)

        x = tf.keras.layers.Dense(1024, activation='silu')(x)
        x = tf.keras.layers.Dropout(0.1)(x) #0.1
        x = tf.keras.layers.Dense(1024, activation='silu')(x)
        x = tf.keras.layers.Dropout(0.1)(x) #0.1
        x = tf.keras.layers.Dense(512, activation='silu')(x)
        x = tf.keras.layers.Dropout(0.1)(x) #0.1

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


EXP_NAME = 'cnn_v2'
    

FEATURES = [f'enc{i}' for i in range(142)]
TARGETS = ['bind1', 'bind2', 'bind3']

X_train = train.loc[:, FEATURES]
y_train = train.loc[:, TARGETS]

X_val = valid.loc[:, FEATURES]
y_val = valid.loc[:, TARGETS]

es = tf.keras.callbacks.EarlyStopping(patience=5, monitor="val_loss", mode='min', verbose=1)
checkpoint = tf.keras.callbacks.ModelCheckpoint(monitor='val_loss', filepath=f"{EXP_NAME}_model.h5",
                                                    save_best_only=True, save_weights_only=True,
                                                mode='min')
reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=5, verbose=1)

map_callback = MeanAveragePrecisionCallback(validation_data=(X_val.values, y_val.values))

model = my_model()
history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CFG.EPOCHS,
        callbacks=[checkpoint, 
                   reduce_lr_loss, 
                   es, 
                   map_callback,
                  ],
        batch_size=CFG.BATCH_SIZE,
        verbose=1,
    )

model.load_weights(f"{EXP_NAME}_model.h5")
oof = model.predict(X_val, batch_size = 2*CFG.BATCH_SIZE)
score = APS(y_val, oof, average = 'micro')
print('CV score =', score)
score_rp = {'CV':score,}


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

