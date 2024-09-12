import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm


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