import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
import xgboost as xgb
from molfeat.trans.fp import FPVecTransformer
import gc

# This script was run on RTX 3090, 124 Gb RAM. 
# Generating ecfp features for 50M train samples took about 1h 40 min.
# Generating secfp features for 50M train samples took about 3h 20 min.

# For GBDT models in the solution, make:
# - ecfp 1024 for train_wide_split_by_bb.parquet (model XGb_ecfp_1024_train_wide_split_by_bb_5f_nonshare_val)
# - ecfp 1024 for train_wide_bb_split_p(1 to 5).parquet (model XGb_ecfp_1024_5_bb_parts_BB_act_50M)
# - ecfp 1024 for train_wide_50M.parquet (model XGb_ecfp1024_plus_BB_act_50m_v2_new_bb_XGb_ecfp1024_574)
# - secfp 1024 for train_no_test_wide_50M.parquet (model XGb_secfp1024_mpnn_che2_10Mtnt_last15_by_prot_train_no_test_wide and lightgbm_secfp_1024_train_no_test_wide_50M_v1)
# - secfp:6 2048 for train_no_test_wide_40M.parquet (model lightgbm_secfp6_2048_train_no_test_wide_40M_v1)


KIND = 'secfp' # ecfp, secfp:6
LENGTH = 1024 # 2048
SUBSET_NAME = 'train_no_test_wide_50M' #train_wide_split_by_bb, train_wide_50M, etc.


CHUNK_SIZE = int(1e6)

OUTPUT_TRAIN = f'../data/{KIND}_{LENGTH}_{SUBSET_NAME}'
OUTPUT_TEST = f'../data/{KIND}_{LENGTH}_test'

def get_features_sparce(df, chunck_size, kind = "ecfp", length = 1024, n_jobs = 15):
    transformer = FPVecTransformer(kind = kind, length = length, dtype = np.float32, n_jobs = n_jobs, verbose = True)
    result = []
    for i in tqdm(range(0, df.shape[0], chunck_size)):
        X = transformer(df.iloc[i : np.min([i+chunck_size, df.shape[0]])].molecule_smiles)
        X = sparse.csr_matrix(X)
        result.append(X)
    result = sparse.vstack(result)
    return result 

# Read parquet file with train data created with make_splits_by_bb.R or make_train_test_split.R
df = pd.read_parquet(f'../data/{SUBSET_NAME}.parquet', columns=['molecule_smiles'])

print(f'Making {KIND}{LENGTH} features for {SUBSET_NAME}')
features = get_features_sparce(df, chunck_size = CHUNK_SIZE, kind = KIND, length = LENGTH)

del df
gc.collect()

print("Saving npz")
sparse.save_npz(f'{OUTPUT_TRAIN}.npz', features)
print(f"Features saved to {OUTPUT_TRAIN}.npz")

print("Saving xgb.DMatrix")
features = xgb.DMatrix(features)
features.save_binary(f'{OUTPUT_TRAIN}.buffer')
print(f"Features saved to {OUTPUT_TRAIN}.buffer")

del features
gc.collect()

# Read competition file with test data
df = pd.read_csv("../data/test.csv") 

print(f'Making {KIND}{LENGTH} features for test')
features = get_features_sparce(df, chunck_size = CHUNK_SIZE, kind = KIND, length = LENGTH)

del df
gc.collect()

print("Saving npz")
sparse.save_npz(f'{OUTPUT_TEST}.npz', features)
print(f"Features saved to {OUTPUT_TEST}.npz")