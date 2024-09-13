import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
import xgboost as xgb
import gc
import os


# Combine ecfp or secfp and building blocks/chemprop features.
# This script was run on RTX 3090, 124 Gb RAM.

CHUNK_SIZE = int(1e6)
SUBSET_NAME = "train_wide_50M" # train_no_test_wide_50M
FIRST_FEATURES = "ecfp_1024" # secfp_1024
SECOND_FEATURES = "bb_features" # mpnn_che2_10Mtnt_last15

all_fearures = pd.read_parquet(os.path.join("../data", f"{SECOND_FEATURES}_for_{SUBSET_NAME}.parquet"))

FEATURES_PATH = os.path.join("../data", f'{FIRST_FEATURES}_{SUBSET_NAME}.npz')

print("Loading features from", FEATURES_PATH)
features = sparse.load_npz(FEATURES_PATH)

targets = ["BRD4", "HSA", "sEH"]

for target in targets:
    
    print("Extracting bb features for", target)    
    sec_features = all_fearures.filter(like = target)
    sec_features = sparse.csr_matrix(sec_features.values).astype('float32')

    comb_features = []
    print("Starting to add by chunks")
    for i in tqdm(range(0, features.shape[0], CHUNK_SIZE)):
        X = features[i : np.min([i+CHUNK_SIZE, features.shape[0]]), :]
        X2 = sec_features[i : np.min([i+CHUNK_SIZE, sec_features.shape[0]]), :]
        X = sparse.hstack([X, X2]) 
        comb_features.append(X)
        del X, X2
        gc.collect()
    comb_features = sparse.vstack(comb_features) 
    
    RESULT_PATH = os.path.join("../data", f"{FIRST_FEATURES}_{SUBSET_NAME}_{target}")
    
    print("Saving xgb.DMatrix")
    comb_features = xgb.DMatrix(comb_features)
    comb_features.save_binary(f'{RESULT_PATH}.buffer')
    print(f"Features saved to {RESULT_PATH}.buffer")
