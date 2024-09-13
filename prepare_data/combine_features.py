import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
import xgboost as xgb
import gc
import os


# Combine ecfp and building blocks features.
# This script was run on RTX 3090, 124 Gb RAM.

CHUNK_SIZE = int(1e6)
SUBSET_NAME = "train_wide_50M"

all_bb_fearures = pd.read_parquet(os.path.join("../data", f"bb_features_for_{SUBSET_NAME}.parquet"))

FEATURES_PATH = os.path.join("../data", f'ecfp_1024_{SUBSET_NAME}.npz')

print("Loading features from", FEATURES_PATH)
features = sparse.load_npz(FEATURES_PATH)

targets = ["BRD4", "HSA", "sEH"]

for target in targets:
    
    print("Extracting bb features for", target)    
    bb_features = all_bb_fearures.filter(like = target)
    bb_features = sparse.csr_matrix(bb_features.values).astype('float32')

    comb_features = []
    print("Starting to add by chunks")
    for i in tqdm(range(0, features.shape[0], CHUNK_SIZE)):
        X = features[i : np.min([i+CHUNK_SIZE, features.shape[0]]), :]
        X2 = bb_features[i : np.min([i+CHUNK_SIZE, bb_features.shape[0]]), :]
        X = sparse.hstack([X, X2]) 
        comb_features.append(X)
        del X, X2
        gc.collect()
    comb_features = sparse.vstack(comb_features) 
    
    RESULT_PATH = os.path.join("../data", f"ecfp_1024_{SUBSET_NAME}_{target}")
    
    print("Saving xgb.DMatrix")
    comb_features = xgb.DMatrix(comb_features)
    comb_features.save_binary(f'{RESULT_PATH}.buffer')
    print(f"Features saved to {RESULT_PATH}.buffer")