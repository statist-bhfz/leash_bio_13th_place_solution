# Steps to Reproduce GBDT Models

This document outlines the steps required to reproduce solution's GBDT models using various feature sets and configurations.

## Models

### 1. XGb_ecfp_1024_train_wide_split_by_bb_5f_nonshare_val

1. Make train subset with `make_splits_by_bb.R` (Output: `train_wide_split_by_bb.parquet`).
2. Generate ECFP 1024 features with `make_molfeat_features.py` for the train subset and test data.
3. Run `XGb_ecfp_1024_train_wide_split_by_bb_5f_nonshare_val.R` to get test predictions.

---

### 2. lightgbm_secfp_1024_train_no_test_wide_50M and lightgbm_secfp6_2048_train_no_test_wide_40M

1. Make train subsets with `make_train_test_split.R` (Outputs: `train_no_test_wide_50M.parquet` and `train_no_test_wide_40M.parquet`).
2. Generate SECFP 1024 and SECFP:6 2048 features with `make_molfeat_features.py` for the train subsets and test data.
3. Run `lightgbm_models.ipynb` to get test predictions.

---

### 3. XGb_ecfp1024_plus_BB_act_50m_v2_new_bb_XGb_ecfp1024_574

1. Run `make_splits_by_bb.R` to create `train_wide.parquet` and its subset of 50M samples `train_wide_50M.parquet`.
2. Use `make_molfeat_features.py` to generate ECFP 1024 features for the train subset and test data.
3. Run `make_bb_features.R` to generate building blocks (BB) features based on `train_wide.parquet` and aligned with `train_wide_50M.parquet` and the test data (Outputs: `bb_features_for_train_wide_50M.parquet` and `bb_features_for_test.parquet`).
4. Run `combine_features.py` to get 3 files (`.buffer` for each protein), containing ECFP4 1024 and BB features.
5. Train XGBoost with `XGb_ecfp1024_plus_BB_act_50m_v2_new_bb_XGb_ecfp1024_574.R` to get test predictions.

---

### 4. XGb_ecfp_1024_5_bb_parts_BB_act_50M

1. Run `make_splits_by_bb.R` to create `train_wide.parquet` and split it into 5 parts (Files: `train_wide_bb_split_p(1 to 5).parquet`).
2. Use `make_molfeat_features.py` to generate ECFP 1024 features for each train subset and test data.
3. Use `make_bb_features.R` to generate BB features based on `train_wide.parquet` and aligned with the train subsets and test data (Outputs: 5 files `bb_features_for_train_wide_bb_split_p(1 to 5).parquet`, and `bb_features_for_test.parquet`).
4. Run `combine_features.py` to combine BB features and ECFP features (Outputs: 15 files `ecfp_1024_train_wide_bb_split_p(1 to 5)_(BRD4, HSA or sEH).buffer`).
5. Train XGBoost with `XGb_ecfp_1024_5_bb_parts_BB_act_50M.R` to get test predictions.

---

### 5. XGb_secfp1024_mpnn_che2_10Mtnt_last15_by_prot_train_no_test_wide

1. Run `make_train_test_split.R` to create subsets: `train_no_test_wide_50M.parquet` and `train_no_test_wide_10M.csv`.
2. Generate SECFP 1024 features with `make_molfeat_features.py` for the train subset `train_no_test_wide_50M.parquet` and test data.
3. Install Chemprop 2.0 and run the following to train the model:

   ```bash
   chemprop train \
   --data-path /path/to/train_no_test_wide_10M.csv  \
   --output-dir /path/to/checkpoints  \
   --log /path/to/checkpoints/train_log.log  \
   --task-type classification  \
   --metric bce -s molecule_smiles  \
   --target-columns BRD4 HSA sEH  \
   --num-workers 15 -vvv \
   --epochs 15  \
   --multi-hot-atom-featurizer-mode v1
   ```

4. Generate Chemprop Fingerprints: To handle the large dataset of 50M samples, you may need to split the `train_no_test_wide_50M.parquet`. We have divided it into 17 files to generate fingerprints on a machine with 124 Gb RAM. Run the following command for each part:

   ```bash
   chemprop fingerprint \
   --test-path /path/to/train_no_test_wide_50M_1.csv \
   -o /path/to/data/mpnn_che2_10Mtnt_last15_train_no_test_wide_50M_1.csv \
   --model-path /path/to/checkpoints/model_0/checkpoints/last.ckpt \
   -s molecule_smiles \
   --multi-hot-atom-featurizer-mode v1 \
   --ffn-block-index 3
   ```
   Once generated, combine the fingerprints into a single dataset using R:

   ```r
   file_paths <- file.path("../data", paste0(
      "mpnn_che2_10Mtnt_last15_train_no_test_wide_50M_", 1:17, "_0.csv")
    )
    dt <- rbindlist(lapply(file_paths, fread))
    setnames(dt, names(dt), c("BRD4_chem2", "HSA_chem2", "sEH_chem2"))
    write_parquet(dt, "mpnn_che2_10Mtnt_last15_for_train_no_test_wide_50M.parquet")
    ```
5. Generate Test Data Fingerprints:
   
    ```bash
    chemprop fingerprint \
    --test-path /path/to/test.csv \
    -o /path/to/data/mpnn_che2_10Mtnt_last15_test.csv \
    --model-path /path/to/checkpoints/model_0/checkpoints/last.ckpt \
    -s molecule_smiles \
    --multi-hot-atom-featurizer-mode v1 \
    --ffn-block-index 3
    ```
    Then load, rename columns, and save the output:

   ```r
   dt <- fread("../data/mpnn_che2_10Mtnt_last15_test.csv")
   setnames(dt, names(dt), c("BRD4_chem2", "HSA_chem2", "sEH_chem2"))
   write_parquet(dt, "../data/mpnn_che2_10Mtnt_last15_for_test.parquet")
   ```
6. Use `combine_features.py` to combine the Chemprop fingerprints with the SECFP features (Output: 3 files `secfp_1024_train_no_test_wide_50M_(BRD4, HSA or sEH).buffer`).
7. Finally, train the XGBoost model using the script `XGb_secfp1024_mpnn_che2_10Mtnt_last15_by_prot_train_no_test_wide.R` to generate predictions for the test data.

   **Note**: Ensure that the correct file paths are used in all commands. Additionally, make sure Chemprop is correctly installed following the [official installation guide](https://chemprop.readthedocs.io/en/latest/installation.html) before running any Chemprop-related commands.

## Requirements
R version >= 4.3  
Python version >= 3.11  
XGBoost version 2.1.0  
LightGBM version 4.5.0  
Chemprop version 2.0.0  
