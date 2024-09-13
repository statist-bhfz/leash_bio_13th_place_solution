# leash_bio_13th_place_solution

NeurIPS 2024 - Predict New Medicines with BELKA 13th place solution by looseRs team:

[@yyyu54](https://www.kaggle.com/yyyu54) 

[@masasato1999](https://www.kaggle.com/masasato1999)

[@antoninadolgorukova](https://www.kaggle.com/antoninadolgorukova)

[@Ogurtsov](https://www.kaggle.com/Ogurtsov)


Our solution consists of 6 weighted ensembles: one ensemble per target for molecules with shared (BRD4_shared, HSA_shared, sEH_shared) and non-shared (BRD4_nonshared, HSA_nonshared, sEH_nonshared) building blocks. Links to the training scripts for all models are grouped in 4 folders according to model type: *GBDT*, *chemberta*, *CNN*, *GNN*.

Please be lenient if it happens so some code will need additional manual tweaking to make it work. Models were trained on different environments including bunch of local PCs, kaggle and rented servers on vast.ai (the last one with default CUDA 12.4 pytorch image).

Initial data preparation should be done by sequentially running `prepare_data/make_train_test_split.R`, `prepare_data/replace_dy.py`. Please refer to GBDT/README.md for guidance on reproducing the solution's GBDT models.


# 1. BRD4_shared

Model name                                                           | Model weight
-------------------------------------------------------------------- | -------------
GBDT/XGb_ecfp1024_plus_BB_act_50m_v2_new_bb_XGb_ecfp1024_574         | 0.075
GBDT/XGb_ecfp_1024_5_bb_parts_BB_act_50M                             | 0.075
GBDT/XGb_secfp1024_mpnn_che2_10Mtnt_last15_by_prot_train_no_test_wide| 0.15
CNN/cnn_v1                                                           | 0.1561
CNN/cnn_v2                                                           | 0.1561
CNN/cnn_v3                                                           | 0.1939
CNN/cnn_v4                                                           | 0.1939


# 2. HSA_shared

Model name                                                           | Model weight
-------------------------------------------------------------------- | -------------
GBDT/XGb_ecfp1024_plus_BB_act_50m_v2_new_bb_XGb_ecfp1024_574         | 0.0375
GBDT/XGb_ecfp_1024_5_bb_parts_BB_act_50M                             | 0.0375
GBDT/XGb_secfp1024_mpnn_che2_10Mtnt_last15_by_prot_train_no_test_wide| 0.075
GNN/gnn_50M_100M                                                     | 0.0375
GNN/gnn_50M_100M_v2                                                  | 0.0375
chemberta/chemberta_all_train_v2_continue                            | 0.075
CNN/cnn_v1                                                           | 0.1561
CNN/cnn_v2                                                           | 0.1561
CNN/cnn_v3                                                           | 0.1939
CNN/cnn_v4                                                           | 0.1939


# 3. sEH_shared

Model name                                                           | Model weight
-------------------------------------------------------------------- | -------------
GBDT/lightgbm_secfp_1024_train_no_test_wide_50M                      | 0.005
GBDT/XGb_secfp1024_mpnn_che2_10Mtnt_last15_by_prot_train_no_test_wide| 0.045
GBDT/lightgbm_secfp:6_2048_train_no_test_wide_40M                    | 0.005
GNN/gnn_50M_100M                                                     | 0.0225
GNN/gnn_50M_100M_v2                                                  | 0.0225
CNN/cnn_v1                                                           | 0.2007
CNN/cnn_v2                                                           | 0.2007
CNN/cnn_v3                                                           | 0.2493
CNN/cnn_v4                                                           | 0.2493


# 4. BRD4_nonshared

Model name                                                           | Model weight
-------------------------------------------------------------------- | -------------
GBDT/XGb_ecfp_1024_train_wide_split_by_bb_5f_nonshare_val            | 0.4
chemberta/model3_5fod_ave                                            | 0.3
chemberta/tabnet_fold_ave                                            | 0.3


# 5. HSA_nonshared

Model name                                                           | Model weight
-------------------------------------------------------------------- | -------------
chemberta/chemberta_weight2_epoch3_mean_5folds                       | 0.6
chemberta/chemberta_all_train_v2_mean_3_epochs                       | 0.2
chemberta/chemberta_model6_nonsharelb0139                            | 0.2


# 6. sEH_nonshared

Model name                                                           | Model weight
-------------------------------------------------------------------- | -------------
chemberta/chemberta_weight2_epoch3_mean_5folds                       | 0.2
chemberta/chemberta_all_train_v2_mean_3_epochs                       | 0.3
chemberta/chemberta_dnn_v4_lb441                                     | 0.3
chemberta/chemberta_weight1_epoch3_mean_5folds                       | 0.2
