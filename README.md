# leash_bio_13th_place_solution

NeurIPS 2024 - Predict New Medicines with BELKA 13th place solution by looseRs team:

[@yyyu54](https://www.kaggle.com/yyyu54) 

[@masasato1999](https://www.kaggle.com/masasato1999)

[@antoninadolgorukova](https://www.kaggle.com/antoninadolgorukova)

[@Ogurtsov](https://www.kaggle.com/Ogurtsov)


Our solution consists of 6 weighted ensembles: one ensemble per target for molecules with shared (BRD4_shared, HSA_shared, sEH_shared) and non-shared (BRD4_nonshared, HSA_nonshared, sEH_nonshared) building blocks. Links to the training scripts for all models are grouped in 4 folders according to model type: *GBDT*, *chemberta*, *CNN*, *GNN*.

# 1. BRD4_nonshared

Model name                                                           | Model weight
-------------------------------------------------------------------- | -------------
GBDT/XGb_ecfp1024_plus_BB_act_50m_v2_new_bb_XGb_ecfp1024_574         | 0.075
GBDT/XGb_ecfp_1024_5_bb_parts_BB_act_50M                             | 0.075
GBDT/XGb_secfp1024_mpnn_che2_10Mtnt_last15_by_prot_train_no_test_wide| 0.15
CNN/cnn_v1                                                           | 0.1561
CNN/cnn_v2                                                           | 0.1561
CNN/cnn_v3                                                           | 0.1939
CNN/cnn_v4                                                           | 0.1939


# 2. HSA_nonshared