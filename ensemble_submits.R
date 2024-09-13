
library(data.table)
library(qs)
library(ggplot2)
library(arrow)

# This script was run on RTX 3090, 124 Gb RAM
# Set working directory to script location
setwd("/home/tonia/BELKA/solution_GBDT_models_git/prepare_data")

dir = "../submits"
test_dt <- fread("../data/test.csv")

# split test to molecules with shared and new building blocks
train_dt <- read_parquet("../data/train_wide.parquet")
setDT(train_dt)

bb1 <- train_dt[, unique(buildingblock1_smiles)]
bb2 <- train_dt[, unique(buildingblock2_smiles)]
bb3 <- train_dt[, unique(buildingblock3_smiles)]

test_dt[, bb_set := ifelse(
    buildingblock1_smiles %chin% bb1 |
    buildingblock2_smiles %chin% bb2 |
    buildingblock3_smiles %chin% bb3, "shared", "new"
)]

rm(train_dt, bb1, bb2, bb3) ; gc()

get_ensembse_preds <- function(ens_files, on_ranks = FALSE) {  
  ensembse <- test_dt[, .(id, protein_name, bb_set)]
  
  lapply(names(ens_files), function(protein) {    
    if (sum(unlist(ens_files[[protein]])) != 1) stop("Weights sum is not 1")
    
    preds_list <- lapply(names(ens_files[[protein]]), function(x) {      
      model_preds <- fread(file.path(dir, x))
      model_preds <- model_preds[test_dt[, .(id, protein_name)], on = "id"]      
      model_preds <- model_preds[protein_name == protein, .(binds)]
      names(model_preds) <- x
      model_preds
    })
    preds_list <- lapply(preds_list, function(pred) {      
      model <- names(pred)[1]      
      if (on_ranks) pred[, (model) := rank(get(model)) / max(rank(get(model)))]      
      weight <- ens_files[[protein]][[model]]
      pred <- as.matrix(pred * weight)
    })    
    ave_preds <- Reduce("+", preds_list)    
    
    ensembse[protein_name == protein, binds := ave_preds]
  })
  return(ensembse)
}

ens_files = list(
  
  "BRD4" = list(
    "XGb_ecfp1024_plus_BB_act_50m_v2_new_bb_XGb_ecfp1024_574.csv" = 0.075,
    "XGb_ecfp_1024_5_bb_parts_BB_act_50M.csv" = 0.075,
    "XGb_secfp1024_mpnn_che2_10Mtnt_last15_by_prot_train_no_test_wide_50M.csv" = 0.15,
    "cnn_v1" = 0.7*0.277,
    "cnn_v2" = 0.7*0.277,
    "cnn_v3" = 0.7*0.223,
    "cnn_v4" = 0.7*0.223
  ),
  "HSA" = list(
    "XGb_ecfp1024_plus_BB_act_50m_v2_new_bb_XGb_ecfp1024_574.csv" = 0.0375,
    "XGb_ecfp_1024_5_bb_parts_BB_act_50M.csv" = 0.0375,
    "XGb_secfp1024_mpnn_che2_10Mtnt_last15_by_prot_train_no_test_wide_50M.csv" = 0.075, 
    "chemberta_all_train_v2_continue.csv"= 0.075,
    "gnn_50M_100M.csv" = 0.0375,
    "gnn_50M_100M_v2.csv" = 0.0375,
    "cnn_v1.csv" = 0.7*0.277,
    "cnn_v2.csv" = 0.7*0.277,
    "cnn_v3.csv" = 0.7*0.223,
    "cnn_v4.csv" = 0.7*0.223
  ),
  "sEH" = list(
    "lightgbm_secfp_1024_train_no_test_wide_50M.csv" = 0.005,
    "lightgbm_secfp:6_2048_train_no_test_wide_40M.csv" = 0.005,
    "XGb_secfp1024_mpnn_che2_10Mtnt_last15_by_prot_train_no_test_wide_50M.csv" = 0.045,
    "gnn_50M_100M.csv" = 0.045/2,
    "gnn_50M_100M_v2.csv" = 0.045/2,
    "cnn_v1.csv" = 0.9*0.277,
    "cnn_v2.csv" = 0.9*0.277,
    "cnn_v3.csv" = 0.9*0.223,
    "cnn_v4.csv" = 0.9*0.223
  )
)

ensemble <- get_ensembse_preds(ens_files)

ens_files = list(
  
  "BRD4" = list(
    "XGb_ecfp_1024_train_wide_split_by_bb_5f_nonshare_val.csv" = 0.4,
    "model3_5fod_ave.csv" = 0.3,
    "tabnet_fold_ave.csv" = 0.3
  ),
  "HSA" = list(
    "chemberta_weight2_epoch3_mean_5folds.csv" = 0.6,
    "chemberta_all_train_v2_mean_3_epochs.csv" = 0.2,
    "chemberta_model6_nonsharelb0139.csv" = 0.2
  ),
  "sEH" = list(
    "chemberta_weight2_epoch3_mean_5folds.csv" = 0.2,
    "chemberta_all_train_v2_mean_3_epochs.csv" = 0.3,
    "chemberta_dnn_v4_lb441.csv" = 0.3,
    "chemberta_weight1_epoch3_mean_5folds.csv" = 0.2
  )
)

ens_nonshare <- get_ensembse_preds(ens_files, on_ranks = TRUE)

ensemble[, non_share := ens_nonshare[, binds]]
ensemble[bb_set == "new", binds := non_share]

ensemble <- ensemble[, .(id, binds)]

fwrite(ensemble, "../submits/ensemble58.csv")