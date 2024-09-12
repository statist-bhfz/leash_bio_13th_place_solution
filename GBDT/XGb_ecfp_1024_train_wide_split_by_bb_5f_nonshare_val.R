library(data.table)
library(qs)
library(yardstick)
library(xgboost)
library(Matrix)
library(arrow)
library(duckdb)
scipy_sparse = reticulate::import("scipy.sparse")

# This script was run on RTX 3090, 124 Gb RAM
# Set working directory to script location
setwd("/home/tonia/BELKA/solution_GBDT_models_git/GBDT")

# Scoring function
average_precision_belka <- function(test_dt) {
  
  evals <- melt(test_dt,
                measure.vars = c("BRD4", "HSA", "sEH"),
                id.vars = "split",
                variable.name = "protein",
                variable.factor = "FALSE",
                value.name = "truth")
  
  preds <- melt(
    test_dt[, .SD, .SDcols = names(test_dt) %like% "pred"],
    measure.vars = patterns("pred")
  )
  
  evals[, pred := preds$value]
  evals[, truth := factor(truth, levels = c("0", "1"))]
  
  evals[, MAP := average_precision_vec(truth, pred, event_level = "second"),
        by = c("split", "protein")]
  
  evals <- unique(evals[, .(split, protein, MAP)])
  return(evals[order(split)])
}

# Data loading

train_features_path <- "../data/ecfp_1024_train_wide_split_by_bb.buffer"
test_features_path <- "../data/ecfp_1024_test.npz"
targets_path <- "../data/train_wide_split_by_bb.parquet"

targets_dt <- as.data.table(read_parquet(targets_path))
dmat_all <- xgb.DMatrix(train_features_path)

test_dt <- fread("../data/test.csv")
test_features <- scipy_sparse$load_npz(test_features_path)
test_features <- xgb.DMatrix(as(test_features, "sparseMatrix"))

g <- gc() ; rm(g)

# Split building blocks into 5 folds

bb1 <- unique(targets_dt[, .(bb = buildingblock1_smiles)])
bb2 <- unique(targets_dt[, .(bb = buildingblock2_smiles)])
bb3 <- unique(targets_dt[, .(bb = buildingblock3_smiles)])

set.seed(1)
bb1[, fold_bb := sample(1:5, .N, replace = TRUE)]
bb2[, fold_bb := sample(1:5, .N, replace = TRUE)]
bb3[, fold_bb := sample(1:5, .N, replace = TRUE)]

# Train models and predict

proteins <- c("BRD4", "HSA", "sEH")
nrounds = 5000
early_stop_rnds = 30
params_list <- list(
  device = "gpu",
  objective = "binary:logistic",
  eta = 0.05,
  max.depth = 25,
  subsample = 0.2,
  sampling_method = "gradient_based",
  colsample_bytree = 0.4,
  min_child_weight = 4,
  gamma = 2,
  eval_metric = "logloss"
)

test_preds <- data.table()
submit_preds <- data.table()

for (fold in 1:5) {
  
  targets_dt[, split := "skip"]
  
  targets_dt[
    buildingblock1_smiles %chin% bb1[fold_bb == fold, bb] &
      buildingblock2_smiles %chin% bb2[fold_bb == fold, bb] &
      buildingblock3_smiles %chin% bb3[fold_bb == fold, bb],
    split := "val_nonshare"
  ]
  
  set.seed(1)
  targets_dt[!
   (buildingblock1_smiles %chin% bb1[fold_bb == fold, bb] |
      buildingblock2_smiles %chin% bb2[fold_bb == fold, bb] |
      buildingblock3_smiles %chin% bb3[fold_bb == fold, bb]),
   split := "train"
  ]
  
  train_idx <- which(targets_dt[, split == "train"])
  val_idx <- which(targets_dt[, split == "val_nonshare"])
  
  oof <- targets_dt[val_idx, c(proteins, "split"), with = FALSE]
  oof[, fold_num := fold]
  
  submit <- copy(test_dt[, .(molecule_smiles)])
  submit[, fold_num := fold]
  
  for (p in proteins) {
    
    cat("\n\n===============", p, "===============\n\n")
    
    setinfo(dmat_all, name = "label", targets_dt[[p]])
    
    evallist <- list(train = xgb.slice.DMatrix(dmat_all, train_idx),
                     test = xgb.slice.DMatrix(dmat_all, val_idx))
    print(evallist)
    
    set.seed(1)
    bst <- xgb.train(
      params = params_list,
      nrounds = nrounds,
      early_stopping_rounds = early_stop_rnds,
      data = evallist$train,
      evals = evallist,
      verbose = 1,
      print_every_n = 50,
      seed_per_iteration = TRUE
    )
    
    cat("\nbest_iteration:", xgb.attr(bst, "best_iteration"))
    cat("\nbest_score:", xgb.attr(bst, "best_score"))
    
    # Save oof preds
    p1 = predict(bst, xgb.slice.DMatrix(dmat_all, val_idx))    
    oof[, (paste0(p, "_pred")) := p1]
    
    # Predict for submit
    submit[, (p) := predict(bst, test_features)]
    
    rm(bst, evallist) ; gc()
  }
  test_preds <- rbind(test_preds, oof)
  submit_preds <- rbind(submit_preds, submit)
}

# Evaluate

folds_list <- split(test_preds, by = "fold_num")

scores <- lapply(folds_list, function(fold_dt) {
  average_precision_belka(fold_dt)
})
scores <- rbindlist(scores, idcol = "fold")
scores[, MAP := round(MAP, 4)]
dcast(scores, fold ~ split + protein, value.var = "MAP")
scores[, .(MAP_total = mean(MAP)), by = "protein"]

# Prepare submit

submit_preds <- melt(
  submit_preds, measure.vars = proteins,
  variable.name = "protein_name",
  value.name = "binds",
  variable.factor = FALSE)

submit_preds <- submit_preds[
  , .(binds = mean(binds)),
  by = c("molecule_smiles", "protein_name")
]

submit <- submit_preds[
  test_dt[, .(id, molecule_smiles, protein_name)],
  on = c("molecule_smiles", "protein_name")
]
submit <- submit[, .(id, binds)]
head(submit)

fwrite(submit, "../submits/XGb_ecfp_1024_train_wide_split_by_bb_5f_nonshare_val.csv")