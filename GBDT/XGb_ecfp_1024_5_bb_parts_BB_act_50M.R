library(data.table)
library(qs)
library(yardstick)
library(xgboost)
library(Matrix)
library(arrow)
library(duckdb)
scipy_sparse = reticulate::import("scipy.sparse")

# This script was run on RTX 3090, 124 Gb RAM.

# To reproduce the XGb_ecfp_1024_5_bb_parts_BB_act_50M predictions,
# run it 5 times, each time using a subset of 5 
# created with make_splits_by_bb.R and its features.

# Set working directory to script location
setwd("/home/tonia/BELKA/solution_GBDT_models_git/GBDT")

# Prepare test data and features

test_dt <- fread("../data/test.csv")
test_features_path <- "../data/ecfp_1024_test.npz"
test_features <- scipy_sparse$load_npz(test_features_path)
test_bb_features <- read_parquet("../data/bb_features_for_test.parquet")

# Train models and predict

submit_preds <- data.table()

for (train_subset_num in 1:5) {

    # Data loading
    train_subset <- paste0("train_wide_bb_split_p", train_subset_num)
    targets_path <- paste0("../data/", train_subset, ".parquet")
    targets_dt <- as.data.table(read_parquet(targets_path))

    submit <- copy(test_dt[, .(molecule_smiles)])
    submit[, subset_num := train_subset_num]

    # Split into training and validation
    num_rows <- targets_dt[, .N]
    num_val = 200000
    num_train <- num_rows - num_val

    set.seed(1)
    indices <- sample(num_rows)
    train_idx <- indices[1:num_train]
    val_idx <- sample(indices[(num_train + 1):length(indices)], num_val)

    proteins = c("BRD4", "HSA", "sEH")
    for (p in proteins) {
        
        cat("\n\n===============", p, "===============\n\n")
    
        train_features_path <- paste0("../data/ecfp_1024_", train_subset, "_", p, ".buffer")
        dmat_all <- xgb.DMatrix(train_features_path)

        setinfo(dmat_all, name = "label", targets_dt[[p]])
    
        evallist <- list(train = xgb.slice.DMatrix(dmat_all, train_idx),
                        test = xgb.slice.DMatrix(dmat_all, val_idx))
        print(evallist)
    
        params <- list(
            device = "gpu",
            objective = "binary:logistic",
            eta = 0.05,
            max.depth = 25,
            subsample = 0.2,
            sampling_method = "gradient_based",
            colsample_bytree = 0.4,
            min_child_weight = 4,
            gamma = 2
        )
    
        set.seed(1)
        bst <- xgb.train(
            params,
            nrounds = 5000,
            early_stopping_rounds = 30,
            data = evallist$train,
            evals = evallist,
            verbose = 1,
            print_every_n = 50,
            seed_per_iteration = TRUE
        )
    
        cat("\nbest_iteration:", xgb.attr(bst, "best_iteration"))
        cat("\nbest_score:", xgb.attr(bst, "best_score"))
        
        rm(dmat_all, evallist) ; gc()

        # Predict for submit
        comb_test_features <- cbind(
            test_features,
            as.matrix(test_bb_features[
                , .SD, .SDcols = names(test_bb_features) %like% paste0("_", p)
                ])
            )
        comb_test_features <- xgb.DMatrix(as(comb_test_features, "sparseMatrix"))
        submit[, (p) := predict(bst, comb_test_features, iterationrange = NULL)]
    
    }
    submit_preds <- rbind(submit_preds, submit)
}

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

fwrite(submit, "../submits/XGb_ecfp_1024_5_bb_parts_BB_act_50M.csv")