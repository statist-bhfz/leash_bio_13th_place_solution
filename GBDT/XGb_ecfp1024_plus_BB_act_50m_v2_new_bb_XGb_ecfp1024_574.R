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

# Prepare train data

targets_path <- "../data/train_wide_50M.parquet"
targets_dt <- as.data.table(read_parquet(targets_path))

# Split into training and validation
num_rows <- targets_dt[, .N]
num_val = 200000
num_train <- num_rows - num_val

set.seed(1)
indices <- sample(num_rows)
train_idx <- indices[1:num_train]
val_idx <- sample(indices[(num_train + 1):length(indices)], num_val)

# Prepare test data and features

test_dt <- fread("../data/test.csv")
test_features_path <- "../data/ecfp_1024_test.npz"
test_features <- scipy_sparse$load_npz(test_features_path)
test_bb_features <- read_parquet("../data/bb_features_for_test.parquet")

# Train models and predict

submit <- copy(test_dt[, .(molecule_smiles)])

proteins = c("BRD4", "HSA", "sEH")
for (p in proteins) {
    
    cat("\n\n===============", p, "===============\n\n")
    
    train_features_path <- paste0("../data/ecfp_1024_train_wide_50M_", p,".buffer")
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

# Prepare submit

submit_preds <- melt(
    submit, measure.vars = proteins,
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

fwrite(submit, "../submits/XGb_ecfp1024_plus_BB_act_50m_v2_new_bb_XGb_ecfp1024_574.csv")