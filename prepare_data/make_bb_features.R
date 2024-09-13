library(data.table)
library(arrow)

# This script was run on RTX 3090, 124 Gb RAM

# Make building blocks (BB) features (BB_act)
# - the fraction of compounds that binds when a given
# BB smiles occurs in a particular position

# Set working directory to script location
setwd("/home/tonia/BELKA/solution_GBDT_models_git/prepare_data")

# Make BB features based on the train data

dt <- read_parquet(paste0("../data/train_wide.parquet"))
setDT(dt)

bb_act <- melt(
  dt[, !"split"], measure.vars = c("BRD4", "HSA", "sEH"),
  variable.factor = FALSE,
  variable.name = "protein",
  value.name = "binds"
)

bb_act <- melt(
  bb_act, measure.vars = c("buildingblock1_smiles",
                           "buildingblock2_smiles",
                           "buildingblock3_smiles"),
  variable.factor = FALSE,
  variable.name = "BB",
  value.name = "smiles"
)
bb_act[, num_a := sum(binds), by = c( "protein", "BB", "smiles")]
bb_act[, n_mols := uniqueN(molecule_smiles), by = c("BB", "smiles")]
bb_act[, p_active := num_a/n_mols]

bb_act[, c("molecule_smiles", "binds", "num_a", "n_mols") := NULL]
gc()

bb_act <- unique(bb_act)
gc()

bb_act <- dcast(bb_act, ... ~ protein, value.var = "p_active")
proteins <- c("BRD4", "HSA", "sEH")
setnames(bb_act, proteins, paste0(proteins, "_TE"))

# Merge BB features with a selected train subset

train_subset <- "train_wide_50M"

dt <- read_parquet(paste0("../data/", train_subset, ".parquet"))

bb_dt <- lapply(c("buildingblock1_smiles",
                  "buildingblock2_smiles",
                  "buildingblock3_smiles"), function(bb_col) {
  
  bb <- bb_act[BB == bb_col, !"BB"]
  setnames(bb,  "smiles", bb_col)
  bb <- bb[dt[, ..bb_col], on = bb_col]
  bb[, (bb_col) := NULL]
  
  setnames(bb, names(bb),
           paste0("bb", gsub("\\D", "", bb_col), "_", names(bb))
  )
  bb
})
bb_dt <- do.call(cbind, bb_dt)

write_parquet(
  bb_dt,
  paste0("../data/bb_features_for_", train_subset, ".parquet"),
  chunk_size = 1e6
)

# Make BB features for the test data

dt <- fread("../data/test.csv")

bb_dt <- lapply(c("buildingblock1_smiles",
                  "buildingblock2_smiles",
                  "buildingblock3_smiles"), function(bb_col) {
  
  bb <- bb_act[BB == bb_col, !"BB"]
  setnames(bb,  "smiles", bb_col)
  bb <- bb[dt[, ..bb_col], on = bb_col]
  bb[, (bb_col) := NULL]
  
  setnames(bb, names(bb),
           paste0("bb", gsub("\\D", "", bb_col), "_", names(bb))
  )
  bb
})

bb_dt <- do.call(cbind, bb_dt)

write_parquet(bb_dt, "../data/bb_features_for_test.parquet")