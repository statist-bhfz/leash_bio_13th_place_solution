library(arrow)
library(duckdb)
library(data.table)
library(dplyr)

# This script was run on RTX 3090, 124 Gb RAM
# Set working directory to script location
setwd("/home/tonia/BELKA/solution_GBDT_models_git/prepare_data")

# 1. Reshaping of initial train data to wide format

con <- dbConnect(duckdb::duckdb())
dt <- open_dataset("../data/train.parquet") |>
  to_duckdb(table_name = "train", con = con)

dbExecute(
  con, 
  "COPY(
    SELECT
      buildingblock1_smiles,
      buildingblock2_smiles,
      buildingblock3_smiles,
      molecule_smiles,
      MAX(CASE WHEN (protein_name = 'sEH') THEN binds END) AS sEH,
      MAX(CASE WHEN (protein_name = 'HSA') THEN binds END) AS HSA,
      MAX(CASE WHEN (protein_name = 'BRD4') THEN binds END) AS BRD4
    FROM (
      SELECT
        buildingblock1_smiles,
        buildingblock2_smiles,
        buildingblock3_smiles,
        molecule_smiles,
        protein_name,
        binds
      FROM train
    ) q01
    GROUP BY
      buildingblock1_smiles,
      buildingblock2_smiles,
      buildingblock3_smiles,
      molecule_smiles)
  TO '../data/train_wide.parquet' 
  (field_ids 'auto')
")
dbDisconnect(con)

# 2. Make 50M subset
# for XGb_ecfp1024_plus_BB_act_50m_v2_new_bb_XGb_ecfp1024_574 model

dt <- read_parquet("../data/train_wide.parquet")
setDT(dt)

proteins <- c("BRD4", "HSA", "sEH")
binds_any <- dt[rowSums(dt[, ..proteins]) > 0, .N]

set.seed(1)
train_subset <- rbind(
  dt[rowSums(dt[, ..proteins]) > 0],
  dt[rowSums(dt[, ..proteins]) == 0][sample(.N, 50e6-binds_any)]
)

write_parquet(train_subset, "../data/train_wide_50M.parquet")
rm(train_subset); gc()

# 3. Split to test and train without overlap by building blocks
# for GBDT/XGb_ecfp_1024_train_wide_split_by_bb_5f_nonshare_val model

dt <- read_parquet("../data/train_wide.parquet")
setDT(dt)

# Random sample 20% of buildingblock1_smiles, 
# buildingblock2_smiles and buildingblock3_smiles
frac <- 0.2
set.seed(1)

bb1 <- dt[, unique(buildingblock1_smiles)]
bb1_val <- sample(bb1, round(length(bb1) * frac))

bb2 <- dt[, unique(buildingblock2_smiles)]
bb2_val <- sample(bb2, round(length(bb2) * frac))

bb3 <- dt[, unique(buildingblock3_smiles)]
bb3_val <- sample(bb3, round(length(bb3) * frac))

# Subset samples with only validation building blocks
dt_val <- dt[
  buildingblock1_smiles %chin% bb1_val &
  buildingblock2_smiles %chin% bb2_val &
  buildingblock3_smiles %chin% bb3_val
]

# Shuffle and divide into validation and test subsets
# that do not overlap by building blocks with the train data
set.seed(1)
dt_val <- dt_val[sample(.N, .N)]
dt_val[, split := sample(
  c("val_nonshare", "test_nonshare"), .N, replace = TRUE, prob = c(0.5, 0.5)
  )
]

# Subset samples without validation building blocks
dt_train <- dt[!
  (buildingblock1_smiles %chin% bb1_val |
   buildingblock2_smiles %chin% bb2_val |
   buildingblock3_smiles %chin% bb3_val)
]

# Shuffle and divide into validation, test, and train subsets
# that overlap by building blocks
set.seed(1)
dt_train <- dt_train[sample(.N, .N)]
dt_train[, 
  split := sample(
    c("train", "test", "val"), .N, replace = TRUE, prob = c(0.99, 0.005, 0.005)
    )
]

# Combine and save the dataset
dt_subset <- rbind(dt_train, dt_val)

setkey(dt_subset, NULL)
write_parquet(
  dt_subset,
  file.path("../data/train_wide_split_by_bb.parquet")
)
rm(dt_subset, dt_train, dt_val)

# 4. Divide into 5 parts, each without a subset of blocks
# for GBDT/XGb_ecfp_1024_5_bb_parts_BB_act_50M model

dt <- read_parquet("../data/train_wide.parquet")
setDT(dt)

bb1 <- unique(dt[, .(bb = buildingblock1_smiles)])
bb2 <- unique(dt[, .(bb = buildingblock2_smiles)])
bb3 <- unique(dt[, .(bb = buildingblock3_smiles)])

set.seed(1)
bb1[, fold_bb := sample(1:5, .N, replace = TRUE)]
bb2[, fold_bb := sample(1:5, .N, replace = TRUE)]
bb3[, fold_bb := sample(1:5, .N, replace = TRUE)]

for (fold in 1:5) {
  
  dt[, split := 0]
  dt[
    buildingblock1_smiles %chin% bb1[fold_bb == fold, bb] &
      buildingblock2_smiles %chin% bb2[fold_bb == fold, bb] &
      buildingblock3_smiles %chin% bb3[fold_bb == fold, bb],
    split := 0
  ]
  dt[! (
    buildingblock1_smiles %chin% bb1[fold_bb == fold, bb] |
      buildingblock2_smiles %chin% bb2[fold_bb == fold, bb] |
      buildingblock3_smiles %chin% bb3[fold_bb == fold, bb]
  ),
    split := 1
  ]
  
  write_parquet(
    dt[split == 1],
    paste0("../data/train_wide_bb_split_p", fold, ".parquet")
  )
}
