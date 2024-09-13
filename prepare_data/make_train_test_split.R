# All competition files should be in data folder

# Long to wide format transformation---------------------------------------

library(arrow)
library(duckdb)
library(dplyr)
library(data.table)

con <- dbConnect(duckdb::duckdb())
train <- open_dataset("../data/train.parquet") %>%
  select(molecule_smiles, protein_name, binds) %>%
  to_duckdb(table_name = "train", con = con)

dbSendQuery(
  con, 
  "COPY(
  SELECT
    molecule_smiles , 
    max(case when protein_name = 'BRD4' then binds else null end) as BRD4, 
    max(case when protein_name = 'HSA' then binds else null end) as HSA, 
    max(case when protein_name = 'sEH' then binds else null end) as sEH
  FROM train
  GROUP BY molecule_smiles) 
  TO '../data/train_wide.parquet' 
  (field_ids 'auto')"
)
gc()

# Train-test split --------------------------------------------------------

dt <- read_parquet("../data/train_wide.parquet")
setDT(dt)

set.seed(42)
test_ids <- sample(1:nrow(dt), 2e6)
dt_test <- dt[test_ids]
dt <- dt[!test_ids]
write_parquet(dt_test, "../data/test_ensemble_wide.parquet")
write_parquet(dt, "../data/train_no_test_wide.parquet")

# Make 50M subset ---------------------------------------------------------
# for XGb_secfp1024_mpnn_che2_10Mtnt_last15_by_prot_train_no_test_wide model

proteins <- c("BRD4", "HSA", "sEH")
binds_any <- dt[rowSums(dt[, ..proteins]) > 0, .N]

set.seed(1)
train_subset <- rbind(
  dt[rowSums(dt[, ..proteins]) > 0],
  dt[rowSums(dt[, ..proteins]) == 0][sample(.N, 50e6-binds_any)]
)

write_parquet(train_subset, "../data/train_no_test_wide_50M.parquet")

# Make 10M subset ---------------------------------------------------------
# for chemprop

proteins <- c("BRD4", "HSA", "sEH")
binds_any <- dt[rowSums(dt[, ..proteins]) > 0, .N]

set.seed(1)
train_subset <- rbind(
  dt[rowSums(dt[, ..proteins]) > 0],
  dt[rowSums(dt[, ..proteins]) == 0][sample(.N, 10e6-binds_any)]
)
fwrite(train_subset[, .(molecule_smiles, BRD4, HSA, sEH)],
       "../data/train_no_test_wide_10M.csv")
