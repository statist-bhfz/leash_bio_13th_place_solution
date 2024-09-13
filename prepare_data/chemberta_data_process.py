from pathlib import Path
import polars as pl
import gc
import random
from transformers import AutoTokenizer
from rdkit import Chem
from rdkit.Chem import AllChem
import datasets

PROTEIN_NAMES = ["BRD4", "HSA", "sEH"]
data_dir = Path("/home/sato/kag/LeashBio/input")
model_name = "ChemBERTa-77M-MTR"
seed = 42

# Link to post-processing dataset


def datasplit_and_tokenize(data_dir, train_num_proc=1, valid_num_proc=5, seed=42):

    data = pl.read_parquet(Path(data_dir, "train.reduced.parquet"))
    random.seed(seed)
    gc.collect()

    bb1ids = list(data['buildingblock1_id'])
    bb2ids = list(data['buildingblock2_id'])
    bb3ids = list(data['buildingblock3_id'])

    group1 = list(set(bb1ids))
    group2 = list(set(bb2ids) & set(bb3ids))
    group3 = list(set(bb3ids) - set(bb2ids))

    bbs1 = random.sample(group1, 17)
    bbs2 = random.sample(group2, 34)
    bbs3 = random.sample(group3, 2)

    train = data.filter(~pl.col("buildingblock1_id").is_in(bbs1) &
                        ~pl.col("buildingblock2_id").is_in(bbs2) &
                        ~pl.col("buildingblock3_id").is_in(bbs3)).drop(['buildingblock1_id', 'buildingblock2_id', 'buildingblock3_id'])

    valid = data.filter(pl.col("buildingblock1_id").is_in(bbs1) |
                        pl.col("buildingblock2_id").is_in(bbs2) |
                        pl.col("buildingblock3_id").is_in(bbs3)).drop(['buildingblock1_id', 'buildingblock2_id', 'buildingblock3_id'])

    print(train.shape, valid.shape)
    del data, bbs1, bbs2, bbs3, group1, group2, group3
    gc.collect()

    train.write_parquet(Path(data_dir, f"train_split_{seed}.parquet"))
    valid.write_parquet(Path(data_dir, f"valid_split_{seed}.parquet"))

    print('data split done')

    m_Dy, m_C = Chem.MolFromSmiles("[Dy]"), Chem.MolFromSmiles("C")

    def proc_smile(df):
        m = Chem.MolFromSmiles(df['molecule_smiles'])
        m = AllChem.ReplaceSubstructs(m, m_Dy, m_C)[0]
        return {'fixed_smiles': Chem.MolToSmiles(m)}

    tokenizer = AutoTokenizer.from_pretrained("DeepChem/" + model_name)

    def tokenize(batch, tokenizer):
        return tokenizer(batch['fixed_smiles'], truncation=True)

    # Train data processing and tokenizing
    train = pl.read_parquet(Path(data_dir, f"train_split_{seed}.parquet"), columns=['molecule_smiles'])
    processed_train = (
        datasets.Dataset
        .from_pandas(train.to_pandas())
        .map(proc_smile, num_proc=1)
        .map(tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer}, num_proc=1)
        .to_pandas()
    )
    processed_train.to_parquet(Path(data_dir, f"train_tokenized_77M-MTR_replaced_dy.parquet"))
    del train, processed_train
    gc.collect()

    # Validation data processing and tokenizing
    valid = pl.read_parquet(Path(data_dir, f"valid_split_{seed}.parquet"))
    processed_valid = (
        datasets.Dataset
        .from_pandas(valid.to_pandas())
        .map(proc_smile, num_proc=1)
        .map(tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer}, num_proc=1)
        .to_pandas()
    )
    processed_valid.to_parquet(Path(data_dir, f"valid_tokenized_77M-MTR_replaced_dy.parquet"))

if __name__ == '__main__':
    datasplit_and_tokenize(data_dir, train_num_proc=1, valid_num_proc=1, seed=2024)
