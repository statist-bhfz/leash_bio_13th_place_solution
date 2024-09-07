import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

tqdm.pandas()

PARALLEL = False # set to True if parallel computing is available

pd.read_csv("../data/test.csv").to_parquet("../data/submit.parquet", index = False)

FILES = [
  "../data/train_no_test_wide",
  "../data/test_ensemble_wide", 
  "../data/submit"
]

if PARALLEL:
    from pandarallel import pandarallel
    pandarallel.initialize(nb_workers = 16, progress_bar = True)


def replace_dy(x: str) -> str: 
    x = AllChem.ReplaceSubstructs(
        Chem.MolFromSmiles(x), 
        Chem.MolFromSmiles('[Dy]'), 
        Chem.MolFromSmiles('C')
    )[0]
    return(Chem.MolToSmiles(x))

for file in FILES:
    df = pd.read_parquet(f"{file}.parquet")
    if PARALLEL:
        df.molecule_smiles = df.molecule_smiles.parallel_apply(replace_dy)
    else:
        df.molecule_smiles = df.molecule_smiles.progress_apply(replace_dy)

    df.to_parquet(f"{file}_replace_dy.parquet", index = False)
