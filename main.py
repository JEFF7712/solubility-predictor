import numpy as np
import pandas as pd


DATA_URL = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
df = pd.read_csv(DATA_URL)

smiles = df["smiles"].values
sols = df["measured log solubility in mols per litre"].values
print(f"Number of samples: {len(smiles)}")
print(f"First SMILES: {smiles[0]}")
print(f"First solubility value: {sols[0]}")