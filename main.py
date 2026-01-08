import numpy as np
import pandas as pd
from rdkit import Chem
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_rdmol

DATA_URL = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
df = pd.read_csv(DATA_URL)

smiles_data = df["smiles"].values
solubilities_data = df["measured log solubility in mols per litre"].values
print(f"Number of samples: {len(smiles_data)}")
print(f"First SMILES: {smiles_data[0]}")
print(f"First solubility value: {solubilities_data[0]}")

def mol_to_edge_index(mol):
    data = from_rdmol(mol)
    return data.edge_index

allowed_atoms = ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I']

def get_atom_features(atom):

    # Get atom symbol as one-hot encoding
    symbol_one_hot = [0] * len(allowed_atoms)
    if atom.GetSymbol() in allowed_atoms:
        symbol_one_hot[allowed_atoms.index(atom.GetSymbol())] = 1

    # Get hybridization as one-hot encoding
    hybridization = [0, 0, 0] 
    hyb = atom.GetHybridization()
    if hyb == Chem.HybridizationType.SP: 
        hybridization[0] = 1
    elif hyb == Chem.HybridizationType.SP2: 
        hybridization[1] = 1
    elif hyb == Chem.HybridizationType.SP3: 
        hybridization[2] = 1

    # Keep mass between 0 and 1 (most of the time)
    mass = [atom.GetMass() / 100.0]

    # Combine all features into a single list
    return symbol_one_hot + hybridization + mass

def get_bond_features(bond):
    bt = bond.GetBondType()
    
    features = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC
    ]

    return [float(f) for f in features]

def smile_to_data(smile, y_value=None):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None

    node_features = []
    edge_attributes = []

    for atom in mol.GetAtoms():
        feature = get_atom_features(atom)
        node_features.append(feature)
    x = torch.tensor(node_features, dtype=torch.float)

    for bond in mol.GetBonds():
        feature = get_bond_features(bond)
        edge_attributes.append(feature)
        edge_attributes.append(feature)
    edge_attr = torch.tensor(edge_attributes, dtype=torch.float)

    edge_index = mol_to_edge_index(mol)

    if y_value is not None:
        y = torch.tensor([y_value], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    else:
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

sample_smile = smiles_data[0]
print(smile_to_data(sample_smile))