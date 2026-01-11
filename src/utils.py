import torch
from rdkit import Chem
from torch_geometric.data import Data

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

    # Get aromaticity
    aromatic = [1 if atom.GetIsAromatic() else 0]

    # Get charge
    charge = [atom.GetFormalCharge()]

    # Get number of hydrogens
    num_hydrogens = [atom.GetTotalNumHs()]

    # Get degree
    degree = [atom.GetDegree()]

    # Get mass and keep between 0 and 1
    mass = [atom.GetMass() / 100.0]

    # Combine all features into a single list
    return symbol_one_hot + hybridization + aromatic + charge + num_hydrogens + degree + mass

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
    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smile)
    if mol is None: return None

    # Node Features
    x_list = []
    for atom in mol.GetAtoms():
        x_list.append(get_atom_features(atom))
    x = torch.tensor(x_list, dtype=torch.float)

    # Edges & Attributes
    src_list = []
    dst_list = []
    attr_list = []

    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = get_bond_features(bond)

        src_list.append(start)
        dst_list.append(end)
        attr_list.append(feat)

        src_list.append(end)
        dst_list.append(start)
        attr_list.append(feat)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(attr_list, dtype=torch.float)

    # Create Data object
    if y_value is not None:
        y = torch.tensor([y_value], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    else:
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)