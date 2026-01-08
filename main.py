import numpy as np
import pandas as pd
from rdkit import Chem
import torch
from torch.nn import Sequential, Linear, ReLU, Dropout, BatchNorm1d
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_rdmol
from torch_geometric.nn import GINEConv, global_mean_pool
import matplotlib.pyplot as plt

EPOCHS = 100

# Define allowed atom types
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

class GNN(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        
        # 1. Edge Encoder (4 -> 12)
        self.edge_encoder = Linear(4, 12)
        
        # 2. GINE Layer 1
        mlp1 = Sequential(Linear(12, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv1 = GINEConv(mlp1)
        self.bn1 = BatchNorm1d(hidden_dim)
        
        # 3. GINE Layer 2
        self.edge_encoder2 = Linear(12, hidden_dim) 
        mlp2 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv2 = GINEConv(mlp2)
        self.bn2 = BatchNorm1d(hidden_dim)

        # 4. GINE Layer 3
        self.edge_encoder3 = Linear(hidden_dim, hidden_dim)
        self.edge_lin = Linear(12, hidden_dim)

        mlp3 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv3 = GINEConv(mlp3)
        self.bn3 = BatchNorm1d(hidden_dim)

        # 5. Prediction Head
        self.head = Sequential(Linear(hidden_dim, hidden_dim // 2), ReLU(), Dropout(0.5), Linear(hidden_dim // 2, 1))
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Layer 1
        edge_attr_1 = self.edge_encoder(edge_attr)
        x = self.conv1(x, edge_index, edge_attr_1)
        x = self.bn1(x)
        x = x.relu()
        
        # Prepare edges for Layer 2 & 3 (Needs to match hidden_dim=64)
        edge_attr_deep = self.edge_lin(edge_attr_1)
        
        # Layer 2
        x = self.conv2(x, edge_index, edge_attr_deep)
        x = self.bn2(x)
        x = x.relu()
        
        # Layer 3
        x = self.conv3(x, edge_index, edge_attr_deep)
        x = self.bn3(x)
        x = x.relu()

        # Global Pooling
        x = global_mean_pool(x, batch)
        
        # Final Prediction
        return self.head(x)


# Load the Delaney solubility dataset
DATA_URL = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
df = pd.read_csv(DATA_URL)

# Extract SMILES strings and solubility values
data_list = []
for index, row in df.iterrows():
    data = smile_to_data(row['smiles'], row['measured log solubility in mols per litre'])
    if data is not None:
        data_list.append(data)

train_loader = DataLoader(data_list[:900], batch_size=64, shuffle=True)
test_loader = DataLoader(data_list[900:], batch_size=64, shuffle=False)

# Training loop

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
model = GNN(hidden_dim=128).to(device)

# Optimizer - Force weights to be smaller to help with convergence
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=0.00001)

# Loss Function
loss_fn = torch.nn.MSELoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out.squeeze(), batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    
    scheduler.step(avg_loss)
    current_lr = optimizer.param_groups[0]['lr']

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

# Evaluation on test set
model.eval()
test_loss = 0
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        pred = model(batch)
        test_loss += loss_fn(pred.squeeze(), batch.y).item()

print(f"\nFinal Test MSE: {test_loss / len(test_loader):.4f}")
print("Sample Prediction (Actual vs Pred):")
print(f"{batch.y[0].item():.2f} vs {pred[0].item():.2f}")

torch.save(model.state_dict(), "gnn_solubility.pth")