import pandas as pd
import random
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from .model import GNN
from .utils import smile_to_data

EPOCHS = 100
SEED = 42

# Set random seeds
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Model
    model = GNN(hidden_dim=128).to(device)
    # Optimizer
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

    torch.save(model.state_dict(), "gnn_solubility.pth")
