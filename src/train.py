import pandas as pd
import random
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from .model import GNN
from .utils import smile_to_data

# Hyperparameters
EPOCHS = 100
SEED = 42
HIDDEN_DIM = 160
LEARNING_RATE = 0.00014067650554894532
WEIGHT_DECAY = 0.000205739489609305
BATCH_SIZE = 32

# Set random seeds
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = loss_fn(out.squeeze(), batch.y)
            total_loss += loss.item()
    return total_loss / max(len(loader), 1)


if __name__ == "__main__":
    # Load Delaney solubility dataset
    DATA_URL = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
    df = pd.read_csv(DATA_URL)

    # Extract SMILES strings and solubility values
    data_list = []
    for _, row in df.iterrows():
        data = smile_to_data(row['smiles'], row['measured log solubility in mols per litre'])
        if data is not None:
            data_list.append(data)

    # Train/val/test split
    num_samples = len(data_list)
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)
    test_size = num_samples - train_size - val_size
    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set, test_set = torch.utils.data.random_split(
        data_list, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNN(hidden_dim=HIDDEN_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=0.00001
    )
    loss_fn = torch.nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out.squeeze(), batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, device, loss_fn)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.6f}")

    # Final eval
    final_val = evaluate(model, val_loader, device, loss_fn)
    final_test = evaluate(model, test_loader, device, loss_fn)
    print(f"Validation Loss: {final_val:.4f} | Test Loss: {final_test:.4f}")

    torch.save(model.state_dict(), "models/gnn_solubility.pth")
