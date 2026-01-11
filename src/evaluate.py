import torch
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from src.model import GNN
from src.utils import smile_to_data
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

# Hyperparameters
EPOCHS = 100
SEED = 42
HIDDEN_DIM = 160
LEARNING_RATE = 0.00014067650554894532
WEIGHT_DECAY = 0.000205739489609305
BATCH_SIZE = 32

DATA_URL = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    # Load data
    df = pd.read_csv(DATA_URL)
    data_list = []
    for _, row in df.iterrows():
        data = smile_to_data(row['smiles'], row['measured log solubility in mols per litre'])
        if data is not None:
            data_list.append(data)
    
    print(f"Loaded {len(data_list)} molecules\n")
    print("Running 5-Fold Cross Validation...\n")
    
    # 5-Fold CV
    rmse_scores, r2_scores = [], []
    kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(data_list)):
        train_data = [data_list[i] for i in train_idx]
        test_data = [data_list[i] for i in test_idx]
        
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
        
        model = GNN(hidden_dim=HIDDEN_DIM).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        loss_fn = torch.nn.MSELoss()
        
        # Train
        model.train()
        for epoch in range(EPOCHS):
            for batch in train_loader:
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                out = model(batch)
                loss = loss_fn(out.squeeze(), batch.y)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        preds, actuals = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(DEVICE)
                pred = model(batch)
                preds.append(pred.view(-1).cpu().numpy())
                actuals.append(batch.y.view(-1).cpu().numpy())
        
        preds = np.concatenate(preds)
        actuals = np.concatenate(actuals)
        
        rmse = np.sqrt(mean_squared_error(actuals, preds))
        r2 = r2_score(actuals, preds)
        
        print(f"Fold {fold+1}: RMSE = {rmse:.4f} | R² = {r2:.4f}")
        rmse_scores.append(rmse)
        r2_scores.append(r2)
    
    print(f"\nAverage RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    print(f"Average R²:   {np.mean(r2_scores):.4f}")