import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader

from .config import (
    BATCH_SIZE,
    EPOCHS,
    HIDDEN_DIM,
    LEARNING_RATE,
    SEED,
    WEIGHT_DECAY,
    get_device,
)
from .data import load_dataset
from .model import GNN

DEVICE = get_device()

if __name__ == "__main__":
    # Load data
    data_list, _ = load_dataset()

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
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
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
        pearson_corr, _ = pearsonr(actuals, preds)

        print(
            f"Fold {fold + 1}: RMSE = {rmse:.4f} | R² = {r2:.4f} | Pearson r = {pearson_corr:.4f}"
        )
        rmse_scores.append(rmse)
        r2_scores.append(r2)

    print(f"\nAverage RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    print(f"Average R²:   {np.mean(r2_scores):.4f}")
