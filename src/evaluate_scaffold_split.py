import torch
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from src.model import GNN
from src.utils import smile_to_data
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from collections import defaultdict

# Hyperparameters
EPOCHS = 100
SEED = 42
HIDDEN_DIM = 160
LEARNING_RATE = 0.00014067650554894532
WEIGHT_DECAY = 0.000205739489609305
BATCH_SIZE = 32

DATA_URL = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except:
        return None

def scaffold_kfold_split(smiles_list, n_splits=5, seed=42):
    np.random.seed(seed)
    
    # Group indices by Scaffold
    scaffold_to_indices = defaultdict(list)
    for idx, smiles in enumerate(smiles_list):
        scaffold = get_scaffold(smiles)
        if scaffold is not None:
            scaffold_to_indices[scaffold].append(idx)
            
    # Sort scaffolds by size to help balance the folds
    sorted_scaffolds = sorted(scaffold_to_indices.items(), 
                              key=lambda x: len(x[1]), reverse=True)
    
    # Initialize K empty folds
    folds = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    
    # Assign scaffolds to the smallest fold
    for scaffold, indices in sorted_scaffolds:
        smallest_fold_idx = np.argmin(fold_sizes)
        folds[smallest_fold_idx].extend(indices)
        fold_sizes[smallest_fold_idx] += len(indices)
        
    # Convert to (Train, Test) tuples for each fold
    cv_splits = []
    for i in range(n_splits):
        test_indices = folds[i]
        train_indices = []
        for j in range(n_splits):
            if i != j:
                train_indices.extend(folds[j])
        cv_splits.append((train_indices, test_indices))
        
    return cv_splits

if __name__ == "__main__":
    # Load data
    print("Loading Data...")
    df = pd.read_csv(DATA_URL)
    data_list = []
    smiles_list = []
    
    for _, row in df.iterrows():
        data = smile_to_data(row['smiles'], row['measured log solubility in mols per litre'])
        if data is not None:
            data_list.append(data)
            smiles_list.append(row['smiles'])
    
    print(f"Loaded {len(data_list)} molecules.")
    print(f"Running {5}-Fold Scaffold CV...\n")
    
    # Generate the 5 unique splits
    splits = scaffold_kfold_split(smiles_list, n_splits=5, seed=SEED)
    
    rmse_scores, r2_scores = [], []

    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"=== Fold {fold+1}/5 ===")
        print(f"Train Size: {len(train_idx)} | Test Size: {len(test_idx)}")
        
        # Subset data
        train_data = [data_list[i] for i in train_idx]
        test_data = [data_list[i] for i in test_idx]
        
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
        
        # Re-init model
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
        pearson_corr, _ = pearsonr(actuals, preds)
        
        print(f"Result: RMSE = {rmse:.4f} | R² = {r2:.4f} | Pearson r = {pearson_corr:.4f}\n")
        rmse_scores.append(rmse)
        r2_scores.append(r2)
    
    print("=== FINAL RESULTS ===")
    print(f"Average RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    print(f"Average R²:   {np.mean(r2_scores):.4f}")