from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score

from .config import SEED, get_device
from .data import load_dataset
from .ensemble import ensemble_predict_dataset, fit_ensemble

DEVICE = get_device()


def get_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return None


def scaffold_kfold_split(smiles_list, n_splits=5, seed=42):
    np.random.seed(seed)

    # Group indices by Scaffold
    scaffold_to_indices = defaultdict(list)
    for idx, smiles in enumerate(smiles_list):
        scaffold = get_scaffold(smiles)
        if scaffold is not None:
            scaffold_to_indices[scaffold].append(idx)

    # Sort scaffolds by size
    sorted_scaffolds = sorted(
        scaffold_to_indices.items(), key=lambda x: len(x[1]), reverse=True
    )

    # Initialize K empty folds
    folds = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits

    # Assign scaffolds to the smallest fold
    for scaffold, indices in sorted_scaffolds:
        smallest_fold_idx = np.argmin(fold_sizes)
        folds[smallest_fold_idx].extend(indices)
        fold_sizes[smallest_fold_idx] += len(indices)

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
    data_list, smiles_list = load_dataset()

    print(f"Loaded {len(data_list)} molecules.")
    print(f"Running {5}-Fold Scaffold CV...\n")

    # Generate the 5 unique splits
    splits = scaffold_kfold_split(smiles_list, n_splits=5, seed=SEED)

    rmse_scores, r2_scores = [], []

    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"=== Fold {fold + 1}/5 ===")
        print(f"Train Size: {len(train_idx)} | Test Size: {len(test_idx)}")

        # Subset data
        train_data = [data_list[i] for i in train_idx]
        test_data = [data_list[i] for i in test_idx]

        # Hold out a validation slice of the training fold for early stopping.
        val_cut = int(0.85 * len(train_data))
        fit_data, val_data = train_data[:val_cut], train_data[val_cut:]

        models, scaler = fit_ensemble(fit_data, val_data, DEVICE)
        preds, actuals = ensemble_predict_dataset(models, scaler, test_data, DEVICE)

        rmse = np.sqrt(mean_squared_error(actuals, preds))
        r2 = r2_score(actuals, preds)
        pearson_corr, _ = pearsonr(actuals, preds)

        print(
            f"Result: RMSE = {rmse:.4f} | R² = {r2:.4f} | Pearson r = {pearson_corr:.4f}\n"
        )
        rmse_scores.append(rmse)
        r2_scores.append(r2)

    print("=== FINAL RESULTS ===")
    print(f"Average RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    print(f"Average R²:   {np.mean(r2_scores):.4f}")
