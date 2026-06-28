import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from .config import SEED, get_device
from .data import load_dataset
from .ensemble import ensemble_predict_dataset, fit_ensemble

DEVICE = get_device()

if __name__ == "__main__":
    data_list, _ = load_dataset()

    print(f"Loaded {len(data_list)} molecules\n")
    print("Running 5-Fold Cross Validation (normalized deep ensemble)...\n")

    rmse_scores, r2_scores = [], []
    kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)

    for fold, (train_idx, test_idx) in enumerate(kfold.split(data_list)):
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
            f"Fold {fold + 1}: RMSE = {rmse:.4f} | R² = {r2:.4f} | "
            f"Pearson r = {pearson_corr:.4f}"
        )
        rmse_scores.append(rmse)
        r2_scores.append(r2)

    print(f"\nAverage RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    print(f"Average R²:   {np.mean(r2_scores):.4f}")
