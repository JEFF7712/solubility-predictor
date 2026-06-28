"""Optuna search over architecture + optimizer hyperparameters.

Uses the exact production pipeline (standardize features/target on the training
split, early-stopping training via train_single_model) so the best trial's
hyperparameters transfer directly to training. Validation loss is reported in
standardized space. Searches width, depth and dropout in addition to the
optimizer settings, since the larger AqSolDB dataset can support more capacity.
"""

import os

import optuna
import torch
from optuna.pruners import MedianPruner

from .config import (
    SEED,
    TRAIN_FRAC,
    VAL_FRAC,
    get_device,
    set_seed,
)
from .data import load_dataset
from .normalize import GraphStandardizer
from .training import train_single_model

N_TRIALS = int(os.environ.get("N_TRIALS", "50"))
EPOCHS = int(os.environ.get("TUNE_EPOCHS", "60"))

set_seed(SEED)
DEVICE = get_device()

# Featurize once, fit the standardizer once on the training split, and reuse
# the standardized splits across all trials.
DATA_LIST, _ = load_dataset()
_num = len(DATA_LIST)
_train_size = int(TRAIN_FRAC * _num)
_val_size = int(VAL_FRAC * _num)
_generator = torch.Generator().manual_seed(SEED)
_train_set, _val_set, _ = torch.utils.data.random_split(
    DATA_LIST,
    [_train_size, _val_size, _num - _train_size - _val_size],
    generator=_generator,
)
_scaler = GraphStandardizer.fit(list(_train_set))
TRAIN_STD = _scaler.transform_dataset(list(_train_set))
VAL_STD = _scaler.transform_dataset(list(_val_set))


def objective(trial):
    model_kwargs = {
        "hidden_dim": trial.suggest_int("hidden_dim", 64, 384, step=64),
        "num_layers": trial.suggest_int("num_layers", 2, 5),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
    }
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    print(
        f"\nTrial {trial.number + 1}/{N_TRIALS}: {model_kwargs}, "
        f"lr={lr:.2e}, wd={weight_decay:.2e}, bs={batch_size}"
    )

    _, best_val_loss = train_single_model(
        TRAIN_STD,
        VAL_STD,
        DEVICE,
        model_kwargs=model_kwargs,
        lr=lr,
        weight_decay=weight_decay,
        epochs=EPOCHS,
        batch_size=batch_size,
        seed=SEED,
    )

    print(f"  Best Val Loss (std space): {best_val_loss:.4f}")
    return best_val_loss


if __name__ == "__main__":
    print(f"Starting hyperparameter search: {N_TRIALS} trials, {EPOCHS} epochs each")

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(
        direction="minimize", sampler=sampler, pruner=MedianPruner()
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    print(f"\nBest Trial: {study.best_trial.number}")
    print(f"Best Validation Loss (std space): {study.best_value:.4f}\n")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
