import numpy as np
import torch

from .config import (
    ENSEMBLE_MODEL_PATH,
    ENSEMBLE_SIZE,
    SEED,
    TRAIN_FRAC,
    VAL_FRAC,
    default_model_kwargs,
    get_device,
    set_seed,
)
from .data import load_dataset
from .ensemble import (
    ensemble_predict,
    load_ensemble,
    save_ensemble,
    train_ensemble,
)
from .normalize import GraphStandardizer

set_seed(SEED)


def evaluate_ensemble_rmse(models, scaler, dataset, device):
    """RMSE in logS space over `dataset` using the full ensemble mean."""
    errors = []
    for data in dataset:
        d = data.clone()
        d.batch = torch.zeros(d.x.shape[0], dtype=torch.long)
        mean, _ = ensemble_predict(models, scaler, d, device)
        errors.append((mean - data.y.item()) ** 2)
    return float(np.sqrt(np.mean(errors)))


def main():
    data_list, _ = load_dataset()

    num_samples = len(data_list)
    train_size = int(TRAIN_FRAC * num_samples)
    val_size = int(VAL_FRAC * num_samples)
    test_size = num_samples - train_size - val_size
    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set, test_set = torch.utils.data.random_split(
        data_list, [train_size, val_size, test_size], generator=generator
    )

    # Fit the standardizer on the training split only, then standardize the
    # splits the ensemble actually trains on.
    scaler = GraphStandardizer.fit(list(train_set))
    train_std = scaler.transform_dataset(list(train_set))
    val_std = scaler.transform_dataset(list(val_set))

    device = get_device()
    model_kwargs = default_model_kwargs()
    print(f"Training {ENSEMBLE_SIZE}-model ensemble on {device} ({model_kwargs})...")
    states, _ = train_ensemble(train_std, val_std, device, model_kwargs=model_kwargs)
    save_ensemble(ENSEMBLE_MODEL_PATH, states, scaler, model_kwargs=model_kwargs)

    # Evaluate the saved ensemble on the held-out test split in logS space.
    models, loaded_scaler = load_ensemble(ENSEMBLE_MODEL_PATH, device)
    test_rmse = evaluate_ensemble_rmse(models, loaded_scaler, list(test_set), device)
    print(f"Test RMSE (logS): {test_rmse:.4f}")
    print(f"Saved ensemble to {ENSEMBLE_MODEL_PATH}")


if __name__ == "__main__":
    main()
