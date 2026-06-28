"""Deep-ensemble training, persistence, and prediction.

An ensemble of K independently-seeded models gives both a point estimate (the
mean) and a calibrated uncertainty (the spread across members). All members
share one standardizer, fit on the training split. The serialized artifact holds
every member's weights plus the standardizer state so inference is fully
reproducible.
"""

import torch

from .config import ENSEMBLE_SIZE, SEED, default_model_kwargs
from .model import GNN
from .normalize import GraphStandardizer
from .training import train_single_model


def train_ensemble(
    train_set,
    val_set,
    device,
    *,
    size=ENSEMBLE_SIZE,
    model_kwargs=None,
    base_seed=SEED,
    **train_kwargs,
):
    """Train `size` models on the (already standardized) data, one per seed.

    Returns (states, val_losses).
    """
    if model_kwargs is None:
        model_kwargs = default_model_kwargs()
    states, val_losses = [], []
    for i in range(size):
        state, val = train_single_model(
            train_set,
            val_set,
            device,
            model_kwargs=model_kwargs,
            seed=base_seed + i,
            **train_kwargs,
        )
        states.append(state)
        val_losses.append(val)
        print(f"  ensemble member {i + 1}/{size}: best val (std space) = {val:.4f}")
    return states, val_losses


def fit_ensemble(
    train_data,
    val_data,
    device,
    *,
    size=ENSEMBLE_SIZE,
    model_kwargs=None,
    **train_kwargs,
):
    """Fit a standardizer on `train_data`, train an ensemble, and return
    (ready-to-predict models, scaler). Convenience wrapper for the CV scripts."""
    if model_kwargs is None:
        model_kwargs = default_model_kwargs()
    scaler = GraphStandardizer.fit(train_data)
    train_std = scaler.transform_dataset(train_data)
    val_std = scaler.transform_dataset(val_data)
    states, _ = train_ensemble(
        train_std, val_std, device, size=size, model_kwargs=model_kwargs, **train_kwargs
    )
    models = []
    for state in states:
        m = GNN(**model_kwargs)
        m.load_state_dict(state)
        m.to(device)
        m.eval()
        models.append(m)
    return models, scaler


def save_ensemble(path, states, scaler, model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = default_model_kwargs()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_states": states,
            "model_kwargs": model_kwargs,
            "scaler": scaler.state(),
        },
        path,
    )


def load_ensemble(path, device):
    """Return (models, scaler) ready for inference."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model_kwargs = ckpt["model_kwargs"]
    models = []
    for state in ckpt["model_states"]:
        m = GNN(**model_kwargs)
        m.load_state_dict(state)
        m.to(device)
        m.eval()
        models.append(m)
    scaler = GraphStandardizer.from_state(ckpt["scaler"])
    return models, scaler


def ensemble_predict(models, scaler, data, device):
    """Predict (mean_logS, std_logS) for a single PyG Data object.

    `data` must carry a `batch` vector. Members predict in standardized space;
    each is mapped back to logS before aggregating.
    """
    data = scaler.transform_data(data).to(device)
    preds = []
    with torch.no_grad():
        for model in models:
            standardized = model(data).item()
            preds.append(scaler.inverse_y(standardized))
    preds = torch.tensor(preds)
    return preds.mean().item(), preds.std(unbiased=False).item()


def ensemble_predict_dataset(models, scaler, dataset, device):
    """Return (means, actuals) numpy arrays of logS over `dataset`."""
    import numpy as np

    means, actuals = [], []
    for data in dataset:
        d = data.clone()
        d.batch = torch.zeros(d.x.shape[0], dtype=torch.long)
        mean, _ = ensemble_predict(models, scaler, d, device)
        means.append(mean)
        actuals.append(data.y.item())
    return np.array(means), np.array(actuals)
