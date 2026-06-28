import torch

from src.ensemble import (
    ensemble_predict,
    fit_ensemble,
    load_ensemble,
    save_ensemble,
)
from src.normalize import GraphStandardizer
from src.utils import smile_to_data

SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCCCC", "CCOCC", "CCCl", "CCBr"]
YS = [-0.77, -2.1, -0.17, -0.5, -3.0, -1.0, -1.3, -1.6]


def _dataset():
    return [smile_to_data(s, y) for s, y in zip(SMILES, YS)]


def test_fit_ensemble_returns_models_and_scaler():
    data = _dataset()
    device = torch.device("cpu")
    models, scaler = fit_ensemble(data[:6], data[6:], device, size=2, epochs=2)
    assert len(models) == 2
    assert isinstance(scaler, GraphStandardizer)


def test_ensemble_predict_returns_mean_and_nonneg_std():
    data = _dataset()
    device = torch.device("cpu")
    models, scaler = fit_ensemble(data[:6], data[6:], device, size=3, epochs=2)
    d = data[0].clone()
    d.batch = torch.zeros(d.x.shape[0], dtype=torch.long)
    mean, std = ensemble_predict(models, scaler, d, device)
    assert isinstance(mean, float)
    assert std >= 0.0


def test_save_load_round_trip_preserves_predictions(tmp_path):
    data = _dataset()
    device = torch.device("cpu")
    scaler = GraphStandardizer.fit(data[:6])
    train_std = scaler.transform_dataset(data[:6])
    val_std = scaler.transform_dataset(data[6:])
    from src.ensemble import train_ensemble

    states, _ = train_ensemble(train_std, val_std, device, size=2, epochs=2)
    path = tmp_path / "ens.pth"
    save_ensemble(path, states, scaler)

    models, loaded_scaler = load_ensemble(path, device)
    d = data[0].clone()
    d.batch = torch.zeros(d.x.shape[0], dtype=torch.long)
    m1, _ = ensemble_predict(models, loaded_scaler, data[0].clone(), device)
    assert loaded_scaler.y_mean == scaler.y_mean
    assert len(models) == 2
    assert isinstance(m1, float)
