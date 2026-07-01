import torch

from src import training


def test_make_loss_fn_huber(monkeypatch):
    monkeypatch.setattr(training, "LOSS", "huber")
    loss = training.make_loss_fn()
    assert isinstance(loss, torch.nn.HuberLoss)


def test_make_loss_fn_mse(monkeypatch):
    monkeypatch.setattr(training, "LOSS", "mse")
    assert isinstance(training.make_loss_fn(), torch.nn.MSELoss)
