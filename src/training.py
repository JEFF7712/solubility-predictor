"""Reusable single-model training loop.

Shared by the ensemble trainer and the cross-validation scripts so the training
logic (AdamW + ReduceLROnPlateau, best-val checkpointing, early stopping) lives
in one place. Datasets passed in are expected to be already standardized; loss
is therefore computed in standardized space, which is fine for model selection.
"""

import copy

import torch
from torch_geometric.loader import DataLoader

from .config import (
    BATCH_SIZE,
    EPOCHS,
    HIDDEN_DIM,
    LEARNING_RATE,
    WEIGHT_DECAY,
)
from .model import GNN

EARLY_STOPPING_PATIENCE = 20


def _eval_loss(model, loader, device, loss_fn):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            total += loss_fn(model(batch).view(-1), batch.y).item()
    return total / max(len(loader), 1)


def train_single_model(
    train_set,
    val_set,
    device,
    *,
    hidden_dim=HIDDEN_DIM,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    seed=None,
    patience=EARLY_STOPPING_PATIENCE,
):
    """Train one model, returning (best_state_dict, best_val_loss).

    Restores and returns the weights from the lowest-val-loss epoch.
    """
    if seed is not None:
        torch.manual_seed(seed)

    model = GNN(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )
    loss_fn = torch.nn.MSELoss()

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch).view(-1), batch.y)
            loss.backward()
            optimizer.step()

        val_loss = _eval_loss(model, val_loader, device, loss_fn)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_state, best_val
