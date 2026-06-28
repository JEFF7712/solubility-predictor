import copy

import torch
from torch_geometric.loader import DataLoader

from .config import (
    BATCH_SIZE,
    EPOCHS,
    HIDDEN_DIM,
    LEARNING_RATE,
    MODEL_PATH,
    SEED,
    TRAIN_FRAC,
    VAL_FRAC,
    WEIGHT_DECAY,
    get_device,
    set_seed,
)
from .data import load_dataset
from .model import GNN

# Early stopping: stop if val loss has not improved for this many epochs.
EARLY_STOPPING_PATIENCE = 20

set_seed(SEED)


def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = loss_fn(out.view(-1), batch.y)
            total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def main():
    data_list, _ = load_dataset()

    # Train/val/test split
    num_samples = len(data_list)
    train_size = int(TRAIN_FRAC * num_samples)
    val_size = int(VAL_FRAC * num_samples)
    test_size = num_samples - train_size - val_size
    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set, test_set = torch.utils.data.random_split(
        data_list, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # Training setup
    device = get_device()
    model = GNN(hidden_dim=HIDDEN_DIM).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )
    loss_fn = torch.nn.MSELoss()

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out.view(-1), batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, device, loss_fn)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1} | Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f} | LR: {current_lr:.6f}"
            )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping at epoch {epoch + 1} (no improvement in "
                f"{EARLY_STOPPING_PATIENCE} epochs)"
            )
            break

    # Restore and persist the best model, not the last-epoch one.
    model.load_state_dict(best_state)
    final_val = evaluate(model, val_loader, device, loss_fn)
    final_test = evaluate(model, test_loader, device, loss_fn)
    print(
        f"Best epoch: {best_epoch} | Validation Loss: {final_val:.4f} | "
        f"Test Loss: {final_test:.4f}"
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODEL_PATH)
    print(f"Saved best model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
