import optuna
import torch
from optuna.pruners import MedianPruner
from torch_geometric.loader import DataLoader

from .config import (
    EPOCHS,
    SEED,
    TRAIN_FRAC,
    VAL_FRAC,
    get_device,
    set_seed,
)
from .data import load_dataset
from .model import GNN

N_TRIALS = 50

set_seed(SEED)
DEVICE = get_device()

# Featurize once and reuse across all trials (previously re-downloaded and
# re-featurized inside every objective call).
DATA_LIST, _ = load_dataset()


def train_and_evaluate(
    model, train_loader, val_loader, device, lr, weight_decay, epochs
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )
    loss_fn = torch.nn.MSELoss()
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out.squeeze(), batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = loss_fn(out.squeeze(), batch.y)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        scheduler.step(val_loss)
        best_val_loss = min(best_val_loss, val_loss)

    return best_val_loss


def objective(trial):
    # Suggest hyperparameters
    hidden_dim = trial.suggest_int("hidden_dim", 64, 256, step=32)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    print(f"\nTrial {trial.number + 1}/{N_TRIALS}")
    print(
        f"  hidden_dim={hidden_dim}, lr={learning_rate:.2e}, wd={weight_decay:.2e}, bs={batch_size}"
    )

    # Split (data featurized once at module load)
    num_samples = len(DATA_LIST)
    train_size = int(TRAIN_FRAC * num_samples)
    val_size = int(VAL_FRAC * num_samples)
    test_size = num_samples - train_size - val_size
    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set, _ = torch.utils.data.random_split(
        DATA_LIST, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Train
    model = GNN(hidden_dim=hidden_dim).to(DEVICE)
    best_val_loss = train_and_evaluate(
        model, train_loader, val_loader, DEVICE, learning_rate, weight_decay, EPOCHS
    )

    print(f"  Best Val Loss: {best_val_loss:.4f}")
    return best_val_loss


if __name__ == "__main__":
    print(f"Starting Hyperparameter Search with {N_TRIALS} trials")

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = MedianPruner()
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Validation Loss: {study.best_value:.4f}\n")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
