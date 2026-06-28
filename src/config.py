"""Central configuration: paths, dataset constants, hyperparameters.

Every script imports from here so there is a single source of truth. Previously
these constants were copy-pasted across train/evaluate/tune.
"""

import random
from pathlib import Path

import numpy as np
import torch

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
MODEL_PATH = MODELS_DIR / "gnn_solubility.pth"
DATA_CACHE_PATH = DATA_DIR / "delaney-processed.csv"

# Dataset (Delaney / ESOL)
DATA_URL = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
SMILES_COLUMN = "smiles"
TARGET_COLUMN = "measured log solubility in mols per litre"

# Featurization
ALLOWED_ATOMS = ["C", "N", "O", "F", "S", "Cl", "Br", "I"]
NUM_NODE_FEATURES = len(ALLOWED_ATOMS) + 3 + 1 + 1 + 1 + 1 + 1  # = 16
NUM_EDGE_FEATURES = 4

# Training hyperparameters (best Optuna trial)
SEED = 42
EPOCHS = 100
HIDDEN_DIM = 160
LEARNING_RATE = 0.00014067650554894532
WEIGHT_DECAY = 0.000205739489609305
BATCH_SIZE = 32

# Split fractions
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15


def set_seed(seed: int = SEED) -> None:
    """Seed all RNGs for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
