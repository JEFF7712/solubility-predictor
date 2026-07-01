"""Central configuration: paths, dataset constants, hyperparameters.

Every script imports from here so there is a single source of truth. Previously
these constants were copy-pasted across train/evaluate/tune.
"""

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
MODEL_PATH = MODELS_DIR / "gnn_solubility.pth"
ENSEMBLE_MODEL_PATH = MODELS_DIR / "gnn_ensemble.pth"


@dataclass(frozen=True)
class DatasetSpec:
    """A solubility dataset: where to fetch it and which columns to read."""

    name: str
    url: str
    cache_path: Path
    smiles_column: str
    target_column: str
    sep: str = ","


# Delaney / ESOL (~1.1k molecules) and AqSolDB (~10k curated). Both report logS
# (log10 mol/L). AqSolDB is the default; it is ~7.7x larger after filtering to
# the supported elements.
DELANEY = DatasetSpec(
    name="delaney",
    url="https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv",
    cache_path=DATA_DIR / "delaney-processed.csv",
    smiles_column="smiles",
    target_column="measured log solubility in mols per litre",
)
AQSOLDB = DatasetSpec(
    name="aqsoldb",
    url="https://dataverse.harvard.edu/api/access/datafile/3407241",
    cache_path=DATA_DIR / "aqsoldb.tsv",
    smiles_column="SMILES",
    target_column="Solubility",
    sep="\t",
)
DEFAULT_DATASET = AQSOLDB

# Featurization
ALLOWED_ATOMS = ["C", "N", "O", "F", "S", "Cl", "Br", "I"]
NUM_NODE_FEATURES = len(ALLOWED_ATOMS) + 3 + 1 + 1 + 1 + 1 + 1  # = 16
NUM_EDGE_FEATURES = 4
# Node-feature column layout (see utils.get_atom_features):
#   0-7  element one-hot     8-10 hybridization one-hot   11 aromatic (binary)
#   12 formal charge   13 num Hs   14 degree   15 mass
# Only the continuous columns are standardized; one-hot/binary pass through.
CONTINUOUS_FEATURE_INDICES = [12, 13, 14, 15]

# Ensemble
ENSEMBLE_SIZE = 5

# Training hyperparameters (best Optuna trial on AqSolDB; 50-trial GPU search)
SEED = 42
EPOCHS = 100
HIDDEN_DIM = 256
NUM_LAYERS = 3
DROPOUT = 0.024621498012050512
LEARNING_RATE = 0.0002288073628058266
WEIGHT_DECAY = 0.00021821392099950144
BATCH_SIZE = 128

# Loss: "mse" or "huber". Huber (smooth-L1) down-weights outliers, which can help
# on AqSolDB's noisier, wide-range labels. Delta is in standardized-target units.
LOSS = "huber"
HUBER_DELTA = 1.0

# Split fractions
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15


def default_model_kwargs() -> dict:
    """Architecture kwargs for GNN(...). Persisted with the model so inference
    reconstructs the exact same network."""
    return {"hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS, "dropout": DROPOUT}


def set_seed(seed: int = SEED) -> None:
    """Seed all RNGs for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
