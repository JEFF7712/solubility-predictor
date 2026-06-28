"""Dataset loading with a local cache.

`load_dataset` is the single entry point used by train/evaluate/tune. It caches
the raw CSV under `data/` so scripts do not re-download it on every run (the
tuner previously re-downloaded and re-featurized inside every Optuna trial).
"""

from pathlib import Path

import pandas as pd

from .config import (
    DATA_CACHE_PATH,
    DATA_URL,
    SMILES_COLUMN,
    TARGET_COLUMN,
)
from .utils import smile_to_data


def load_raw_dataframe(
    url: str = DATA_URL, cache_path: Path = DATA_CACHE_PATH
) -> pd.DataFrame:
    """Return the raw Delaney dataframe, downloading and caching it once."""
    if cache_path.exists():
        return pd.read_csv(cache_path)
    df = pd.read_csv(url)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def load_dataset(url: str = DATA_URL, cache_path: Path = DATA_CACHE_PATH):
    """Load and featurize the dataset.

    Returns (data_list, smiles_list) where data_list holds PyG `Data` objects
    with targets and smiles_list holds the corresponding SMILES strings (kept in
    lockstep so scaffold splitting can reference them).
    """
    df = load_raw_dataframe(url, cache_path)
    data_list = []
    smiles_list = []
    for _, row in df.iterrows():
        data = smile_to_data(row[SMILES_COLUMN], row[TARGET_COLUMN])
        if data is not None:
            data_list.append(data)
            smiles_list.append(row[SMILES_COLUMN])
    return data_list, smiles_list
