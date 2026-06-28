"""Dataset loading with a local cache and element filtering.

`load_dataset` is the single entry point used by train/evaluate/tune. It caches
the raw file under `data/` so scripts do not re-download it on every run, and by
default keeps only molecules built from the supported elements (the model's
fixed vocabulary), dropping the rest as out of domain.
"""

import pandas as pd

from .config import DEFAULT_DATASET, DatasetSpec
from .postprocess import applicability_report
from .utils import smile_to_data


def load_raw_dataframe(spec: DatasetSpec = DEFAULT_DATASET) -> pd.DataFrame:
    """Return the raw dataframe for `spec`, downloading and caching it once."""
    if spec.cache_path.exists():
        return pd.read_csv(spec.cache_path, sep=spec.sep)
    df = pd.read_csv(spec.url, sep=spec.sep)
    spec.cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(spec.cache_path, sep=spec.sep, index=False)
    return df


def load_dataset(spec: DatasetSpec = DEFAULT_DATASET, filter_elements: bool = True):
    """Load and featurize a dataset.

    Returns (data_list, smiles_list) in lockstep. When `filter_elements` is True,
    molecules containing elements outside the supported set are dropped (the
    model cannot represent them), as are unparseable SMILES.
    """
    df = load_raw_dataframe(spec)
    data_list = []
    smiles_list = []
    for _, row in df.iterrows():
        smiles = row[spec.smiles_column]
        if filter_elements and not applicability_report(smiles)["in_domain"]:
            continue
        data = smile_to_data(smiles, row[spec.target_column])
        if data is not None:
            data_list.append(data)
            smiles_list.append(smiles)
    return data_list, smiles_list
