import pandas as pd
import pytest

from src.config import SMILES_COLUMN, TARGET_COLUMN
from src import data as data_module


def test_load_raw_dataframe_uses_cache_without_network(tmp_path):
    cache = tmp_path / "cache.csv"
    df = pd.DataFrame({SMILES_COLUMN: ["CCO"], TARGET_COLUMN: [-0.77]})
    df.to_csv(cache, index=False)

    # A bogus URL would fail if the cache were ignored.
    out = data_module.load_raw_dataframe(url="http://invalid.invalid", cache_path=cache)
    assert list(out[SMILES_COLUMN]) == ["CCO"]


def test_load_raw_dataframe_writes_cache(tmp_path, monkeypatch):
    cache = tmp_path / "nested" / "cache.csv"
    fake = pd.DataFrame({SMILES_COLUMN: ["CCO"], TARGET_COLUMN: [-0.77]})
    monkeypatch.setattr(data_module.pd, "read_csv", lambda *a, **k: fake)

    out = data_module.load_raw_dataframe(url="http://example.com", cache_path=cache)
    assert cache.exists()
    assert list(out[SMILES_COLUMN]) == ["CCO"]


def test_load_dataset_keeps_data_and_smiles_in_lockstep(tmp_path):
    cache = tmp_path / "cache.csv"
    pd.DataFrame(
        {
            SMILES_COLUMN: ["CCO", "not_a_smiles", "c1ccccc1"],
            TARGET_COLUMN: [-0.77, 0.0, -2.1],
        }
    ).to_csv(cache, index=False)

    data_list, smiles_list = data_module.load_dataset(
        url="http://invalid.invalid", cache_path=cache
    )

    # invalid SMILES dropped; valid ones kept in order, lockstep with targets
    assert smiles_list == ["CCO", "c1ccccc1"]
    assert len(data_list) == 2
    assert data_list[0].y.item() == pytest.approx(-0.77)
    assert data_list[1].y.item() == pytest.approx(-2.1)
