import pandas as pd
import pytest

from src import data as data_module
from src.config import DatasetSpec


def _spec(cache_path):
    return DatasetSpec(
        name="test",
        url="http://invalid.invalid",
        cache_path=cache_path,
        smiles_column="smiles",
        target_column="target",
    )


def test_load_raw_dataframe_uses_cache_without_network(tmp_path):
    cache = tmp_path / "cache.csv"
    pd.DataFrame({"smiles": ["CCO"], "target": [-0.77]}).to_csv(cache, index=False)

    # A bogus URL would fail if the cache were ignored.
    out = data_module.load_raw_dataframe(_spec(cache))
    assert list(out["smiles"]) == ["CCO"]


def test_load_raw_dataframe_writes_cache(tmp_path, monkeypatch):
    cache = tmp_path / "nested" / "cache.csv"
    fake = pd.DataFrame({"smiles": ["CCO"], "target": [-0.77]})
    monkeypatch.setattr(data_module.pd, "read_csv", lambda *a, **k: fake)

    out = data_module.load_raw_dataframe(_spec(cache))
    assert cache.exists()
    assert list(out["smiles"]) == ["CCO"]


def test_load_dataset_keeps_data_and_smiles_in_lockstep(tmp_path):
    cache = tmp_path / "cache.csv"
    pd.DataFrame(
        {
            "smiles": ["CCO", "not_a_smiles", "c1ccccc1"],
            "target": [-0.77, 0.0, -2.1],
        }
    ).to_csv(cache, index=False)

    data_list, smiles_list = data_module.load_dataset(_spec(cache))

    # invalid SMILES dropped; valid ones kept in order, lockstep with targets
    assert smiles_list == ["CCO", "c1ccccc1"]
    assert len(data_list) == 2
    assert data_list[0].y.item() == pytest.approx(-0.77)
    assert data_list[1].y.item() == pytest.approx(-2.1)


def test_load_dataset_filters_out_of_domain_elements(tmp_path):
    cache = tmp_path / "cache.csv"
    pd.DataFrame(
        {
            "smiles": ["CCO", "OP(=O)(O)O", "[Na+].[Cl-]"],  # P and Na unsupported
            "target": [-0.77, 0.5, 0.4],
        }
    ).to_csv(cache, index=False)

    data_list, smiles_list = data_module.load_dataset(_spec(cache))
    assert smiles_list == ["CCO"]
    assert len(data_list) == 1


def test_load_dataset_can_disable_element_filter(tmp_path):
    cache = tmp_path / "cache.csv"
    pd.DataFrame({"smiles": ["CCO", "OP(=O)(O)O"], "target": [-0.77, 0.5]}).to_csv(
        cache, index=False
    )

    data_list, smiles_list = data_module.load_dataset(
        _spec(cache), filter_elements=False
    )
    assert smiles_list == ["CCO", "OP(=O)(O)O"]
