import pytest
import torch

from src.config import CONTINUOUS_FEATURE_INDICES, NUM_NODE_FEATURES
from src.normalize import GraphStandardizer
from src.utils import smile_to_data


@pytest.fixture
def dataset():
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCCCC"]
    ys = [-0.77, -2.1, -0.17, -0.5, -3.0]
    return [smile_to_data(s, y) for s, y in zip(smiles, ys)]


def test_categorical_columns_pass_through(dataset):
    scaler = GraphStandardizer.fit(dataset)
    for i in range(NUM_NODE_FEATURES):
        if i in CONTINUOUS_FEATURE_INDICES:
            continue
        assert scaler.feature_mean[i].item() == 0.0
        assert scaler.feature_std[i].item() == 1.0


def test_continuous_columns_are_standardized(dataset):
    scaler = GraphStandardizer.fit(dataset)
    transformed = scaler.transform_dataset(dataset)
    x = torch.cat([d.x for d in transformed], dim=0)
    for i in CONTINUOUS_FEATURE_INDICES:
        col = x[:, i]
        # Standardized columns are zero-mean. Variance is only driven to 1 where
        # the raw column actually varied (constant columns stay constant via the
        # std clamp); assert that only when the fitted std is non-trivial.
        assert col.mean().abs().item() < 1e-4
        if scaler.feature_std[i].item() > 1e-3:
            # Scaler divides by the unbiased (n-1) std, so check with the same.
            assert abs(col.std().item() - 1.0) < 1e-3


def test_target_standardize_inverse_round_trip(dataset):
    scaler = GraphStandardizer.fit(dataset)
    y = torch.tensor([-1.5, 0.0, -4.2])
    assert torch.allclose(scaler.inverse_y(scaler.standardize_y(y)), y, atol=1e-5)


def test_transform_data_standardizes_target(dataset):
    scaler = GraphStandardizer.fit(dataset)
    out = scaler.transform_data(dataset[0])
    expected = (dataset[0].y - scaler.y_mean) / scaler.y_std
    assert torch.allclose(out.y, expected, atol=1e-5)


def test_state_round_trip(dataset):
    scaler = GraphStandardizer.fit(dataset)
    restored = GraphStandardizer.from_state(scaler.state())
    assert restored.y_mean == scaler.y_mean
    assert restored.y_std == scaler.y_std
    assert torch.allclose(restored.feature_mean, scaler.feature_mean)
    assert torch.allclose(restored.feature_std, scaler.feature_std)
