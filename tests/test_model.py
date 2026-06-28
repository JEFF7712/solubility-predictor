import pytest
import torch

from src.config import HIDDEN_DIM
from src.model import GNN
from src.utils import smile_to_data


def test_forward_returns_single_scalar_per_graph():
    model = GNN(hidden_dim=HIDDEN_DIM)
    model.eval()
    data = smile_to_data("CCO")
    data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
    with torch.no_grad():
        out = model(data)
    assert out.shape == (1, 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_dim": 64, "num_layers": 2, "dropout": 0.0},
        {"hidden_dim": 128, "num_layers": 4, "dropout": 0.2},
        {"hidden_dim": 96, "num_layers": 5, "dropout": 0.1},
    ],
)
def test_forward_with_varying_architecture(kwargs):
    model = GNN(**kwargs)
    model.eval()
    data = smile_to_data("CC(=O)Oc1ccccc1C(=O)O")
    data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
    with torch.no_grad():
        out = model(data)
    assert out.shape == (1, 1)


def test_bondless_molecule_forwards():
    model = GNN(hidden_dim=64, num_layers=3)
    model.eval()
    data = smile_to_data("C")  # no bonds
    data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
    with torch.no_grad():
        out = model(data)
    assert out.shape == (1, 1)


def test_batched_forward_returns_one_prediction_per_molecule():
    from torch_geometric.loader import DataLoader

    model = GNN(hidden_dim=HIDDEN_DIM)
    model.eval()
    mols = [smile_to_data(s) for s in ["CCO", "c1ccccc1", "CC(=O)O"]]
    loader = DataLoader(mols, batch_size=3)
    batch = next(iter(loader))
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (3, 1)
