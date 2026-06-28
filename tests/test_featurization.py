import pytest

from src.config import ALLOWED_ATOMS, NUM_EDGE_FEATURES, NUM_NODE_FEATURES
from src.utils import get_atom_features, smile_to_data


def test_valid_smiles_produces_data():
    data = smile_to_data("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    assert data is not None
    # aspirin has 13 heavy atoms
    assert data.x.shape == (13, NUM_NODE_FEATURES)
    # 13 bonds, each stored in both directions
    assert data.edge_index.shape[0] == 2
    assert data.edge_attr.shape[1] == NUM_EDGE_FEATURES
    assert data.edge_index.shape[1] == data.edge_attr.shape[0]


def test_invalid_smiles_returns_none():
    assert smile_to_data("not_a_smiles") is None
    assert smile_to_data("") is None


def test_bondless_molecule_has_correctly_shaped_edges():
    # A single-atom species has no bonds; edge tensors must still be
    # [2, 0] / [0, NUM_EDGE_FEATURES] so the edge encoder accepts them.
    data = smile_to_data("C")  # methane: one heavy atom, no bonds
    assert data is not None
    assert data.edge_index.shape == (2, 0)
    assert data.edge_attr.shape == (0, NUM_EDGE_FEATURES)


def test_target_attached_when_provided():
    data = smile_to_data("CCO", y_value=-0.77)
    assert data.y.item() == pytest.approx(-0.77)
    assert "y" not in smile_to_data("CCO")


def test_node_feature_width_matches_config():
    from rdkit import Chem

    mol = Chem.MolFromSmiles("CCO")
    feats = get_atom_features(mol.GetAtomWithIdx(0))
    assert len(feats) == NUM_NODE_FEATURES


def test_allowed_atom_one_hot():
    from rdkit import Chem

    mol = Chem.MolFromSmiles("CCO")  # carbon first
    feats = get_atom_features(mol.GetAtomWithIdx(0))
    one_hot = feats[: len(ALLOWED_ATOMS)]
    assert sum(one_hot) == 1
    assert one_hot[ALLOWED_ATOMS.index("C")] == 1


def test_out_of_domain_atom_has_zero_one_hot():
    # Documents current behaviour: atoms outside ALLOWED_ATOMS (e.g. phosphorus)
    # get an all-zero element one-hot. Applicability-domain detection (phase 2)
    # will surface this instead of silently predicting.
    from rdkit import Chem

    mol = Chem.MolFromSmiles("OP(=O)(O)O")  # phosphoric acid
    p_atom = next(a for a in mol.GetAtoms() if a.GetSymbol() == "P")
    feats = get_atom_features(p_atom)
    assert sum(feats[: len(ALLOWED_ATOMS)]) == 0
