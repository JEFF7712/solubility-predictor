"""Interpretation helpers applied to a raw model prediction.

Two concerns the raw GNN output does not capture on its own:

1. Applicability domain: the model was trained only on molecules built from
   ALLOWED_ATOMS. Anything outside that is extrapolation, and the featurizer
   silently encodes unknown atoms as an all-zero element one-hot, so the model
   returns a confident-looking number with no signal that it is out of domain.
   `applicability_report` surfaces that instead.

2. A single logS number is hard to read. `classify_solubility` maps it to a
   qualitative class.
"""

from rdkit import Chem

from .config import ALLOWED_ATOMS

# Qualitative solubility classes by logS (log10 mol/L). Boundaries follow the
# common pharma descriptive scale (very/freely soluble down to practically
# insoluble), collapsed to four readable bands.
SOLUBILITY_CLASSES = [
    (0.0, "Highly soluble"),
    (-2.0, "Soluble"),
    (-4.0, "Slightly soluble"),
    (float("-inf"), "Poorly soluble"),
]


def classify_solubility(log_s: float) -> str:
    """Map a logS value to a qualitative solubility class."""
    for threshold, label in SOLUBILITY_CLASSES:
        if log_s >= threshold:
            return label
    return SOLUBILITY_CLASSES[-1][1]


def applicability_report(smiles: str) -> dict:
    """Assess whether a molecule falls inside the model's training domain.

    Returns {"in_domain": bool, "warnings": list[str], "unsupported_elements":
    list[str]}.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return {
            "in_domain": False,
            "warnings": ["Could not parse SMILES into a molecule."],
            "unsupported_elements": [],
        }

    unsupported = sorted(
        {a.GetSymbol() for a in mol.GetAtoms() if a.GetSymbol() not in ALLOWED_ATOMS}
    )
    warnings = []
    if unsupported:
        warnings.append(
            f"Contains element(s) the model was not trained on: "
            f"{', '.join(unsupported)}. Supported: {', '.join(ALLOWED_ATOMS)}. "
            f"Treat this prediction as unreliable extrapolation."
        )

    return {
        "in_domain": len(warnings) == 0,
        "warnings": warnings,
        "unsupported_elements": unsupported,
    }
