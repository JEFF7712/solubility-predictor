from src.postprocess import applicability_report, classify_solubility


def test_classify_solubility_bands():
    assert classify_solubility(1.0) == "Highly soluble"
    assert classify_solubility(0.0) == "Highly soluble"
    assert classify_solubility(-1.0) == "Soluble"
    assert classify_solubility(-2.0) == "Soluble"
    assert classify_solubility(-3.0) == "Slightly soluble"
    assert classify_solubility(-4.0) == "Slightly soluble"
    assert classify_solubility(-6.0) == "Poorly soluble"


def test_in_domain_molecule():
    report = applicability_report("CC(=O)Oc1ccccc1C(=O)O")  # aspirin: C, O
    assert report["in_domain"] is True
    assert report["warnings"] == []
    assert report["unsupported_elements"] == []


def test_out_of_domain_unsupported_element():
    report = applicability_report("OP(=O)(O)O")  # phosphoric acid: P unsupported
    assert report["in_domain"] is False
    assert report["unsupported_elements"] == ["P"]
    assert len(report["warnings"]) == 1
    assert "P" in report["warnings"][0]


def test_unparseable_smiles_is_out_of_domain():
    report = applicability_report("not_a_smiles")
    assert report["in_domain"] is False
    assert report["warnings"]


def test_multiple_unsupported_elements_sorted_unique():
    report = applicability_report("[Si]([Na])O")  # Si and Na unsupported
    assert report["unsupported_elements"] == ["Na", "Si"]
