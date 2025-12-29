import numpy as np
import pytest

import neurothera_map


def test_public_api_imports_and_symbols_present():
    # Simple guard: importing the top-level package should be enough
    # to access the intended MVP entry points.
    for symbol in [
        "RegionMap",
        "ReceptorMap",
        "ActivityMap",
        "ConnectivityGraph",
        "DrugProfile",
        "build_drug_profile",
        "translate_to_human",
        "validate_against_pet_and_fmri",
    ]:
        assert hasattr(neurothera_map, symbol)


def test_public_api_workflow_smoke_roundtrip():
    # Build a drug profile via the public API.
    drug = neurothera_map.build_drug_profile("caffeine", mode="seed")
    assert drug.name == "caffeine"
    assert len(drug.interactions) > 0

    # Create a minimal ActivityMap and translate it.
    mouse_pred = neurothera_map.ActivityMap(
        region_ids=np.asarray(["VISp", "CA1"], dtype=str),
        values=np.asarray([1.0, -0.5], dtype=float),
        space="allen_ccf",
        name="mouse_pred",
    )
    human_pred = neurothera_map.translate_to_human(mouse_pred, human_space="mni152")
    assert human_pred.space == "mni152"
    assert human_pred.region_ids.tolist() == mouse_pred.region_ids.tolist()

    # Construct a ReceptorMap aligned to the prediction regions and validate.
    rm = neurothera_map.RegionMap(
        region_ids=np.asarray(["VISp", "CA1"], dtype=str),
        values=np.asarray([0.2, 0.1], dtype=float),
        space="mni152",
        name="ADORA1",
    )
    pet = neurothera_map.ReceptorMap(receptors={"ADORA1": rm}, space="mni152")

    report = neurothera_map.validate_against_pet_and_fmri(human_pred, pet_receptors=pet, top_n=5)
    assert report.human_prediction.name == human_pred.name
    assert "pet_receptor_correlations" in report.ranked_items

    ranked = report.ranked_items["pet_receptor_correlations"]
    assert isinstance(ranked, list)
    assert len(ranked) in (0, 1)

    if ranked:
        name, corr = ranked[0]
        assert name == "ADORA1"
        assert -1.0 <= corr <= 1.0
        assert corr == pytest.approx(float(corr))
