import numpy as np

from neurothera_map.core.types import ActivityMap
from neurothera_map.human.receptors import load_human_pet_receptor_maps
from neurothera_map.human.validate import validate_against_pet_and_fmri


def test_validate_against_pet_and_fmri_returns_report():
    pred = ActivityMap(
        region_ids=np.array(["ctx-lh-precentral", "ctx-lh-postcentral", "Left-Hippocampus"], dtype=str),
        values=np.array([1.0, 0.0, 0.5], dtype=float),
        space="mni152",
        name="pred",
    )

    pet = load_human_pet_receptor_maps("datasets/human_pet_receptor_fixture.csv")
    report = validate_against_pet_and_fmri(pred, pet_receptors=pet, top_n=2)

    assert "pet_receptor_correlations" in report.ranked_items
    assert len(report.ranked_items["pet_receptor_correlations"]) == 2
