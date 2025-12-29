import numpy as np
import pytest

from neurothera_map import (
    ActivityMap,
    ConnectivityGraph,
    build_drug_profile,
    translate_to_human,
    validate_against_pet_and_fmri,
)
from neurothera_map.human.receptors import load_human_pet_receptor_maps
from neurothera_map.mouse.expression import load_receptor_map_from_csv
from neurothera_map.mouse.mvp_predict import predict_mouse_effects
from neurothera_map.validation.schema import (
    validate_activity_map,
    validate_connectivity_graph,
    validate_drug_profile,
    validate_receptor_map,
)


def test_end_to_end_mouse_pipeline_offline():
    receptor_map = load_receptor_map_from_csv("datasets/mouse_receptor_expression_fixture.csv")
    validate_receptor_map(receptor_map, require_nonempty=True)

    # Exercise the auto-mode fallback path deterministically (no network).
    drug = build_drug_profile("caffeine", mode="auto", use_iuphar=False, use_chembl=False)
    validate_drug_profile(drug, require_targets=True)

    regions = np.array(["VISp", "CA1", "MOp"], dtype=str)
    adjacency = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    graph = ConnectivityGraph(region_ids=regions, adjacency=adjacency, name="toy_mouse")
    validate_connectivity_graph(graph)

    pred = predict_mouse_effects(drug, receptor_map, graph, alpha=0.9, steps=5)
    validate_activity_map(pred, allow_nan=False)

    assert pred.region_ids.tolist() == regions.tolist()
    assert np.any(pred.values != 0.0)
    # Caffeine is a seed antagonist at ADORA receptors; expression is positive ⇒ negative prior.
    assert float(np.min(pred.values)) < 0.0


def test_end_to_end_human_validation_pipeline_correlations():
    pet = load_human_pet_receptor_maps("datasets/human_pet_receptor_fixture.csv")
    validate_receptor_map(pet, require_nonempty=True)

    d1 = pet.get("D1")
    assert d1 is not None

    pred = ActivityMap(
        region_ids=np.asarray(d1.region_ids, dtype=str),
        values=np.asarray(d1.values, dtype=float),
        space="mni152",
        name="pred_from_D1",
    )
    observed = ActivityMap(
        region_ids=np.asarray(d1.region_ids, dtype=str),
        values=np.asarray(d1.values, dtype=float),
        space="mni152",
        name="observed",
    )

    report = validate_against_pet_and_fmri(
        pred,
        pet_receptors=pet,
        observed_activity=observed,
        top_n=3,
    )

    assert report.metrics["corr_pred_vs_observed"] == pytest.approx(1.0)

    ranked = report.ranked_items.get("pet_receptor_correlations")
    assert ranked is not None
    assert len(ranked) >= 1
    top_name, top_corr = ranked[0]
    assert top_name == "D1"
    assert top_corr == pytest.approx(1.0)


def test_end_to_end_cross_species_placeholder_translation_does_not_crash():
    receptor_map = load_receptor_map_from_csv("datasets/mouse_receptor_expression_fixture.csv")
    drug = build_drug_profile("caffeine", mode="seed")

    regions = np.array(["VISp", "CA1", "MOp"], dtype=str)
    graph = ConnectivityGraph(region_ids=regions, adjacency=np.eye(3, dtype=float), name="toy_identity")

    mouse_pred = predict_mouse_effects(drug, receptor_map, graph, alpha=0.85, steps=3)
    human_pred = translate_to_human(mouse_pred, human_space="mni152")

    pet = load_human_pet_receptor_maps("datasets/human_pet_receptor_fixture.csv")

    report = validate_against_pet_and_fmri(human_pred, pet_receptors=pet, top_n=5)

    assert report.human_prediction.space == "mni152"
    assert "pet_receptor_correlations" in report.ranked_items
    # There is no region overlap in the fixtures; this should be a clean empty result.
    assert report.ranked_items["pet_receptor_correlations"] == []


def test_end_to_end_cross_species_with_minimal_mapping_produces_signal():
    # Build a mouse prediction that will line up with a tiny human PET fixture
    # via a simple region-id mapping.
    full = load_receptor_map_from_csv("datasets/mouse_receptor_expression_fixture.csv")
    adora1 = full.get("ADORA1")
    assert adora1 is not None
    receptor_map = type(full)(receptors={"ADORA1": adora1}, space=full.space, provenance=full.provenance)

    drug = build_drug_profile("caffeine", mode="seed")
    graph = ConnectivityGraph(
        region_ids=np.asarray(adora1.region_ids, dtype=str),
        adjacency=np.eye(len(adora1.region_ids), dtype=float),
        name="identity",
    )
    mouse_pred = predict_mouse_effects(drug, receptor_map, graph, alpha=0.85, steps=3)

    # Load mapping (mouse regions → human regions) for translation.
    mapping = {}
    with open("datasets/cross_species_region_map_fixture.csv", "r", encoding="utf-8") as f:
        header = f.readline()
        assert "mouse_region" in header and "human_region" in header
        for line in f:
            line = line.strip()
            if not line:
                continue
            mouse_region, human_region = line.split(",", 1)
            mapping[mouse_region] = human_region

    human_pred = translate_to_human(mouse_pred, human_space="mni152", region_id_map=mapping)

    pet = load_human_pet_receptor_maps("datasets/human_pet_receptor_cross_species_fixture.csv")
    report = validate_against_pet_and_fmri(human_pred, pet_receptors=pet, top_n=5)

    ranked = report.ranked_items.get("pet_receptor_correlations")
    assert ranked is not None
    assert len(ranked) >= 1
    assert ranked[0][0] == "ADORA1"
    assert ranked[0][1] == pytest.approx(1.0)
