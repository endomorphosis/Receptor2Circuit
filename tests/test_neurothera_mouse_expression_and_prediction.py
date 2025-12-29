import numpy as np

from neurothera_map.core.types import ActivityMap, ConnectivityGraph
from neurothera_map.drug.profile import build_drug_profile
from neurothera_map.mouse.enrichment import rank_receptors_by_activity
from neurothera_map.mouse.expression import load_receptor_map_from_csv
from neurothera_map.mouse.mvp_predict import predict_mouse_effects


def test_load_receptor_map_from_csv_long_format():
    rm = load_receptor_map_from_csv(
        "datasets/mouse_receptor_expression_fixture.csv",
        receptors=["ADORA1", "ADORA2A"],
    )

    assert set(rm.receptor_names()) == {"ADORA1", "ADORA2A"}
    a1 = rm.get("ADORA1")
    assert a1 is not None
    assert a1.region_ids.tolist() == ["CA1", "MOp", "VISp"]


def test_rank_receptors_by_activity_scores_expected_order():
    receptor_map = load_receptor_map_from_csv("datasets/mouse_receptor_expression_fixture.csv")
    activity = ActivityMap(
        region_ids=np.array(["VISp", "CA1", "MOp"], dtype=str),
        values=np.array([1.0, 0.5, 0.0], dtype=float),
        space="allen_ccf",
        name="test_activity",
    )

    ranked = rank_receptors_by_activity(activity, receptor_map)
    # ADORA1 should score highest because it's highest in CA1 + some in VISp.
    assert ranked[0][0] == "ADORA1"


def test_predict_mouse_effects_returns_activity_map_aligned_to_graph():
    receptor_map = load_receptor_map_from_csv("datasets/mouse_receptor_expression_fixture.csv")
    drug = build_drug_profile("caffeine")  # targets ADORA1/ADORA2A in the MVP seed DB

    graph = ConnectivityGraph(
        region_ids=np.array(["VISp", "CA1", "MOp"], dtype=str),
        adjacency=np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        name="toy",
    )

    pred = predict_mouse_effects(drug, receptor_map, graph, alpha=0.9, steps=5)

    assert pred.region_ids.tolist() == ["VISp", "CA1", "MOp"]
    assert pred.values.shape == (3,)
    # There should be some non-zero signal driven by receptor expression + diffusion.
    assert float(np.sum(np.abs(pred.values))) > 0.0
