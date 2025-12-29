import numpy as np

from neurothera_map.core.types import ActivityMap, ConnectivityGraph, DrugInteraction, DrugProfile, RegionMap, ReceptorMap
from neurothera_map.validation.schema import (
    validate_activity_map,
    validate_connectivity_graph,
    validate_drug_profile,
    validate_receptor_map,
)


def test_schema_validators_accept_basic_objects():
    am = ActivityMap(region_ids=np.array(["A"], dtype=str), values=np.array([1.0], dtype=float))
    validate_activity_map(am)

    g = ConnectivityGraph(region_ids=np.array(["A"], dtype=str), adjacency=np.array([[0.0]], dtype=float))
    validate_connectivity_graph(g)

    rm = ReceptorMap(receptors={"X": RegionMap(region_ids=np.array(["A"], dtype=str), values=np.array([0.2], dtype=float))})
    validate_receptor_map(rm, require_nonempty=True)

    dp = DrugProfile(name="d", interactions=(DrugInteraction(target="X", affinity_nM=10.0, evidence=0.5),))
    validate_drug_profile(dp, require_targets=True)
