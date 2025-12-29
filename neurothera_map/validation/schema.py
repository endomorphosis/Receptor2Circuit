from __future__ import annotations

from typing import Iterable, Mapping, Optional

import numpy as np

from ..core.types import ActivityMap, ConnectivityGraph, DrugProfile, ReceptorMap


def validate_activity_map(activity: ActivityMap, *, allow_nan: bool = True) -> None:
    if len(activity.region_ids) != len(activity.values):
        raise ValueError("ActivityMap region_ids and values must match")
    if not allow_nan and np.any(~np.isfinite(activity.values)):
        raise ValueError("ActivityMap values contain non-finite entries")


def validate_connectivity_graph(graph: ConnectivityGraph, *, allow_nan: bool = False) -> None:
    if graph.adjacency.shape[0] != graph.adjacency.shape[1]:
        raise ValueError("ConnectivityGraph adjacency must be square")
    if graph.adjacency.shape[0] != len(graph.region_ids):
        raise ValueError("ConnectivityGraph adjacency must match region_ids length")
    if not allow_nan and np.any(~np.isfinite(graph.adjacency)):
        raise ValueError("ConnectivityGraph adjacency contains non-finite entries")


def validate_receptor_map(receptors: ReceptorMap, *, require_nonempty: bool = False) -> None:
    names = receptors.receptor_names()
    if require_nonempty and len(names) == 0:
        raise ValueError("ReceptorMap is empty")

    for name in names:
        rm = receptors.get(name)
        if rm is None:
            raise ValueError(f"ReceptorMap missing receptor '{name}'")
        if len(rm.region_ids) != len(rm.values):
            raise ValueError(f"RegionMap for '{name}' has mismatched lengths")


def validate_drug_profile(drug: DrugProfile, *, require_targets: bool = False) -> None:
    if require_targets and len(drug.interactions) == 0:
        raise ValueError("DrugProfile has no interactions")
    for inter in drug.interactions:
        if not inter.target:
            raise ValueError("DrugInteraction has empty target")
        if inter.evidence < 0.0 or inter.evidence > 1.0:
            raise ValueError("DrugInteraction evidence must be in [0, 1]")
        if inter.affinity_nM is not None and inter.affinity_nM <= 0:
            raise ValueError("DrugInteraction affinity_nM must be positive")
