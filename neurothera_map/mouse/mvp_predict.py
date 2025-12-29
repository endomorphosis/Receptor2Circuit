from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.types import ActivityMap, ConnectivityGraph, DrugProfile, ReceptorMap
from .predict import diffuse_activity


def predict_mouse_effects(
    drug: DrugProfile,
    receptor_map: ReceptorMap,
    graph: Optional[ConnectivityGraph] = None,
    *,
    space: str = "allen_ccf",
    alpha: float = 0.85,
    steps: int = 25,
    fill_value: float = 0.0,
    name: Optional[str] = None,
) -> ActivityMap:
    """Minimal end-to-end mouse prediction: DrugProfile → region prior → propagation.

    This is intentionally simple and deterministic so the MVP can run offline.

    Direct prior construction:
    - For each drug interaction whose target exists in `receptor_map`:
      weight = evidence * (1 / affinity_nM) if affinity is present else evidence
      sign is inferred from action (agonist=+1, antagonist/inhibitor=-1, else +1)
    - region_prior = Σ_targets sign * weight * expression(target, region)

    Propagation:
    - If `graph` is provided, diffuses the prior over the connectivity graph using
      `diffuse_activity`.

    Args:
        drug: DrugProfile.
        receptor_map: ReceptorMap keyed by target gene symbols.
        graph: Optional ConnectivityGraph.
        space: Space label for output ActivityMap.
        alpha: Diffusion strength.
        steps: Diffusion iterations.
        fill_value: Fill for regions missing from the direct prior when aligning to `graph`.
        name: Optional output name.

    Returns:
        ActivityMap (aligned to `graph.region_ids` if graph is provided).
    """

    if graph is None:
        # Use the first available receptor's region set.
        names = receptor_map.receptor_names()
        if len(names) == 0:
            return ActivityMap(region_ids=np.array([], dtype=str), values=np.array([], dtype=float), space=space, name=name or f"pred({drug.name})")
        base = receptor_map.get(names[0])
        assert base is not None
        region_ids = base.region_ids
    else:
        region_ids = graph.region_ids

    prior = np.zeros(len(region_ids), dtype=float)

    for inter in drug.interactions:
        target = str(inter.target)
        rm = receptor_map.get(target)
        if rm is None:
            continue

        action = str(inter.action or "").lower()
        if "antagon" in action or "inhib" in action:
            sign = -1.0
        elif "agon" in action:
            sign = 1.0
        else:
            sign = 1.0

        evidence = float(inter.evidence) if inter.evidence is not None else 0.0
        if inter.affinity_nM is not None and inter.affinity_nM > 0:
            weight = evidence * (1.0 / float(inter.affinity_nM))
        else:
            weight = evidence

        aligned = rm.reindex(region_ids.tolist(), fill_value=np.nan)
        expr = aligned.values
        expr = np.where(np.isnan(expr), 0.0, expr)

        prior = prior + sign * weight * expr

    direct = ActivityMap(
        region_ids=np.asarray(region_ids, dtype=str),
        values=prior,
        space=space,
        name=name or f"direct({drug.name})",
        provenance={
            "builder": "neurothera_map.mouse.mvp_predict.predict_mouse_effects",
            "drug": drug.name,
            "alpha": alpha,
            "steps": steps,
            "used_graph": graph is not None,
        },
    )

    if graph is None:
        return direct

    return diffuse_activity(
        direct,
        graph,
        alpha=alpha,
        steps=steps,
        fill_value=fill_value,
        name=name or f"pred({drug.name})",
    )
