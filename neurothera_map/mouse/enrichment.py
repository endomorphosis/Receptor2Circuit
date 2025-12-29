from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..core.types import ActivityMap, ReceptorMap


def rank_receptors_by_activity(
    activity: ActivityMap,
    receptors: ReceptorMap,
    *,
    receptor_names: Optional[Sequence[str]] = None,
    use_absolute_activity: bool = True,
    normalize_activity_weights: bool = True,
    top_n: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """Score receptors by activity-weighted expression.

    Implements the Phase 1.3 primitive:
        score(receptor) = Σ_regions w(region) * expr(receptor, region)

    Where w(region) is derived from `activity.values`.

    Args:
        activity: ActivityMap defining per-region weights/signals.
        receptors: ReceptorMap containing one RegionMap per receptor.
        receptor_names: Optional subset of receptors to score.
        use_absolute_activity: If True, uses |activity| as weights.
        normalize_activity_weights: If True, rescales weights to sum to 1.
        top_n: Optional number of top receptors to return.

    Returns:
        List of (receptor, score) sorted descending by score.
    """
    names = receptor_names if receptor_names is not None else receptors.receptor_names()

    weights = activity.values.astype(float)
    if use_absolute_activity:
        weights = np.abs(weights)

    if normalize_activity_weights:
        denom = float(np.nansum(weights))
        if denom > 0:
            weights = weights / denom

    out: List[Tuple[str, float]] = []

    for receptor in names:
        rm = receptors.get(str(receptor))
        if rm is None:
            continue

        aligned = rm.reindex(activity.region_ids.tolist(), fill_value=np.nan)
        expr = aligned.values.astype(float)

        mask = ~np.isnan(expr)
        if not np.any(mask):
            score = float("nan")
        else:
            score = float(np.nansum(weights[mask] * expr[mask]))

        out.append((str(receptor), score))

    out.sort(key=lambda x: (np.isnan(x[1]), -x[1] if not np.isnan(x[1]) else 0.0))

    if top_n is not None:
        return out[: int(top_n)]
    return out
