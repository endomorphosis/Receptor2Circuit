from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.types import ActivityMap, ReceptorMap
from ..validation.report import ValidationReport


def validate_against_pet_and_fmri(
    human_prediction: ActivityMap,
    *,
    pet_receptors: Optional[ReceptorMap] = None,
    observed_activity: Optional[ActivityMap] = None,
    top_n: int = 10,
) -> ValidationReport:
    """Minimal, interpretable validation hook.

    Computes simple correlations:
    - between predicted human activity and each PET receptor density map (if provided)
    - between predicted and observed parcellated activity (if provided)

    Args:
        human_prediction: predicted ActivityMap in human space.
        pet_receptors: optional PET ReceptorMap.
        observed_activity: optional observed ActivityMap for empirical alignment.
        top_n: number of top correlations to return.

    Returns:
        ValidationReport
    """
    metrics: Dict[str, float] = {}
    ranked_items: Dict[str, List[Tuple[str, float]]] = {}

    if observed_activity is not None:
        obs = observed_activity.reindex(human_prediction.region_ids.tolist(), fill_value=np.nan)
        x = human_prediction.values.astype(float)
        y = obs.values.astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.any(mask):
            corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
            metrics["corr_pred_vs_observed"] = corr

    if pet_receptors is not None:
        rows: List[Tuple[str, float]] = []
        x = human_prediction.values.astype(float)
        for receptor in pet_receptors.receptor_names():
            rm = pet_receptors.get(receptor)
            if rm is None:
                continue
            aligned = rm.reindex(human_prediction.region_ids.tolist(), fill_value=np.nan)
            y = aligned.values.astype(float)
            mask = np.isfinite(x) & np.isfinite(y)
            if not np.any(mask):
                continue
            corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
            rows.append((receptor, corr))

        rows.sort(key=lambda t: -t[1])
        ranked_items["pet_receptor_correlations"] = rows[: int(top_n)]

    return ValidationReport(
        human_prediction=human_prediction,
        metrics=metrics,
        ranked_items=ranked_items,
        provenance={"validator": "neurothera_map.human.validate_against_pet_and_fmri"},
    )
