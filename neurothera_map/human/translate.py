from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

from ..core.types import ActivityMap


def translate_to_human(
    mouse_prediction: ActivityMap,
    *,
    human_space: str = "mni152",
    name: Optional[str] = None,
    region_id_map: Optional[Mapping[str, str]] = None,
) -> ActivityMap:
    """Pragmatic MVP translation hook.

    The comprehensive plan explicitly avoids assuming a perfect region mapping.
    For the MVP, this function provides a *workflow placeholder* that preserves
    the numeric signature and records provenance, while allowing downstream
    human-space validation adapters to run.

    In real use, you would replace/extend this with:
    - ortholog-aware receptor-system translation
    - coarse network motif alignment
    - parcellation mapping

    Args:
        mouse_prediction: ActivityMap (typically mouse/CCF-indexed).
        human_space: Output space label (default: mni152).
        name: Optional output name.

    Returns:
        ActivityMap with optionally remapped region_ids and updated `space`.
    """

    if region_id_map is None:
        region_ids = np.asarray(mouse_prediction.region_ids, dtype=str)
    else:
        region_ids = np.asarray(
            [region_id_map.get(str(r), str(r)) for r in mouse_prediction.region_ids],
            dtype=str,
        )

    return ActivityMap(
        region_ids=region_ids,
        values=np.asarray(mouse_prediction.values, dtype=float),
        space=human_space,
        name=name or f"human({mouse_prediction.name})",
        provenance={
            "method": "placeholder_identity_translation",
            "input_space": mouse_prediction.space,
            "note": "Optional region_id_map applied when provided; otherwise identity.",
            "region_id_map_provided": region_id_map is not None,
        },
    )
