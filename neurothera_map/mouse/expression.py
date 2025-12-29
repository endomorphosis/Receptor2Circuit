from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..core.types import ReceptorMap, RegionMap


@dataclass(frozen=True)
class ExpressionTableSpec:
    region_col: str = "region"
    receptor_col: str = "receptor"
    value_col: str = "value"
    uncertainty_col: Optional[str] = None


def load_receptor_map_from_csv(
    path: str | Path,
    *,
    receptors: Optional[Sequence[str]] = None,
    spec: ExpressionTableSpec = ExpressionTableSpec(),
    space: str = "allen_ccf",
    name_prefix: str = "mouse_expression",
) -> ReceptorMap:
    """Load a mouse receptor/gene expression panel into a `ReceptorMap`.

    This is the Phase 1.1 MVP connector for "Allen Mouse Brain Atlas expression summaries"
    in an *offline* form: you provide a CSV table that already contains region-level
    expression summaries.

    Supported CSV layouts:

    1) Long format (recommended):
       - columns: region, receptor, value (and optionally uncertainty)

    2) Wide format:
       - one row per region
       - columns: region, <RECEPTOR_1>, <RECEPTOR_2>, ...

    Args:
        path: CSV file path.
        receptors: Optional subset of receptors/genes to load.
        spec: Column names for long-format CSV.
        space: Space label (default: "allen_ccf").
        name_prefix: Prefix used for `RegionMap.name`.

    Returns:
        ReceptorMap
    """
    p = Path(path)
    df = pd.read_csv(p)

    if spec.receptor_col in df.columns and spec.value_col in df.columns:
        return _load_long_format(df, receptors=receptors, spec=spec, space=space, name_prefix=name_prefix, src=p)

    if spec.region_col in df.columns:
        return _load_wide_format(df, receptors=receptors, region_col=spec.region_col, space=space, name_prefix=name_prefix, src=p)

    raise ValueError(
        "Unrecognized expression CSV format. Expected long format with columns "
        f"'{spec.region_col}', '{spec.receptor_col}', '{spec.value_col}' "
        "or wide format with a region column and receptor columns."
    )


def _load_long_format(
    df: pd.DataFrame,
    *,
    receptors: Optional[Sequence[str]],
    spec: ExpressionTableSpec,
    space: str,
    name_prefix: str,
    src: Path,
) -> ReceptorMap:
    if spec.region_col not in df.columns:
        raise ValueError(f"Missing required column '{spec.region_col}'")

    work = df.copy()
    work[spec.region_col] = work[spec.region_col].astype(str)
    work[spec.receptor_col] = work[spec.receptor_col].astype(str)

    if receptors is not None:
        keep = {str(r) for r in receptors}
        work = work[work[spec.receptor_col].isin(keep)]

    receptors_out: Dict[str, RegionMap] = {}

    for receptor, sub in work.groupby(spec.receptor_col, sort=True):
        sub2 = sub[[spec.region_col, spec.value_col] + ([spec.uncertainty_col] if spec.uncertainty_col else [])].copy()
        sub2 = sub2.dropna(subset=[spec.value_col])
        sub2 = sub2.groupby(spec.region_col, as_index=False).mean(numeric_only=True)
        sub2 = sub2.sort_values(spec.region_col)

        region_ids = sub2[spec.region_col].to_numpy(dtype=str)
        values = sub2[spec.value_col].to_numpy(dtype=float)

        uncertainty = None
        if spec.uncertainty_col and spec.uncertainty_col in sub2.columns:
            uncertainty = sub2[spec.uncertainty_col].to_numpy(dtype=float)

        receptors_out[str(receptor)] = RegionMap(
            region_ids=region_ids,
            values=values,
            uncertainty=uncertainty,
            space=space,
            name=f"{name_prefix}:{receptor}",
            provenance={
                "source": "csv",
                "path": str(src),
                "format": "long",
                "region_col": spec.region_col,
                "value_col": spec.value_col,
            },
        )

    return ReceptorMap(
        receptors=receptors_out,
        space=space,
        provenance={"source": "csv", "path": str(src)},
    )


def _load_wide_format(
    df: pd.DataFrame,
    *,
    receptors: Optional[Sequence[str]],
    region_col: str,
    space: str,
    name_prefix: str,
    src: Path,
) -> ReceptorMap:
    work = df.copy()
    work[region_col] = work[region_col].astype(str)

    receptor_cols = [c for c in work.columns if c != region_col]
    if len(receptor_cols) == 0:
        raise ValueError("Wide-format expression CSV has no receptor columns")

    if receptors is not None:
        keep = {str(r) for r in receptors}
        receptor_cols = [c for c in receptor_cols if str(c) in keep]

    work = work[[region_col] + receptor_cols]
    work = work.sort_values(region_col)
    region_ids = work[region_col].to_numpy(dtype=str)

    receptors_out: Dict[str, RegionMap] = {}
    for receptor in receptor_cols:
        values = pd.to_numeric(work[receptor], errors="coerce").to_numpy(dtype=float)
        receptors_out[str(receptor)] = RegionMap(
            region_ids=region_ids,
            values=values,
            space=space,
            name=f"{name_prefix}:{receptor}",
            provenance={"source": "csv", "path": str(src), "format": "wide", "region_col": region_col},
        )

    return ReceptorMap(receptors=receptors_out, space=space, provenance={"source": "csv", "path": str(src)})
