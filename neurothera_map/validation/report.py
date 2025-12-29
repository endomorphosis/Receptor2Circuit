from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..core.types import ActivityMap, DrugProfile


@dataclass(frozen=True)
class ValidationReport:
    """Minimal, serializable report for Phase 4 mutual validation."""

    drug: Optional[DrugProfile] = None
    mouse_prediction: Optional[ActivityMap] = None
    human_prediction: Optional[ActivityMap] = None

    # Simple numeric components (losses or scores)
    metrics: Dict[str, float] = field(default_factory=dict)

    # Optional ranked lists for interpretability
    ranked_items: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)

    provenance: Dict[str, Any] = field(default_factory=dict)
