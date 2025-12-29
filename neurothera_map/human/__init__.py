"""Human translation subpackage.

Phase 2.3 MVP: ActivityMap ingestion from parcellated tables.
"""

from .activity import activity_map_from_parcellated_table

__all__ = ["activity_map_from_parcellated_table"]
