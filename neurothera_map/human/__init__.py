"""Human translation subpackage.

Phase 2 MVP includes:
- PET receptor map loading (offline fixtures; optional hansen_receptors)
- transcriptomics loading (offline fixtures; optional abagen)
- ActivityMap ingestion from parcellated tables
- pragmatic translation/validation hooks
"""

from .activity import activity_map_from_parcellated_table, activity_map_from_nifti
from .receptors import PETReceptorTableSpec, load_human_pet_receptor_maps
from .transcriptomics import (
    TranscriptomicTableSpec,
    load_transcriptomic_map_from_csv,
    load_transcriptomic_map_with_abagen,
)
from .translate import translate_to_human
from .validate import validate_against_pet_and_fmri

__all__ = [
    "load_human_pet_receptor_maps",
    "PETReceptorTableSpec",
    "load_transcriptomic_map_from_csv",
    "load_transcriptomic_map_with_abagen",
    "TranscriptomicTableSpec",
    "activity_map_from_parcellated_table",
    "activity_map_from_nifti",
    "translate_to_human",
    "validate_against_pet_and_fmri",
]
