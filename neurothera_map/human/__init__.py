"""Human translation subpackage.

Provides loaders for human brain receptor maps and related functionality.
"""

from .receptors import PETReceptorTableSpec, load_human_pet_receptor_maps

__all__ = [
    "load_human_pet_receptor_maps",
    "PETReceptorTableSpec",
]
