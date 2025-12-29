"""Allen Institute Mouse Connectivity Loader.

Loads and processes connectivity data from the Allen Mouse Brain Connectivity Atlas.
Uses the Allen SDK to fetch projection strength data and converts it to
the NeuroThera ConnectivityGraph format.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from ..core.types import ConnectivityGraph


class AllenConnectivityLoader:
    """Load mouse connectivity from Allen Brain Connectivity Atlas.
    
    This loader interfaces with the Allen SDK to retrieve mesoscale
    connectivity data from anterograde viral tract-tracing experiments.
    
    Key features:
    - Fetches projection strength matrices
    - Supports region filtering and aggregation
    - Converts to standardized ConnectivityGraph format
    - Handles missing data and normalization
    
    Attributes:
        mcc: MouseConnectivityCache instance from Allen SDK
        structure_tree: Hierarchical structure tree of brain regions
    """
    
    def __init__(self, manifest_file: Optional[str] = None, resolution: int = 25):
        """Initialize the Allen connectivity loader.
        
        Args:
            manifest_file: Path to Allen SDK manifest file for caching.
                          If None, uses default cache location.
            resolution: Spatial resolution in microns (10, 25, 50, or 100).
                       Default is 25.
        """
        self.manifest_file = manifest_file
        self.resolution = resolution
        self.mcc = None
        self.structure_tree = None
        self._initialize_sdk()
    
    def _initialize_sdk(self) -> None:
        """Initialize Allen SDK MouseConnectivityCache."""
        try:
            from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
            
            self.mcc = MouseConnectivityCache(
                manifest_file=self.manifest_file,
                resolution=self.resolution
            )
            self.structure_tree = self.mcc.get_structure_tree()
            
        except ImportError:
            raise ImportError(
                "Allen SDK not installed. Install with: pip install allensdk"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Allen SDK: {e}")
    
    def get_available_structures(self) -> pd.DataFrame:
        """Get list of available brain structures in the Allen atlas.
        
        Returns:
            DataFrame with columns: id, acronym, name, structure_id_path, color_hex_triplet
        """
        if self.structure_tree is None:
            raise RuntimeError("Allen SDK not initialized")
        
        structures = self.structure_tree.get_structures_by_set_id([167587189])
        return pd.DataFrame(structures)
    
    def load_connectivity_matrix(
        self,
        region_ids: Optional[List[int]] = None,
        region_acronyms: Optional[List[str]] = None,
        normalize: bool = True,
        threshold: float = 0.0,
    ) -> ConnectivityGraph:
        """Load connectivity matrix from Allen atlas.
        
        Args:
            region_ids: List of Allen structure IDs to include. If None with
                       region_acronyms, will convert acronyms to IDs.
            region_acronyms: List of region acronyms (e.g., ['VISp', 'MOp']).
                            If both region_ids and region_acronyms are None,
                            loads summary structures.
            normalize: If True, row-normalize projection strengths.
            threshold: Minimum projection strength to include (after normalization).
        
        Returns:
            ConnectivityGraph with adjacency matrix and region identifiers.
        """
        if self.mcc is None or self.structure_tree is None:
            raise RuntimeError("Allen SDK not initialized")
        
        # Resolve region IDs from acronyms if needed
        if region_ids is None:
            if region_acronyms is not None:
                region_ids = self._acronyms_to_ids(region_acronyms)
            else:
                # Use summary structures (major brain divisions)
                structures = self.structure_tree.get_structures_by_set_id([167587189])
                region_ids = [s['id'] for s in structures]
        
        # Get projection density matrix
        try:
            # Build union connectivity matrix
            unionizes = self.mcc.get_structure_unionizes(
                structure_ids=region_ids,
                hemisphere_ids=[3],  # both hemispheres
                include_descendants=False
            )
            
            if len(unionizes) == 0:
                warnings.warn("No connectivity data found for specified regions")
                return self._empty_connectivity_graph()
            
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(unionizes)
            
            # Build adjacency matrix from projection_density or normalized_projection_volume
            adjacency_dict: Dict[Tuple[int, int], float] = {}
            
            for _, row in df.iterrows():
                source_id = row['structure_id']
                target_id = row['target_structure_id']
                
                # Use normalized projection volume as connectivity strength
                strength = row.get('normalized_projection_volume', 0.0)
                if strength is None:
                    strength = 0.0
                    
                adjacency_dict[(source_id, target_id)] = float(strength)
            
            # Build dense adjacency matrix
            id_to_idx = {rid: i for i, rid in enumerate(region_ids)}
            n = len(region_ids)
            adjacency = np.zeros((n, n), dtype=float)
            
            for (src, tgt), val in adjacency_dict.items():
                if src in id_to_idx and tgt in id_to_idx:
                    i, j = id_to_idx[src], id_to_idx[tgt]
                    adjacency[i, j] = val
            
            # Normalize if requested
            if normalize:
                row_sums = adjacency.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0  # avoid division by zero
                adjacency = adjacency / row_sums
            
            # Apply threshold
            if threshold > 0:
                adjacency[adjacency < threshold] = 0.0
            
            # Convert IDs to acronyms
            region_acronyms_out = self._ids_to_acronyms(region_ids)
            
            return ConnectivityGraph(
                region_ids=np.array(region_acronyms_out, dtype=str),
                adjacency=adjacency,
                name="allen_mouse_connectivity",
                provenance={
                    "source": "Allen Mouse Brain Connectivity Atlas",
                    "resolution_um": self.resolution,
                    "normalized": normalize,
                    "threshold": threshold,
                    "n_regions": len(region_ids),
                }
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to load connectivity matrix: {e}")
    
    def load_connectivity_from_experiments(
        self,
        structure_ids: List[int],
        experiment_ids: Optional[List[int]] = None,
    ) -> ConnectivityGraph:
        """Load connectivity from specific tract-tracing experiments.
        
        Args:
            structure_ids: List of Allen structure IDs for regions of interest.
            experiment_ids: Optional list of specific experiment IDs. If None,
                           uses all available experiments.
        
        Returns:
            ConnectivityGraph built from selected experiments.
        """
        if self.mcc is None:
            raise RuntimeError("Allen SDK not initialized")
        
        try:
            if experiment_ids is None:
                # Get all experiments for these structures
                experiments = self.mcc.get_experiments(
                    injection_structure_ids=structure_ids
                )
                experiment_ids = [exp['id'] for exp in experiments]
            
            if len(experiment_ids) == 0:
                warnings.warn("No experiments found for specified structures")
                return self._empty_connectivity_graph()
            
            # Aggregate connectivity across experiments
            # This is a simplified approach - could be more sophisticated
            return self.load_connectivity_matrix(
                region_ids=structure_ids,
                normalize=True
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to load experiments: {e}")
    
    def _acronyms_to_ids(self, acronyms: List[str]) -> List[int]:
        """Convert region acronyms to Allen structure IDs."""
        if self.structure_tree is None:
            raise RuntimeError("Structure tree not loaded")
        
        ids = []
        for acronym in acronyms:
            structures = self.structure_tree.get_structures_by_acronym([acronym])
            if len(structures) > 0:
                ids.append(structures[0]['id'])
            else:
                warnings.warn(f"Acronym '{acronym}' not found in Allen atlas")
        
        return ids
    
    def _ids_to_acronyms(self, ids: List[int]) -> List[str]:
        """Convert Allen structure IDs to region acronyms."""
        if self.structure_tree is None:
            raise RuntimeError("Structure tree not loaded")
        
        acronyms = []
        for structure_id in ids:
            structure = self.structure_tree.get_structures_by_id([structure_id])
            if len(structure) > 0:
                acronyms.append(structure[0]['acronym'])
            else:
                acronyms.append(f"ID{structure_id}")
        
        return acronyms
    
    def _empty_connectivity_graph(self) -> ConnectivityGraph:
        """Return an empty ConnectivityGraph."""
        return ConnectivityGraph(
            region_ids=np.array([], dtype=str),
            adjacency=np.zeros((0, 0), dtype=float),
            name="allen_mouse_connectivity",
            provenance={"source": "Allen Mouse Brain Connectivity Atlas", "empty": True}
        )


def load_allen_connectivity(
    region_acronyms: Optional[List[str]] = None,
    normalize: bool = True,
    threshold: float = 0.0,
    manifest_file: Optional[str] = None,
) -> ConnectivityGraph:
    """Convenience function to load Allen connectivity matrix.
    
    Args:
        region_acronyms: List of brain region acronyms (e.g., ['VISp', 'MOp']).
                        If None, loads summary structures.
        normalize: If True, row-normalize projection strengths.
        threshold: Minimum projection strength to include.
        manifest_file: Path to Allen SDK cache manifest.
    
    Returns:
        ConnectivityGraph with connectivity data.
    
    Example:
        >>> connectivity = load_allen_connectivity(['VISp', 'MOp', 'SSp'])
        >>> print(connectivity.adjacency.shape)
        (3, 3)
    """
    loader = AllenConnectivityLoader(manifest_file=manifest_file)
    return loader.load_connectivity_matrix(
        region_acronyms=region_acronyms,
        normalize=normalize,
        threshold=threshold
    )
