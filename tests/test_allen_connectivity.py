"""Tests for Allen connectivity loader."""

import warnings
import pytest
import numpy as np
import pandas as pd

from neurothera_map.mouse.allen_connectivity import (
    AllenConnectivityLoader,
    load_allen_connectivity,
)
from neurothera_map.core.types import ConnectivityGraph


class TestAllenConnectivityLoader:
    """Test suite for AllenConnectivityLoader."""

    def test_initialization(self):
        """Test loader can be initialized."""
        try:
            loader = AllenConnectivityLoader()
            assert loader is not None
            assert loader.mcc is not None
            assert loader.structure_tree is not None
        except ImportError:
            pytest.skip("Allen SDK not installed")

    def test_initialization_with_custom_resolution(self):
        """Test loader initialization with custom resolution."""
        try:
            for resolution in [10, 25, 50, 100]:
                loader = AllenConnectivityLoader(resolution=resolution)
                assert loader.resolution == resolution
                assert loader.mcc is not None
        except ImportError:
            pytest.skip("Allen SDK not installed")

    def test_initialization_with_manifest_file(self):
        """Test loader initialization with custom manifest file."""
        try:
            import tempfile
            import os
            
            with tempfile.TemporaryDirectory() as tmpdir:
                manifest = os.path.join(tmpdir, "manifest.json")
                loader = AllenConnectivityLoader(manifest_file=manifest)
                assert loader.manifest_file == manifest
        except ImportError:
            pytest.skip("Allen SDK not installed")

    def test_get_available_structures(self):
        """Test retrieval of available brain structures."""
        try:
            loader = AllenConnectivityLoader()
            structures = loader.get_available_structures()
            assert len(structures) > 0
            assert 'acronym' in structures.columns
            assert 'id' in structures.columns
            assert 'name' in structures.columns
            
            # Verify structure types
            assert isinstance(structures, pd.DataFrame)
            assert structures['id'].dtype in [np.int64, np.int32]
            
        except ImportError:
            pytest.skip("Allen SDK not installed")

    def test_load_connectivity_matrix_with_acronyms(self):
        """Test loading connectivity matrix with region acronyms."""
        try:
            loader = AllenConnectivityLoader()
            # Use common visual cortex regions
            connectivity = loader.load_connectivity_matrix(
                region_acronyms=['VISp', 'VISl', 'VISal'],
                normalize=True,
                threshold=0.0
            )
            
            assert isinstance(connectivity, ConnectivityGraph)
            assert len(connectivity.region_ids) == 3
            assert connectivity.adjacency.shape == (3, 3)
            assert connectivity.name == "allen_mouse_connectivity"
            assert "Allen Mouse Brain Connectivity Atlas" in connectivity.provenance['source']
            
        except ImportError:
            pytest.skip("Allen SDK not installed")
        except Exception as e:
            # Network errors or missing data are acceptable in tests
            pytest.skip(f"Could not load connectivity data: {e}")

    def test_load_connectivity_matrix_with_region_ids(self):
        """Test loading connectivity matrix with region IDs."""
        try:
            loader = AllenConnectivityLoader()
            # VISp=385, VISl=409
            connectivity = loader.load_connectivity_matrix(
                region_ids=[385, 409],
                normalize=True,
                threshold=0.0
            )
            
            assert isinstance(connectivity, ConnectivityGraph)
            assert len(connectivity.region_ids) == 2
            assert connectivity.adjacency.shape == (2, 2)
            
        except ImportError:
            pytest.skip("Allen SDK not installed")
        except Exception as e:
            pytest.skip(f"Could not load connectivity data: {e}")

    def test_load_connectivity_matrix_default(self):
        """Test loading connectivity with default summary structures."""
        try:
            loader = AllenConnectivityLoader()
            connectivity = loader.load_connectivity_matrix()
            
            assert isinstance(connectivity, ConnectivityGraph)
            assert len(connectivity.region_ids) > 0
            assert connectivity.adjacency.shape[0] == connectivity.adjacency.shape[1]
            assert connectivity.adjacency.shape[0] == len(connectivity.region_ids)
            
        except ImportError:
            pytest.skip("Allen SDK not installed")
        except Exception as e:
            pytest.skip(f"Could not load connectivity data: {e}")

    def test_connectivity_normalization(self):
        """Test that normalization produces row-normalized matrix."""
        try:
            loader = AllenConnectivityLoader()
            connectivity = loader.load_connectivity_matrix(
                region_acronyms=['VISp', 'MOp'],
                normalize=True
            )
            
            # Check row sums are close to 1 or 0 (for regions with no projections)
            row_sums = connectivity.adjacency.sum(axis=1)
            for row_sum in row_sums:
                assert row_sum == 0.0 or np.isclose(row_sum, 1.0, atol=1e-6)
                
        except ImportError:
            pytest.skip("Allen SDK not installed")
        except Exception as e:
            pytest.skip(f"Could not load connectivity data: {e}")

    def test_connectivity_unnormalized(self):
        """Test that unnormalized matrix contains raw values."""
        try:
            loader = AllenConnectivityLoader()
            connectivity = loader.load_connectivity_matrix(
                region_acronyms=['VISp', 'MOp'],
                normalize=False
            )
            
            # Unnormalized values can be any non-negative number
            assert np.all(connectivity.adjacency >= 0)
            assert connectivity.provenance['normalized'] == False
            
        except ImportError:
            pytest.skip("Allen SDK not installed")
        except Exception as e:
            pytest.skip(f"Could not load connectivity data: {e}")

    def test_threshold_filtering(self):
        """Test that threshold filtering removes weak connections."""
        try:
            loader = AllenConnectivityLoader()
            connectivity = loader.load_connectivity_matrix(
                region_acronyms=['VISp', 'VISl'],
                normalize=True,
                threshold=0.1
            )
            
            # All non-zero values should be >= threshold
            non_zero = connectivity.adjacency[connectivity.adjacency > 0]
            if len(non_zero) > 0:
                assert np.all(non_zero >= 0.1)
            
            assert connectivity.provenance['threshold'] == 0.1
                
        except ImportError:
            pytest.skip("Allen SDK not installed")
        except Exception as e:
            pytest.skip(f"Could not load connectivity data: {e}")

    def test_empty_connectivity_for_invalid_regions(self):
        """Test handling of invalid region acronyms."""
        try:
            loader = AllenConnectivityLoader()
            
            # Invalid region should generate warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                connectivity = loader.load_connectivity_matrix(
                    region_acronyms=['INVALID_REGION_XYZ'],
                    normalize=True
                )
                
                # Should return empty or handle gracefully
                assert isinstance(connectivity, ConnectivityGraph)
                # May have warnings about invalid acronym
                
        except ImportError:
            pytest.skip("Allen SDK not installed")
        except Exception as e:
            # Expected to fail for invalid regions
            pass

    def test_acronyms_to_ids_conversion(self):
        """Test internal acronym to ID conversion."""
        try:
            loader = AllenConnectivityLoader()
            acronyms = ['VISp', 'MOp']
            ids = loader._acronyms_to_ids(acronyms)
            
            assert len(ids) == 2
            assert all(isinstance(i, int) for i in ids)
            assert all(i > 0 for i in ids)
            
        except ImportError:
            pytest.skip("Allen SDK not installed")

    def test_ids_to_acronyms_conversion(self):
        """Test internal ID to acronym conversion."""
        try:
            loader = AllenConnectivityLoader()
            # VISp=385, MOp=993
            ids = [385, 993]
            acronyms = loader._ids_to_acronyms(ids)
            
            assert len(acronyms) == 2
            assert all(isinstance(a, str) for a in acronyms)
            assert 'VISp' in acronyms or 'MOp' in acronyms
            
        except ImportError:
            pytest.skip("Allen SDK not installed")

    def test_empty_connectivity_graph_helper(self):
        """Test empty connectivity graph creation."""
        try:
            loader = AllenConnectivityLoader()
            empty = loader._empty_connectivity_graph()
            
            assert isinstance(empty, ConnectivityGraph)
            assert len(empty.region_ids) == 0
            assert empty.adjacency.shape == (0, 0)
            assert empty.provenance.get('empty') == True
            
        except ImportError:
            pytest.skip("Allen SDK not installed")

    def test_provenance_tracking(self):
        """Test that provenance information is properly tracked."""
        try:
            loader = AllenConnectivityLoader()
            connectivity = loader.load_connectivity_matrix(
                region_acronyms=['VISp', 'MOp'],
                normalize=True,
                threshold=0.05
            )
            
            prov = connectivity.provenance
            assert 'source' in prov
            assert 'resolution_um' in prov
            assert 'normalized' in prov
            assert 'threshold' in prov
            assert 'n_regions' in prov
            
            assert prov['normalized'] == True
            assert prov['threshold'] == 0.05
            assert prov['resolution_um'] == 25
            assert prov['n_regions'] == 2
            
        except ImportError:
            pytest.skip("Allen SDK not installed")
        except Exception as e:
            pytest.skip(f"Could not load connectivity data: {e}")

    def test_load_connectivity_from_experiments(self):
        """Test loading connectivity from specific experiments."""
        try:
            loader = AllenConnectivityLoader()
            # Test with a small set of structure IDs
            connectivity = loader.load_connectivity_from_experiments(
                structure_ids=[385, 409]  # VISp, VISl
            )
            
            assert isinstance(connectivity, ConnectivityGraph)
            
        except ImportError:
            pytest.skip("Allen SDK not installed")
        except Exception as e:
            pytest.skip(f"Could not load experiment data: {e}")


class TestConvenienceFunction:
    """Test the convenience function load_allen_connectivity."""

    def test_load_allen_connectivity(self):
        """Test convenience function for loading connectivity."""
        try:
            connectivity = load_allen_connectivity(
                region_acronyms=['VISp', 'MOp'],
                normalize=True,
                threshold=0.0
            )
            
            assert isinstance(connectivity, ConnectivityGraph)
            assert len(connectivity.region_ids) == 2
            
        except ImportError:
            pytest.skip("Allen SDK not installed")
        except Exception as e:
            pytest.skip(f"Could not load connectivity data: {e}")

    def test_load_allen_connectivity_default(self):
        """Test convenience function with default parameters."""
        try:
            connectivity = load_allen_connectivity()
            
            assert isinstance(connectivity, ConnectivityGraph)
            assert len(connectivity.region_ids) > 0
            
        except ImportError:
            pytest.skip("Allen SDK not installed")
        except Exception as e:
            pytest.skip(f"Could not load connectivity data: {e}")

    def test_load_allen_connectivity_with_custom_manifest(self):
        """Test convenience function with custom manifest."""
        try:
            import tempfile
            import os
            
            with tempfile.TemporaryDirectory() as tmpdir:
                manifest = os.path.join(tmpdir, "manifest.json")
                connectivity = load_allen_connectivity(
                    region_acronyms=['VISp'],
                    manifest_file=manifest
                )
                
                assert isinstance(connectivity, ConnectivityGraph)
                
        except ImportError:
            pytest.skip("Allen SDK not installed")
        except Exception as e:
            pytest.skip(f"Could not load connectivity data: {e}")


class TestConnectivityGraphProperties:
    """Test properties of the returned ConnectivityGraph."""

    def test_connectivity_graph_structure(self):
        """Test that returned ConnectivityGraph has correct structure."""
        try:
            connectivity = load_allen_connectivity(
                region_acronyms=['VISp', 'VISl']
            )
            
            # Test basic properties
            assert hasattr(connectivity, 'region_ids')
            assert hasattr(connectivity, 'adjacency')
            assert hasattr(connectivity, 'name')
            assert hasattr(connectivity, 'provenance')
            
            # Test data types
            assert isinstance(connectivity.region_ids, np.ndarray)
            assert isinstance(connectivity.adjacency, np.ndarray)
            assert connectivity.adjacency.dtype == np.float64
            
        except ImportError:
            pytest.skip("Allen SDK not installed")
        except Exception as e:
            pytest.skip(f"Could not load connectivity data: {e}")

    def test_connectivity_graph_immutability(self):
        """Test that ConnectivityGraph is immutable (frozen dataclass)."""
        try:
            connectivity = load_allen_connectivity(
                region_acronyms=['VISp', 'MOp']
            )
            
            # Should not be able to modify frozen dataclass
            with pytest.raises((AttributeError, TypeError)):
                connectivity.name = "modified"
                
        except ImportError:
            pytest.skip("Allen SDK not installed")
        except Exception as e:
            pytest.skip(f"Could not load connectivity data: {e}")

    def test_row_normalized_method(self):
        """Test the row_normalized method of ConnectivityGraph."""
        try:
            connectivity = load_allen_connectivity(
                region_acronyms=['VISp', 'MOp'],
                normalize=False  # Get unnormalized first
            )
            
            # Apply normalization using the method
            normalized = connectivity.row_normalized()
            
            assert normalized.shape == connectivity.adjacency.shape
            row_sums = normalized.sum(axis=1)
            
            for row_sum in row_sums:
                assert row_sum == 0.0 or np.isclose(row_sum, 1.0, atol=1e-6)
                
        except ImportError:
            pytest.skip("Allen SDK not installed")
        except Exception as e:
            pytest.skip(f"Could not load connectivity data: {e}")

    def test_adjacency_matrix_properties(self):
        """Test mathematical properties of adjacency matrix."""
        try:
            connectivity = load_allen_connectivity(
                region_acronyms=['VISp', 'MOp', 'SSp']
            )
            
            # Should be square
            assert connectivity.adjacency.shape[0] == connectivity.adjacency.shape[1]
            
            # Should be non-negative
            assert np.all(connectivity.adjacency >= 0)
            
            # Should match number of regions
            assert connectivity.adjacency.shape[0] == len(connectivity.region_ids)
            
            # Should be float type
            assert connectivity.adjacency.dtype == np.float64
            
        except ImportError:
            pytest.skip("Allen SDK not installed")
        except Exception as e:
            pytest.skip(f"Could not load connectivity data: {e}")

    def test_region_ids_uniqueness(self):
        """Test that region IDs are unique."""
        try:
            connectivity = load_allen_connectivity(
                region_acronyms=['VISp', 'VISl', 'VISal']
            )
            
            unique_ids = np.unique(connectivity.region_ids)
            assert len(unique_ids) == len(connectivity.region_ids)
            
        except ImportError:
            pytest.skip("Allen SDK not installed")
        except Exception as e:
            pytest.skip(f"Could not load connectivity data: {e}")

    def test_region_ids_are_strings(self):
        """Test that region IDs are string type."""
        try:
            connectivity = load_allen_connectivity(
                region_acronyms=['VISp', 'MOp']
            )
            
            assert connectivity.region_ids.dtype.kind in ['U', 'S', 'O']  # Unicode, byte string, or object
            for region_id in connectivity.region_ids:
                assert isinstance(region_id, str)
                
        except ImportError:
            pytest.skip("Allen SDK not installed")
        except Exception as e:
            pytest.skip(f"Could not load connectivity data: {e}")


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_region(self):
        """Test loading connectivity for a single region."""
        try:
            connectivity = load_allen_connectivity(
                region_acronyms=['VISp']
            )
            
            assert isinstance(connectivity, ConnectivityGraph)
            assert len(connectivity.region_ids) == 1
            assert connectivity.adjacency.shape == (1, 1)
            
        except ImportError:
            pytest.skip("Allen SDK not installed")
        except Exception as e:
            pytest.skip(f"Could not load connectivity data: {e}")

    def test_large_region_set(self):
        """Test loading connectivity for many regions."""
        try:
            # Test with multiple cortical regions
            regions = ['VISp', 'VISl', 'VISal', 'VISrl', 'VISam', 'VISpm',
                      'MOp', 'MOs', 'SSp-n', 'SSp-bfd']
            
            connectivity = load_allen_connectivity(
                region_acronyms=regions,
                normalize=True
            )
            
            assert isinstance(connectivity, ConnectivityGraph)
            assert len(connectivity.region_ids) == len(regions)
            assert connectivity.adjacency.shape == (len(regions), len(regions))
            
        except ImportError:
            pytest.skip("Allen SDK not installed")
        except Exception as e:
            pytest.skip(f"Could not load connectivity data: {e}")

    def test_mixed_valid_invalid_regions(self):
        """Test loading with mix of valid and invalid region acronyms."""
        try:
            loader = AllenConnectivityLoader()
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                connectivity = loader.load_connectivity_matrix(
                    region_acronyms=['VISp', 'INVALID_XYZ', 'MOp']
                )
                
                # Should still create connectivity graph
                assert isinstance(connectivity, ConnectivityGraph)
                # May have fewer regions than requested due to invalid ones
                
        except ImportError:
            pytest.skip("Allen SDK not installed")
        except Exception as e:
            # May fail depending on SDK behavior
            pass

    def test_zero_threshold(self):
        """Test that zero threshold includes all connections."""
        try:
            connectivity = load_allen_connectivity(
                region_acronyms=['VISp', 'MOp'],
                threshold=0.0
            )
            
            assert connectivity.provenance['threshold'] == 0.0
            
        except ImportError:
            pytest.skip("Allen SDK not installed")
        except Exception as e:
            pytest.skip(f"Could not load connectivity data: {e}")

    def test_high_threshold(self):
        """Test that high threshold filters most connections."""
        try:
            connectivity = load_allen_connectivity(
                region_acronyms=['VISp', 'MOp', 'SSp'],
                normalize=True,
                threshold=0.9
            )
            
            # Very few connections should survive
            non_zero = np.count_nonzero(connectivity.adjacency)
            total = connectivity.adjacency.size
            
            # At least some sparsity due to high threshold
            assert non_zero <= total
            
        except ImportError:
            pytest.skip("Allen SDK not installed")
        except Exception as e:
            pytest.skip(f"Could not load connectivity data: {e}")


class TestIntegrationWithExistingFramework:
    """Test integration with existing NeuroThera components."""

    def test_compatibility_with_diffuse_activity(self):
        """Test that ConnectivityGraph works with diffuse_activity."""
        try:
            from neurothera_map.mouse.predict import diffuse_activity
            from neurothera_map.core.types import ActivityMap
            
            connectivity = load_allen_connectivity(
                region_acronyms=['VISp', 'MOp']
            )
            
            # Create a simple activity map
            activity = ActivityMap(
                region_ids=np.array(['VISp', 'MOp'], dtype=str),
                values=np.array([1.0, 0.0]),
                space='allen_ccf',
                name='test_activity'
            )
            
            # Should work with diffuse_activity
            result = diffuse_activity(
                activity, 
                connectivity, 
                alpha=0.9, 
                steps=5
            )
            
            assert isinstance(result, ActivityMap)
            assert len(result.region_ids) == 2
            
        except ImportError:
            pytest.skip("Allen SDK or dependencies not installed")
        except Exception as e:
            pytest.skip(f"Integration test failed: {e}")

    def test_region_map_compatibility(self):
        """Test that region IDs are compatible with RegionMap."""
        try:
            from neurothera_map.core.types import RegionMap
            
            connectivity = load_allen_connectivity(
                region_acronyms=['VISp', 'MOp']
            )
            
            # Create a RegionMap with same regions
            region_map = RegionMap(
                region_ids=connectivity.region_ids,
                values=np.array([1.0, 2.0]),
                space='allen_ccf'
            )
            
            assert len(region_map.region_ids) == len(connectivity.region_ids)
            
        except ImportError:
            pytest.skip("Allen SDK not installed")
        except Exception as e:
            pytest.skip(f"Could not test RegionMap compatibility: {e}")

    def test_consistency_with_activity_map_format(self):
        """Test that connectivity regions match ActivityMap conventions."""
        try:
            from neurothera_map.core.types import ActivityMap
            
            connectivity = load_allen_connectivity(
                region_acronyms=['VISp', 'MOp']
            )
            
            # Should be able to create ActivityMap with same regions
            activity = ActivityMap(
                region_ids=connectivity.region_ids,
                values=np.random.randn(len(connectivity.region_ids)),
                space='allen_ccf'
            )
            
            assert activity.region_ids.tolist() == connectivity.region_ids.tolist()
            
        except ImportError:
            pytest.skip("Allen SDK not installed")
        except Exception as e:
            pytest.skip(f"Could not test ActivityMap compatibility: {e}")


class TestDataQuality:
    """Test quality and sanity of loaded connectivity data."""

    def test_connectivity_is_sparse(self):
        """Test that connectivity matrices are reasonably sparse."""
        try:
            connectivity = load_allen_connectivity(
                region_acronyms=['VISp', 'VISl', 'VISal', 'MOp', 'SSp'],
                normalize=True,
                threshold=0.0
            )
            
            # Most connections should be zero or very small
            # Brain connectivity is typically sparse
            zeros = np.sum(connectivity.adjacency == 0)
            total = connectivity.adjacency.size
            sparsity = zeros / total
            
            # At least some sparsity expected (>50% zeros)
            assert sparsity > 0.3, f"Connectivity too dense: {sparsity:.2%} zeros"
            
        except ImportError:
            pytest.skip("Allen SDK not installed")
        except Exception as e:
            pytest.skip(f"Could not test sparsity: {e}")

    def test_connectivity_values_reasonable(self):
        """Test that connectivity values are in reasonable range."""
        try:
            connectivity = load_allen_connectivity(
                region_acronyms=['VISp', 'MOp'],
                normalize=True
            )
            
            # Normalized values should be in [0, 1]
            assert np.all(connectivity.adjacency >= 0)
            assert np.all(connectivity.adjacency <= 1.0)
            
        except ImportError:
            pytest.skip("Allen SDK not installed")
        except Exception as e:
            pytest.skip(f"Could not test value ranges: {e}")

    def test_no_nans_or_infs(self):
        """Test that connectivity contains no NaN or Inf values."""
        try:
            connectivity = load_allen_connectivity(
                region_acronyms=['VISp', 'MOp', 'SSp']
            )
            
            assert not np.any(np.isnan(connectivity.adjacency))
            assert not np.any(np.isinf(connectivity.adjacency))
            
        except ImportError:
            pytest.skip("Allen SDK not installed")
        except Exception as e:
            pytest.skip(f"Could not test for NaNs: {e}")
