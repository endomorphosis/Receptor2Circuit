# Allen Connectivity Loader - Test Coverage Report

## Test Statistics

### Test Count
```
Total test methods: 36
Test classes: 6
Total lines: 708
```

## Test Classes and Methods


### TestAllenConnectivityLoader
    - test_initialization
    - test_initialization_with_custom_resolution
    - test_initialization_with_manifest_file
    - test_get_available_structures
    - test_load_connectivity_matrix_with_acronyms
    - test_load_connectivity_matrix_with_region_ids
    - test_load_connectivity_matrix_default
    - test_connectivity_normalization
    - test_connectivity_unnormalized
    - test_threshold_filtering
    - test_empty_connectivity_for_invalid_regions
    - test_acronyms_to_ids_conversion
    - test_ids_to_acronyms_conversion
    - test_empty_connectivity_graph_helper
    - test_provenance_tracking
    - test_load_connectivity_from_experiments

### TestConvenienceFunction
    - test_load_allen_connectivity
    - test_load_allen_connectivity_default
    - test_load_allen_connectivity_with_custom_manifest

### TestConnectivityGraphProperties
    - test_connectivity_graph_structure
    - test_connectivity_graph_immutability
    - test_row_normalized_method
    - test_adjacency_matrix_properties
    - test_region_ids_uniqueness
    - test_region_ids_are_strings

### TestEdgeCases
    - test_single_region
    - test_large_region_set
    - test_mixed_valid_invalid_regions
    - test_zero_threshold
    - test_high_threshold

### TestIntegrationWithExistingFramework
    - test_compatibility_with_diffuse_activity
    - test_region_map_compatibility
    - test_consistency_with_activity_map_format

### TestDataQuality
    - test_connectivity_is_sparse
    - test_connectivity_values_reasonable
    - test_no_nans_or_infs
