"""Tests for human ActivityMap ingestion from parcellated tables.

These tests are designed to run offline using fixture data.
Optional NIfTI tests are skipped if neuroimaging libraries are not available.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neurothera_map.core.types import ActivityMap
from neurothera_map.human.activity import (
    NIBABEL_AVAILABLE,
    activity_map_from_parcellated_table,
)

# Check if optional dependencies are available for NIfTI tests
NIFTI_SUPPORT = NIBABEL_AVAILABLE


@pytest.fixture
def fixture_table_path():
    """Path to the offline fixture table in datasets/."""
    repo_root = Path(__file__).parent.parent
    return repo_root / "datasets" / "human_activity_parcellated_fixture.csv"


@pytest.fixture
def temp_csv_file(tmp_path):
    """Create a temporary CSV file for testing."""
    temp_file = tmp_path / "test_data.csv"
    temp_file.write_text("region,value\nACC,0.5\nDLPFC,0.3\nmPFC,-0.2\n")
    return temp_file


def test_activity_map_from_parcellated_table_basic(fixture_table_path):
    """Test basic loading from the fixture table."""
    am = activity_map_from_parcellated_table(fixture_table_path)

    assert isinstance(am, ActivityMap)
    assert len(am.region_ids) > 0
    assert len(am.values) == len(am.region_ids)
    assert am.space == "mni152"
    assert "source_file" in am.provenance


def test_activity_map_from_parcellated_table_region_values(fixture_table_path):
    """Test that specific regions and values are correctly loaded."""
    am = activity_map_from_parcellated_table(fixture_table_path)

    # Check that the fixture data is correctly parsed
    region_dict = am.to_dict()
    
    # The fixture should contain these regions
    assert "ACC" in region_dict
    assert "DLPFC" in region_dict
    assert "Hippocampus" in region_dict
    
    # Check specific values from fixture
    assert region_dict["ACC"] == 0.25
    assert region_dict["DLPFC"] == 0.15
    assert region_dict["mPFC"] == -0.10


def test_activity_map_from_parcellated_table_custom_columns(tmp_path):
    """Test loading with custom column names."""
    # Create a file with custom column names
    df = pd.DataFrame({
        "brain_region": ["A", "B", "C"],
        "activation": [1.0, 2.0, 3.0],
    })
    custom_path = tmp_path / "custom_columns.csv"
    df.to_csv(custom_path, index=False)

    am = activity_map_from_parcellated_table(
        custom_path,
        region_col="brain_region",
        value_col="activation",
    )

    assert len(am.region_ids) == 3
    assert set(am.region_ids) == {"A", "B", "C"}
    region_dict = am.to_dict()
    assert region_dict["A"] == 1.0
    assert region_dict["B"] == 2.0
    assert region_dict["C"] == 3.0


def test_activity_map_from_parcellated_table_custom_space_name(fixture_table_path):
    """Test that space and name parameters are correctly set."""
    am = activity_map_from_parcellated_table(
        fixture_table_path,
        space="custom_space",
        name="test_activity",
    )

    assert am.space == "custom_space"
    assert am.name == "test_activity"


def test_activity_map_from_parcellated_table_provenance(fixture_table_path):
    """Test that provenance information is correctly recorded."""
    am = activity_map_from_parcellated_table(
        fixture_table_path,
        region_col="region",
        value_col="value",
    )

    assert "source_file" in am.provenance
    assert "region_col" in am.provenance
    assert "value_col" in am.provenance
    assert "n_regions" in am.provenance
    
    assert am.provenance["region_col"] == "region"
    assert am.provenance["value_col"] == "value"
    assert am.provenance["n_regions"] == len(am.region_ids)


def test_activity_map_from_parcellated_table_file_not_found():
    """Test error handling for missing file."""
    with pytest.raises(FileNotFoundError):
        activity_map_from_parcellated_table("/nonexistent/path/file.csv")


def test_activity_map_from_parcellated_table_missing_region_column(temp_csv_file):
    """Test error handling when region column is missing."""
    df = pd.DataFrame({"wrong_col": ["A", "B"], "value": [1.0, 2.0]})
    df.to_csv(temp_csv_file, index=False)

    with pytest.raises(ValueError, match="Column 'region' not found"):
        activity_map_from_parcellated_table(temp_csv_file)


def test_activity_map_from_parcellated_table_missing_value_column(temp_csv_file):
    """Test error handling when value column is missing."""
    df = pd.DataFrame({"region": ["A", "B"], "wrong_col": [1.0, 2.0]})
    df.to_csv(temp_csv_file, index=False)

    with pytest.raises(ValueError, match="Column 'value' not found"):
        activity_map_from_parcellated_table(temp_csv_file)


def test_activity_map_from_parcellated_table_empty_file(temp_csv_file):
    """Test error handling for empty table."""
    df = pd.DataFrame({"region": [], "value": []})
    df.to_csv(temp_csv_file, index=False)

    with pytest.raises(ValueError, match="contains no data rows"):
        activity_map_from_parcellated_table(temp_csv_file)


def test_activity_map_from_parcellated_table_with_nan_values(temp_csv_file):
    """Test that rows with NaN values are handled gracefully."""
    df = pd.DataFrame({
        "region": ["A", "B", "C", "D"],
        "value": [1.0, np.nan, 3.0, 4.0],
    })
    df.to_csv(temp_csv_file, index=False)

    am = activity_map_from_parcellated_table(temp_csv_file)

    # Should skip the row with NaN
    assert len(am.region_ids) == 3
    assert set(am.region_ids) == {"A", "C", "D"}


def test_activity_map_from_parcellated_table_all_nan_values(temp_csv_file):
    """Test error handling when all values are NaN."""
    df = pd.DataFrame({
        "region": ["A", "B", "C"],
        "value": [np.nan, np.nan, np.nan],
    })
    df.to_csv(temp_csv_file, index=False)

    with pytest.raises(ValueError, match="No valid.*pairs found"):
        activity_map_from_parcellated_table(temp_csv_file)


def test_activity_map_from_parcellated_table_numeric_string_values(temp_csv_file):
    """Test that numeric values stored as strings are correctly parsed."""
    # Write CSV with string numbers
    with open(temp_csv_file, "w") as f:
        f.write("region,value\n")
        f.write("A,1.5\n")
        f.write("B,2.5\n")
    
    am = activity_map_from_parcellated_table(temp_csv_file)
    
    assert len(am.region_ids) == 2
    assert am.to_dict()["A"] == 1.5
    assert am.to_dict()["B"] == 2.5


def test_activity_map_from_parcellated_table_negative_values(fixture_table_path):
    """Test that negative values are correctly handled."""
    am = activity_map_from_parcellated_table(fixture_table_path)
    
    region_dict = am.to_dict()
    # mPFC has negative value in fixture
    assert region_dict["mPFC"] < 0


def test_activity_map_default_name_from_filename(tmp_path):
    """Test that default name is derived from filename."""
    # Create a temporary file with a specific name
    specific_path = tmp_path / "my_activity_data.csv"
    specific_path.write_text("region,value\nA,1.0\n")
    
    am = activity_map_from_parcellated_table(specific_path)
    # Name should be the stem (filename without extension)
    assert am.name == specific_path.stem


def test_nifti_import_available():
    """Test that NIfTI support is available when dependencies are installed."""
    import neurothera_map.human.activity as mod

    # Simulate nibabel being available.
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(mod, "NIBABEL_AVAILABLE", True)
        assert callable(mod.activity_map_from_nifti)
    finally:
        monkeypatch.undo()


def test_nifti_import_error_when_not_available():
    """Test that appropriate error is raised when NIfTI dependencies are missing."""
    import neurothera_map.human.activity as mod

    # Simulate nibabel missing regardless of the actual environment.
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(mod, "NIBABEL_AVAILABLE", False)
        with pytest.raises(ImportError, match="nibabel is required"):
            mod.activity_map_from_nifti("dummy.nii.gz", "atlas.nii.gz")
    finally:
        monkeypatch.undo()


def test_activity_map_from_nifti_smoke_with_fake_nibabel(tmp_path):
    """Smoke test the NIfTI path with a tiny fake nibabel implementation."""
    import neurothera_map.human.activity as mod

    # Create minimal dummy files (existence is checked).
    data_path = tmp_path / "data.nii.gz"
    atlas_path = tmp_path / "atlas.nii.gz"
    data_path.write_bytes(b"fake")
    atlas_path.write_bytes(b"fake")

    data = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=float,
    )
    atlas = np.array(
        [
            [[1, 1], [2, 2]],
            [[1, 1], [2, 2]],
        ],
        dtype=int,
    )

    class _FakeImg:
        def __init__(self, arr):
            self._arr = arr

        def get_fdata(self):
            return np.asarray(self._arr)

    class _FakeNib:
        @staticmethod
        def load(path):
            p = str(path)
            if p.endswith("data.nii.gz"):
                return _FakeImg(data)
            return _FakeImg(atlas)

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(mod, "NIBABEL_AVAILABLE", True)
        monkeypatch.setattr(mod, "nib", _FakeNib, raising=False)

        am = mod.activity_map_from_nifti(data_path, atlas_path, name="fake")
        assert len(am.region_ids) == 2
        out = am.to_dict()
        assert out["1"] == pytest.approx(3.5)
        assert out["2"] == pytest.approx(5.5)
    finally:
        monkeypatch.undo()


def test_activity_map_integration_with_core_types(fixture_table_path):
    """Test that ActivityMap returned integrates well with core types."""
    am = activity_map_from_parcellated_table(fixture_table_path)
    
    # Test to_dict() method inherited from RegionMap
    region_dict = am.to_dict()
    assert isinstance(region_dict, dict)
    assert all(isinstance(k, str) for k in region_dict.keys())
    assert all(isinstance(v, float) for v in region_dict.values())
    
    # Test reindex method inherited from RegionMap
    new_regions = ["ACC", "DLPFC", "NewRegion"]
    reindexed = am.reindex(new_regions)
    assert len(reindexed.region_ids) == 3
    assert np.isnan(reindexed.to_dict()["NewRegion"])  # Fill value for missing region


def test_activity_map_immutability(fixture_table_path):
    """Test that ActivityMap is immutable (frozen dataclass)."""
    am = activity_map_from_parcellated_table(fixture_table_path)
    
    # Should not be able to modify frozen dataclass
    with pytest.raises((AttributeError, TypeError)):
        am.name = "new_name"
    
    with pytest.raises((AttributeError, TypeError)):
        am.space = "new_space"
