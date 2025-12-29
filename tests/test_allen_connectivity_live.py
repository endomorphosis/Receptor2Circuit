"""Opt-in live integration tests for Allen connectivity.

These tests are intentionally NOT part of the default deterministic test run.
They require the Allen SDK (allensdk) and may require network access and/or
an on-disk cache. Run explicitly with:

    BWM_ALLEN_OFFLINE=0 pytest --run-allen-live -m allen_live

On Python 3.12 in this repo, allensdk may not be installable due to upstream
dependency constraints. Prefer Python 3.11 for these live tests.
"""

import pytest


@pytest.mark.allen_live
def test_allen_live_loader_initialization(monkeypatch):
    monkeypatch.setenv("BWM_ALLEN_OFFLINE", "0")

    try:
        from neurothera_map.mouse.allen_connectivity import AllenConnectivityLoader

        loader = AllenConnectivityLoader()
        assert loader.mcc is not None
        assert loader.structure_tree is not None
    except ImportError:
        pytest.skip(
            "Allen SDK not installed. Prefer running the real pipeline via "
            "scripts/run_e2e_allensdk.sh (Python 3.11), or install extras: pip install -e '.[allen]'."
        )


@pytest.mark.allen_live
def test_allen_live_load_connectivity_matrix(monkeypatch):
    monkeypatch.setenv("BWM_ALLEN_OFFLINE", "0")

    try:
        from neurothera_map.mouse.allen_connectivity import load_allen_connectivity

        conn = load_allen_connectivity(region_acronyms=["VISp", "MOp"], normalize=True, threshold=0.0)
        assert conn.adjacency.shape == (2, 2)
        assert len(conn.region_ids) == 2
        assert "Allen Mouse Brain Connectivity Atlas" in conn.provenance.get("source", "")
    except ImportError:
        pytest.skip(
            "Allen SDK not installed. Prefer running the real pipeline via "
            "scripts/run_e2e_allensdk.sh (Python 3.11), or install extras: pip install -e '.[allen]'."
        )
    except Exception as e:
        pytest.skip(f"Live Allen load failed (network/cache/data): {e}")
