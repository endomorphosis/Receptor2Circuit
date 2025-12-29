import os
from pathlib import Path

import numpy as np
import pytest


def _require_installed(module: str) -> None:
    try:
        __import__(module)
    except Exception as e:  # pragma: no cover
        raise AssertionError(f"Required dependency '{module}' is not usable: {e}")


def _load_mapping(path: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        assert header[:2] == ["mouse_region", "human_region"]
        for line in f:
            line = line.strip()
            if not line:
                continue
            mouse_region, human_region = line.split(",", 1)
            mapping[mouse_region] = human_region
    return mapping


def test_e2e_allensdk_pipeline_real_dependencies(tmp_path):
    """True end-to-end test.

    This test is expected to FAIL (not skip) if:
    - allensdk is missing/unusable
    - nibabel is missing/unusable
    - the Allen connectivity loader cannot fetch/produce a matrix

    Run via the installer script: scripts/run_e2e_allensdk.sh
    """

    # Ensure we are using the real Allen SDK path.
    os.environ["BWM_ALLEN_OFFLINE"] = "0"

    _require_installed("allensdk")
    _require_installed("nibabel")

    from neurothera_map import build_drug_profile, translate_to_human, validate_against_pet_and_fmri
    from neurothera_map.human.activity import activity_map_from_nifti
    from neurothera_map.human.receptors import load_human_pet_receptor_maps
    from neurothera_map.mouse.expression import load_receptor_map_from_csv
    from neurothera_map.mouse.mvp_predict import predict_mouse_effects
    from neurothera_map.mouse.allen_connectivity import load_allen_connectivity

    # 1) Real Allen connectivity (may download/cache data)
    # Use a persistent cache dir so repeated runs do not re-download data.
    cache_dir = Path(os.environ.get("BWM_E2E_CACHE_DIR", str(tmp_path / "allen_cache")))
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = cache_dir / "allen_manifest.json"
    graph = load_allen_connectivity(region_acronyms=["VISp", "MOp"], manifest_file=str(manifest))
    assert graph.adjacency.shape == (2, 2)
    assert graph.region_ids.tolist() == ["VISp", "MOp"]

    # 2) Drug profile + mouse receptor expression fixture + prediction
    drug = build_drug_profile("caffeine", mode="seed")
    receptor_map = load_receptor_map_from_csv("datasets/mouse_receptor_expression_fixture.csv")

    mouse_pred = predict_mouse_effects(drug, receptor_map, graph, alpha=0.9, steps=5)
    assert mouse_pred.region_ids.tolist() == graph.region_ids.tolist()
    assert np.all(np.isfinite(mouse_pred.values))

    # 3) Translation using a small mapping fixture (ensures translation hook works)
    mapping = _load_mapping("datasets/cross_species_region_map_fixture.csv")
    human_pred = translate_to_human(mouse_pred, human_space="mni152", region_id_map=mapping)

    # 4) PET loader + validation hook
    pet = load_human_pet_receptor_maps("datasets/human_pet_receptor_cross_species_fixture.csv", space="mni152")
    report = validate_against_pet_and_fmri(human_pred, pet_receptors=pet, top_n=5)

    ranked = report.ranked_items.get("pet_receptor_correlations")
    assert ranked is not None
    assert len(ranked) >= 1
    assert ranked[0][0] == "ADORA1"
    assert np.isfinite(ranked[0][1])

    # 5) NIfTI end-to-end: build real NIfTI files with nibabel and run ingestion
    import nibabel as nib

    data = np.zeros((2, 2, 2), dtype=float)
    data[0, :, :] = 1.0
    data[1, :, :] = 2.0
    atlas = np.zeros((2, 2, 2), dtype=np.int16)
    atlas[0, :, :] = 1
    atlas[1, :, :] = 2

    affine = np.eye(4)
    data_img = nib.Nifti1Image(data, affine)
    atlas_img = nib.Nifti1Image(atlas, affine)

    data_path = tmp_path / "data.nii.gz"
    atlas_path = tmp_path / "atlas.nii.gz"
    nib.save(data_img, str(data_path))
    nib.save(atlas_img, str(atlas_path))

    am = activity_map_from_nifti(data_path, atlas_path, space="mni152", name="e2e_nifti")
    assert set(am.region_ids.tolist()) == {"1", "2"}
    out = am.to_dict()
    assert out["1"] == pytest.approx(1.0)
    assert out["2"] == pytest.approx(2.0)
