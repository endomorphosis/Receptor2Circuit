import numpy as np

from neurothera_map.human.activity import activity_map_from_parcellated_table
from neurothera_map.human.receptors import load_human_pet_receptor_maps
from neurothera_map.human.transcriptomics import load_transcriptomic_map_from_csv


def test_human_pet_receptor_loader_fixture():
    rm = load_human_pet_receptor_maps("datasets/human_pet_receptor_fixture.csv")
    assert {"D1", "D2"}.issubset(set(rm.receptor_names()))
    assert "5HT1a" in set(rm.receptor_names())
    d1 = rm.get("D1")
    assert d1 is not None
    # Loader sorts region ids lexicographically.
    assert d1.region_ids.tolist() == sorted(d1.region_ids.tolist())


def test_human_transcriptomics_loader_fixture():
    tm = load_transcriptomic_map_from_csv("datasets/human_ahba_expression_fixture.csv")
    assert {"DRD1", "DRD2", "HTR1A"}.issubset(set(tm.receptor_names()))
    drd2 = tm.get("DRD2")
    assert drd2 is not None
    assert drd2.region_ids.tolist() == sorted(drd2.region_ids.tolist())


def test_human_activity_from_parcellated_table_fixture():
    am = activity_map_from_parcellated_table("datasets/human_activity_parcellated_fixture.csv")
    assert am.region_ids.tolist() == ["ACC", "DLPFC", "mPFC", "Hippocampus", "Amygdala", "Thalamus", "Striatum", "Insula"]
    assert np.allclose(am.values, np.array([0.25, 0.15, -0.10, 0.05, 0.30, 0.20, 0.12, 0.08]))
