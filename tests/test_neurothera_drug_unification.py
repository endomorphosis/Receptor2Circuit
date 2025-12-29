from neurothera_map.drug.profile import build_drug_profile, convert_ingestion_profile_to_neurothera


def test_build_drug_profile_seed_mode_works_offline():
    dp = build_drug_profile("caffeine", mode="seed")
    assert dp.name == "caffeine"
    assert set(dp.targets()) == {"ADORA1", "ADORA2A"}


def test_build_drug_profile_auto_falls_back_without_network():
    # Avoid any network by forcing ingestion loader to not call adapters.
    dp = build_drug_profile("caffeine", mode="auto", use_iuphar=False, use_chembl=False)
    assert set(dp.targets()) == {"ADORA1", "ADORA2A"}


def test_convert_ingestion_profile_to_neurothera_synthetic():
    from drug.schemas import DrugProfile as IngestProfile, TargetInteraction, PotencyMeasure, PotencyUnit, InteractionType

    ingest = IngestProfile(
        common_name="demo",
        source_databases=["IUPHAR"],
        interactions=[
            TargetInteraction(
                target_gene_symbol="DRD2",
                interaction_type=InteractionType.ANTAGONIST,
                potency_measures=[PotencyMeasure(value=10.0, unit=PotencyUnit.NANOMOLAR, measure_type="Ki")],
                evidence_score=0.8,
                source_database="IUPHAR",
            )
        ],
    )

    dp = convert_ingestion_profile_to_neurothera(ingest)
    assert dp.name == "demo"
    assert dp.targets() == ["DRD2"]
    assert dp.interactions[0].affinity_nM == 10.0
