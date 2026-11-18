"""Unit tests for dataset & organ alias resolvers and 2D photographic modality filtering."""

from agent_gmmis.extraction.alias_resolver import (
    DatasetAliasResolver,
    OrganAliasResolver,
    is_non_3d_photographic_score,
)


def test_dataset_alias_resolver_new_benchmarks():
    resolver = DatasetAliasResolver()
    assert resolver.resolve("CrossMoDA") == "CrossMoDA"
    assert resolver.resolve("crossmoda23") == "CrossMoDA"
    assert resolver.resolve("CrossMoDA 2022") == "CrossMoDA"
    assert resolver.resolve("atlas2023") == "ATLAS 2023"


def test_organ_alias_resolver_canonical_mappings():
    assert OrganAliasResolver.resolve("Overall") == "Dataset-Level"
    assert OrganAliasResolver.resolve("Mean") == "Dataset-Level"
    assert OrganAliasResolver.resolve("mDice") == "Dataset-Level"
    assert OrganAliasResolver.resolve("Multi-organ mean") == "Dataset-Level"
    assert OrganAliasResolver.is_dataset_level("Avg") is True
    assert OrganAliasResolver.is_dataset_level("Liver") is False

    assert OrganAliasResolver.resolve("WT") == "Whole Tumor (WT)"
    assert OrganAliasResolver.resolve("ET") == "Enhancing Tumor (ET)"
    assert OrganAliasResolver.resolve("TC") == "Tumor Core (TC)"
    assert OrganAliasResolver.resolve("Liv") == "Liver"
    assert OrganAliasResolver.resolve("Kidney (L)") == "Left Kidney"
    assert OrganAliasResolver.resolve("Kidney (R)") == "Right Kidney"
    assert OrganAliasResolver.resolve("Myo") == "Myocardium"
    # Unknown organs gracefully title-case
    assert OrganAliasResolver.resolve("thyroid gland") == "Thyroid Gland"


def test_is_non_3d_photographic_score():
    assert is_non_3d_photographic_score("ISIC-2018", "Dermoscopy", "Skin Lesion") is True
    assert is_non_3d_photographic_score("DRIVE", "Retinal Fundus", "Vessels") is True
    assert is_non_3d_photographic_score("MoNuSeg", "Histopathology", "Nuclei") is True

    assert is_non_3d_photographic_score("BTCV", "CT", "Liver") is False
    assert is_non_3d_photographic_score("BraTS", "MRI", "Whole Tumor (WT)") is False
    assert is_non_3d_photographic_score("ACDC", "Cardiac MRI", "Myocardium") is False

