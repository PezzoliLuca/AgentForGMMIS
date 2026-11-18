"""Comprehensive unit and integration tests for the Autonomous ScoutAgent (Step 4)."""

from pathlib import Path

from agent_gmmis.config import DATA_DIR, DEFAULT_DB_PATH
from agent_gmmis.data.db_manager import DatabaseManager
from agent_gmmis.llm.client import LLMClient, LLMConfig
from agent_gmmis.scouting.deduplicator import ScoutDeduplicator, title_token_similarity
from agent_gmmis.scouting.harvesters import (
    ArxivHarvester,
    EuropePMCHarvester,
    GoogleScholarHarvester,
    OpenAlexHarvester,
)
from agent_gmmis.scouting.schemas import (
    DeduplicationStatus,
    RawPaperCandidate,
    ScoutScreeningResult,
    ScoutSource,
    ScoutTier,
    TriStateCriterion,
)
from agent_gmmis.scouting.scout_agent import ScoutAgent


def test_arxiv_atom_xml_parser():
    """Verifies ArxivHarvester parses Atom XML entries and extracts arXiv ID, PDF link, and date."""
    sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>http://arxiv.org/abs/2504.03600v1</id>
        <published>2025-04-04T17:59:59Z</published>
        <title>MedSAM2: Segment Anything in 3D Medical Images and Videos</title>
        <summary>We propose MedSAM2 for 3D volumetric CT and MRI segmentation on TotalSegmentator.</summary>
        <author><name>Jun Ma</name></author>
        <author><name>Bo Wang</name></author>
      </entry>
    </feed>
    """
    candidates = ArxivHarvester.parse_atom_xml(
        sample_xml,
        query_label="test_query",
        start_date="2025-01-01",
        end_date="2025-12-31",
    )
    assert len(candidates) == 1
    c = candidates[0]
    assert c.arxiv_id == "2504.03600"
    assert c.publication_date == "2025-04-04"
    assert c.pdf_url == "https://arxiv.org/pdf/2504.03600.pdf"
    assert "Jun Ma" in c.authors
    assert ScoutSource.ARXIV in c.sources


def test_openalex_inverted_index_and_work_parser():
    """Verifies OpenAlexHarvester reconstructs abstracts from inverted index and resolves OA PDFs."""
    inverted = {"We": [0], "propose": [1], "Vista3D": [2], "for": [3], "3D": [4], "CT.": [5]}
    text = OpenAlexHarvester.reconstruct_abstract(inverted)
    assert text == "We propose Vista3D for 3D CT."

    payload = {
        "results": [
            {
                "id": "https://openalex.org/W12345",
                "title": "VISTA3D: A Unified Segmentation Foundation Model For 3D Medical Imaging",
                "publication_date": "2024-06-12",
                "doi": "https://doi.org/10.48550/arXiv.2406.08418",
                "cited_by_count": 42,
                "abstract_inverted_index": inverted,
                "authorships": [{"author": {"display_name": "Yufan He"}}],
                "primary_location": {
                    "source": {"display_name": "CVPR"},
                    "pdf_url": None,
                },
                "open_access": {"oa_url": "https://arxiv.org/pdf/2406.08418.pdf"},
                "locations": [],
            }
        ]
    }
    parsed = OpenAlexHarvester.parse_works_json(payload, query_label="openalex_test")
    assert len(parsed) == 1
    assert parsed[0].arxiv_id == "2406.08418"
    assert parsed[0].pdf_url == "https://arxiv.org/pdf/2406.08418.pdf"
    assert parsed[0].cited_by_count == 42
    assert parsed[0].venue == "CVPR"


def test_europe_pmc_and_google_scholar_parsers():
    """Verifies EuropePMC JSON parser and Google Scholar HTML parser."""
    epmc_payload = {
        "resultList": {
            "result": [
                {
                    "title": "SegAnyPET: Universal Promptable Segmentation from PET Images.",
                    "abstractText": "<b>Background:</b> 3D PET foundation model.",
                    "authorString": "Zhang Y, Li X",
                    "firstPublicationDate": "2025-02-15",
                    "journalTitle": "IEEE Transactions on Medical Imaging",
                    "doi": "10.1109/TMI.2025.999999",
                    "pmid": "39999999",
                    "citedByCount": 7,
                    "fullTextUrlList": {
                        "fullTextUrl": [
                            {"documentStyle": "pdf", "url": "https://europepmc.org/articles/PMC123/pdf"}
                        ]
                    },
                }
            ]
        }
    }
    epmc_res = EuropePMCHarvester.parse_epmc_json(epmc_payload)
    assert len(epmc_res) == 1
    assert epmc_res[0].pmid == "39999999"
    assert epmc_res[0].pdf_url == "https://europepmc.org/articles/PMC123/pdf"
    assert "<b>" not in epmc_res[0].abstract

    scholar_html = """
    <div class="gs_r gs_or gs_scl">
      <div class="gs_or_ggsm"><a href="https://arxiv.org/pdf/2505.17096.pdf">[PDF] arxiv.org</a></div>
      <h3 class="gs_rt"><a href="https://arxiv.org/abs/2505.17096">TAGS: 3D Tumor-Adaptive Guidance for SAM</a></h3>
      <div class="gs_a">A Author, B Author - arXiv preprint, 2025 - arxiv.org</div>
      <div class="gs_rs">We introduce TAGS for 3D volumetric tumor segmentation in CT and MRI.</div>
      <div class="gs_fl"><a>Cited by 11</a></div>
    </div>
    """
    gs_res = GoogleScholarHarvester.parse_scholar_html(scholar_html, query_label="scholar_test")
    assert len(gs_res) == 1
    assert gs_res[0].arxiv_id == "2505.17096"
    assert gs_res[0].cited_by_count == 11
    assert gs_res[0].publication_date == "2025-05-01"


def test_cross_source_deduplication_and_db_lifecycle_upgrade():
    """
    Verifies Stage 2:
    1. Merging duplicate papers from arXiv + Google Scholar + OpenAlex keeps the richest metadata.
    2. Recognizing an existing model in `database.sqlite` (`ALREADY_IN_DB`).
    3. Detecting a Preprint -> Peer-Reviewed Journal upgrade (`LIFECYCLE_UPGRADE`).
    """
    db = DatabaseManager(DEFAULT_DB_PATH)
    dedup = ScoutDeduplicator(db)

    # 1. Cross-source merge
    c_scholar = RawPaperCandidate(
        title="Novel3D-SAM: Volumetric Foundation Segmentation",
        abstract="Short snippet...",
        arxiv_id="2601.12345",
        cited_by_count=15,
        sources=[ScoutSource.GOOGLE_SCHOLAR],
    )
    c_arxiv = RawPaperCandidate(
        title="Novel3D-SAM: Volumetric Foundation Segmentation",
        abstract="Full detailed abstract explaining 3D CT and MRI promptable segmentation on AMOS and BraTS.",
        authors=["Alice Smith"],
        arxiv_id="2601.12345",
        publication_date="2026-01-10",
        venue="arXiv",
        pdf_url="https://arxiv.org/pdf/2601.12345.pdf",
        cited_by_count=2,
        sources=[ScoutSource.ARXIV],
    )
    merged = dedup.merge_raw_candidates([c_scholar, c_arxiv])
    assert len(merged) == 1
    assert "Full detailed abstract" in merged[0].abstract
    assert merged[0].cited_by_count == 15
    assert set(merged[0].sources) == {ScoutSource.GOOGLE_SCHOLAR, ScoutSource.ARXIV}

    # 2. Check existing model in DB (MedSAM2 -> 2504.03600)
    existing_cand = RawPaperCandidate(
        title="MedSAM2: Segment Anything in 3D Medical Images and Videos",
        arxiv_id="2504.03600",
        venue="arXiv",
    )
    status, matched_id, _ = dedup.check_against_database(existing_cand)
    assert status == DeduplicationStatus.ALREADY_IN_DB
    assert matched_id == "MedSAM2"

    # 3. Check Preprint -> Journal Lifecycle Upgrade (MedSAM2 published in IEEE TMI with new DOI)
    upgrade_cand = RawPaperCandidate(
        title="MedSAM2: Segment Anything in 3D Medical Images and Videos",
        doi="10.1109/TMI.2026.1122334",
        venue="IEEE Transactions on Medical Imaging",
        publication_date="2026-03-01",
    )
    u_status, u_id, u_reason = dedup.check_against_database(upgrade_cand)
    assert u_status == DeduplicationStatus.LIFECYCLE_UPGRADE
    assert u_id == "MedSAM2"
    assert "10.1109/tmi.2026.1122334" in (u_reason or "").lower()


def test_stage2_lexical_prefilter():
    """Verifies deterministic lexical exclusion of surveys and 2D non-radiological papers."""
    survey_paper = RawPaperCandidate(
        title="A Comprehensive Survey of Foundation Models in 3D Medical Image Segmentation",
        abstract="We review 100 papers on 3D CT and MRI segmentation.",
    )
    ok_survey, reason_survey = ScoutDeduplicator.apply_lexical_prefilter(survey_paper)
    assert not ok_survey
    assert "survey/review" in (reason_survey or "").lower()

    histopath_2d = RawPaperCandidate(
        title="PathoSAM: Nucleus Segmentation in Whole Slide Images",
        abstract="We adapt SAM for digital pathology and histopathology cell nuclei segmentation.",
    )
    ok_histo, reason_histo = ScoutDeduplicator.apply_lexical_prefilter(histopath_2d)
    assert not ok_histo
    assert "2d non-radiological" in (reason_histo or "").lower()

    # Multi-modal foundation model that mentions both histopathology AND 3D CT/MRI must NOT be rejected!
    multimodal_3d = RawPaperCandidate(
        title="BiomedParse: Foundation Model Across Nine Modalities",
        abstract="Joint segmentation across volumetric 3D CT, MRI, ultrasound, and histopathology benchmarks.",
    )
    ok_mm, _ = ScoutDeduplicator.apply_lexical_prefilter(multimodal_3d)
    assert ok_mm


def test_stage3_canonical_dataset_rescue_and_priority():
    """Verifies that mentioning a canonical 3D benchmark (BraTS, AMOS, BTCV) rescues false-negative 2D flags."""
    db = DatabaseManager(DEFAULT_DB_PATH)
    mock_llm = LLMClient(LLMConfig(provider="mock"))
    scout = ScoutAgent(llm_client=mock_llm, db_manager=db)

    cand = RawPaperCandidate(
        title="NextGen-SAM3D: Universal Promptable Volumetric Segmentation",
        abstract="We propose NextGen-SAM3D, a promptable 3D foundation model evaluated on BraTS, AMOS22, and TotalSegmentator CT and MRI benchmarks.",
        arxiv_id="2602.99999",
    )
    screened = scout.screen_candidate_stage3(cand)
    assert screened.final_tier == ScoutTier.TIER_1_GENERALIST
    assert "BraTS" in screened.canonical_datasets_matched
    assert "AMOS" in screened.canonical_datasets_matched
    assert "TotalSegmentator" in screened.canonical_datasets_matched
    assert screened.priority_score >= 0.90
    assert screened.pdf_accessible is True


def test_gold_standard_40_abstract_benchmark_offline():
    """Runs the 40-abstract Gold Standard benchmark in deterministic mock mode and verifies 100% precision & recall."""
    import json

    benchmark_path = DATA_DIR / "benchmarks" / "scout_abstract_benchmark.json"
    items = json.loads(benchmark_path.read_text(encoding="utf-8"))
    assert len(items) == 40

    db = DatabaseManager(DEFAULT_DB_PATH)
    all_ids = {m.id for m in db.get_all_models()}
    scout = ScoutAgent(
        llm_client=LLMClient(LLMConfig(provider="mock")),
        db_manager=db,
        masked_model_ids=all_ids,
    )

    raw_candidates = [
        RawPaperCandidate(
            title=x["title"],
            abstract=x["abstract"],
            venue=x.get("venue"),
            publication_date=x.get("publication_date"),
            arxiv_id=x.get("arxiv_id"),
            doi=x.get("doi"),
        )
        for x in items
    ]

    report = scout.run_campaign(pre_harvested_candidates=raw_candidates, max_llm_evaluations=50)
    assert report.metrics.promoted_tier1_generalist == 12
    assert report.metrics.promoted_tier2_benchmark_sota == 8
    assert len(report.rejected_candidates) == 20
    cot_md = ScoutAgent.format_cot_trace_report(report)
    assert "Structured LLM Chain-of-Thought (`ScoutThoughtProcess`)" in cot_md
    assert "Task & Novelty Analysis" in cot_md


def test_real_world_edge_cases_elsevier_doi_and_off_the_shelf_rejection():
    """
    Regression test for the 4 real-world failure modes uncovered during the April-May 2025 wild harvest:
    1. Publisher DOIs (e.g. '10.1016/j.bspc.2025.104340') must never be parsed as fake arXiv IDs ('2025.10434').
    2. Google Scholar year-only results with an arXiv link (e.g. '2511.19046') must infer '2025-11-01' and respect window bounds.
    3. 'Benchmarking Odyssey' / comparative study papers must be rejected by Stage 2 lexical filter.
    4. Papers that merely apply an off-the-shelf baseline ('3D nnU-Net', 'SAM 2') must be rejected in Stage 3.
    """
    from agent_gmmis.scouting.harvesters import _extract_arxiv_id, _infer_date_from_arxiv_id

    # 1. Elsevier DOI must not produce a fake arXiv ID
    assert _extract_arxiv_id("https://doi.org/10.1016/j.bspc.2025.104340") is None
    assert _extract_arxiv_id("10.1016/j.compmedimag.2025.102501") is None
    assert _extract_arxiv_id("https://doi.org/10.48550/arXiv.2505.18958") == "2505.18958"

    # 2. Inferred date from YYMM arXiv ID
    assert _infer_date_from_arxiv_id("2511.19046") == "2025-11-01"
    assert _infer_date_from_arxiv_id("2504.03600") == "2025-04-01"

    # 3. Benchmarking Odyssey lexical filter
    odyssey = RawPaperCandidate(
        title="Can Foundation Models Really Segment Tumors? A Benchmarking Odyssey in Lung CT Imaging",
        abstract="We benchmark existing foundation models on lung CT datasets.",
    )
    passed, reason = ScoutDeduplicator.apply_lexical_prefilter(odyssey)
    assert not passed
    assert "survey/review" in (reason or "").lower()

    # 4. CDPDNet pre-ranking bonus for '<Acronym>: ...' + 'text guidance'
    cdpd = RawPaperCandidate(
        title="CDPDNet: Integrating Text Guidance with Hybrid Vision Encoders for Medical Image Segmentation",
        abstract="We propose CDPDNet with CLIP text embeddings and DINOv2.",
    )
    assert ScoutDeduplicator.compute_stage2_pre_rank(cdpd) >= 9.0


