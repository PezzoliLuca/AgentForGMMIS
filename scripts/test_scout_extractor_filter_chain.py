#!/usr/bin/env python3
"""
Empirical Verification of the 3-Stage Defense-in-Depth Funnel:
  Stage 3 (`ScoutAgent` Title/Abstract Screening with 3-State Epistemic Logic)
      |
      v  (Promotes both True Positives and 'IMPLICIT_UNKNOWN' Stealth Traps)
  Full-Text Reading (`ExtractionAgent` with `ExtractionThoughtProcess` CoT)
      |
      v  (Extracts full-text `InclusionCriteriaCheck` + quantitative tables)
  Quality Gatekeeper (`AuditorAgent` 'Il Vigile')
      |
      v  (Certifies True Positives -> `VERIFIED`/`FLAGGED_WARNING`; Blocks Stealth Traps -> `REJECTED`)

Outputs a complete, structured Chain-of-Thought trace log at:
  `logs/scout_runs/defense_in_depth_chain_trace.md`
  `logs/scout_runs/defense_in_depth_chain_trace.json`
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent_gmmis.auditing.auditor import AuditorAgent
from agent_gmmis.auditing.schemas import AuditStatus
from agent_gmmis.config import DATA_DIR, DEFAULT_DB_PATH, LOGS_DIR
from agent_gmmis.data.db_manager import DatabaseManager
from agent_gmmis.extraction.extractor import ExtractionAgent
from agent_gmmis.extraction.pdf_parser import ParsedManuscript
from agent_gmmis.llm.client import LLMClient, LLMConfig
from agent_gmmis.scouting.schemas import RawPaperCandidate, ScoutSource, ScoutTier
from agent_gmmis.scouting.scout_agent import ScoutAgent


def build_test_scenarios() -> List[Dict[str, Any]]:
    """Constructs 4 end-to-end test scenarios (1 Real PDF True Positive + 3 Stealth False Positive Traps)."""
    medsam2_pdf = DATA_DIR / "cache_papers" / "2504.03600.pdf"
    return [
        {
            "scenario_id": "CASE_1_REAL_TP_MEDSAM2",
            "scenario_label": "Real 2025 Survey Model (MedSAM2 — Real 9MB PDF)",
            "expected_scout_decision": "PROMOTE",
            "expected_final_audit": "ACCEPTED",
            "raw_candidate": RawPaperCandidate(
                title="MedSAM2: Segment Anything in 3D Medical Images and Videos",
                abstract=(
                    "We present MedSAM2, a promptable 3D foundation model for volumetric medical image "
                    "and video segmentation across CT, MRI, PET, and ultrasound modalities, achieving "
                    "state-of-the-art performance across public benchmarks."
                ),
                authors=["Jun Ma", "Bo Wang"],
                publication_date="2025-04-04",
                venue="arXiv",
                arxiv_id="2504.03600",
                pdf_url="https://arxiv.org/pdf/2504.03600.pdf",
                sources=[ScoutSource.ARXIV],
                discovery_query="PRISMA Systematic + Agentic Query",
            ),
            "pdf_path": medsam2_pdf if medsam2_pdf.exists() else None,
            "manuscript": None,
        },
        {
            "scenario_id": "CASE_2_STEALTH_TRAP_PRIVATE_COHORT",
            "scenario_label": "Stealth Trap A: Ambiguous 3D MRI Abstract -> Private Hospital Cohort Only in Full Text",
            "expected_scout_decision": "PROMOTE",
            "expected_final_audit": "REJECTED",
            "raw_candidate": RawPaperCandidate(
                title="NeuroVol-Net: A 3D Vision Transformer for Automated Brain Lesion Segmentation in Multi-Sequence MRI",
                abstract=(
                    "We propose NeuroVol-Net, a novel 3D volumetric vision transformer architecture for automated "
                    "brain lesion segmentation in multi-sequence clinical MRI scans. Our method introduces cross-slice "
                    "volumetric attention and demonstrates superior boundary delineation in clinical evaluation."
                ),
                authors=["M. Rossi", "L. Bianchi"],
                publication_date="2025-05-10",
                venue="arXiv",
                arxiv_id="2505.99001",
                sources=[ScoutSource.ARXIV],
                discovery_query="Agentic Query: 3D Vision Transformer MRI segmentation",
            ),
            "pdf_path": None,
            "manuscript": ParsedManuscript(
                file_path="synthetic_2505.99001.pdf",
                title="NeuroVol-Net: A 3D Vision Transformer for Automated Brain Lesion Segmentation in Multi-Sequence MRI",
                abstract=(
                    "We propose NeuroVol-Net, a novel 3D volumetric vision transformer architecture for automated "
                    "brain lesion segmentation in multi-sequence clinical MRI scans."
                ),
                total_pages=8,
                pages_text={
                    1: (
                        "Title: NeuroVol-Net: A 3D Vision Transformer for Automated Brain Lesion Segmentation in Multi-Sequence MRI\n"
                        "Abstract: We propose NeuroVol-Net, a 3D volumetric vision transformer (ViT-B backbone, 92.4M parameters) "
                        "for brain lesion segmentation in multi-sequence MRI.\n"
                        "1. Introduction: Accurate 3D brain lesion segmentation in MRI is vital for surgical planning."
                    ),
                    2: (
                        "4. Experiments and Datasets:\n"
                        "4.1 Dataset Description (Private Institutional Cohort Only):\n"
                        "All training and evaluation experiments were conducted EXCLUSIVELY on an internal, private, "
                        "proprietary clinical cohort of 185 patients collected at Hospital San Raffaele (Internal-HSR-185). "
                        "Due to strict institutional patient privacy regulations, the dataset is confidential and NOT publicly available. "
                        "No public benchmarks (such as BraTS, ISLES, or MSD) were used in this study.\n\n"
                        "Table 1: Quantitative results on the private Internal-HSR-185 hospital cohort.\n"
                        "| Model | Dataset | Organ | Dice (%) |\n"
                        "| nnU-Net | Internal-HSR-185 (Private) | Brain Lesion | 81.20 |\n"
                        "| NeuroVol-Net (Ours) | Internal-HSR-185 (Private) | Brain Lesion | 84.50 |\n"
                    ),
                },
            ),
        },
        {
            "scenario_id": "CASE_3_STEALTH_TRAP_2D_NON_RADIOLOGICAL",
            "scenario_label": "Stealth Trap B: Vague Generalist Abstract (CDPDNet-style) -> 2D Retinal & Dermoscopy Only in Full Text",
            "expected_scout_decision": "PROMOTE",
            "expected_final_audit": "REJECTED",
            "raw_candidate": RawPaperCandidate(
                title="PromptMed-Net: Dynamic Prompt Distillation Network for Medical Image Segmentation",
                abstract=(
                    "We propose PromptMed-Net, a prompt-driven hybrid vision-encoder architecture with text guidance "
                    "for medical image segmentation. Extensive experiments across multiple public medical segmentation "
                    "benchmarks demonstrate that PromptMed-Net outperforms existing state-of-the-art methods."
                ),
                authors=["K. Zhang", "Y. Liu"],
                publication_date="2025-05-14",
                venue="arXiv",
                arxiv_id="2505.99002",
                sources=[ScoutSource.ARXIV],
                discovery_query="PRISMA Systematic Query",
            ),
            "pdf_path": None,
            "manuscript": ParsedManuscript(
                file_path="synthetic_2505.99002.pdf",
                title="PromptMed-Net: Dynamic Prompt Distillation Network for Medical Image Segmentation",
                abstract=(
                    "We propose PromptMed-Net, a prompt-driven hybrid vision-encoder architecture with text guidance "
                    "for medical image segmentation across multiple public benchmarks."
                ),
                total_pages=10,
                pages_text={
                    1: (
                        "Title: PromptMed-Net: Dynamic Prompt Distillation Network for Medical Image Segmentation\n"
                        "Abstract: We propose PromptMed-Net for medical image segmentation.\n"
                        "1. Introduction: We design a 2D prompt distillation network specifically for 2D optical medical photographs."
                    ),
                    2: (
                        "4. Experimental Setup and Datasets:\n"
                        "IMPORTANT: Our architecture is strictly a 2D network (`is_3d = False`) and is evaluated EXCLUSIVELY on "
                        "non-radiological 2D color photography modalities (`is_radiological = False`): specifically 2D retinal fundus "
                        "vessel images (DRIVE and STARE datasets) and 2D dermoscopic skin lesion photographs (ISIC-2018). "
                        "No 3D volumetric radiological imaging (no CT, no MRI, no PET) is supported or evaluated.\n\n"
                        "Table 1: 2D Retinal and Dermoscopy Segmentation Results (Dice %).\n"
                        "| Method | DRIVE (2D Retina) | ISIC-2018 (2D Dermoscopy) |\n"
                        "| TransUNet | 81.4 | 88.2 |\n"
                        "| PromptMed-Net (Ours) | 83.1 | 90.4 |\n"
                    ),
                },
            ),
        },
        {
            "scenario_id": "CASE_4_STEALTH_TRAP_NO_DICE_BASELINE_ONLY",
            "scenario_label": "Stealth Trap C: 3D Abdominal CT Abstract -> Full Text Uses Only Off-the-Shelf nnU-Net & HD95 (Zero Dice)",
            "expected_scout_decision": "PROMOTE",
            "expected_final_audit": "REJECTED",
            "raw_candidate": RawPaperCandidate(
                title="AbdoPipe-3D: Volumetric Deep Learning for Multi-Organ Abdominal CT Segmentation on BTCV",
                abstract=(
                    "We present AbdoPipe-3D, an automated volumetric deep learning workflow for 3D multi-organ "
                    "segmentation in abdominal CT scans evaluated on the public BTCV benchmark."
                ),
                authors=["H. Muller", "S. Weber"],
                publication_date="2025-05-19",
                venue="arXiv",
                arxiv_id="2505.99003",
                sources=[ScoutSource.OPENALEX],
                discovery_query="OpenAlex Keyword Search",
            ),
            "pdf_path": None,
            "manuscript": ParsedManuscript(
                file_path="synthetic_2505.99003.pdf",
                title="AbdoPipe-3D: Volumetric Deep Learning for Multi-Organ Abdominal CT Segmentation on BTCV",
                abstract="We present AbdoPipe-3D, an automated workflow for 3D abdominal CT segmentation on BTCV.",
                total_pages=7,
                pages_text={
                    1: (
                        "Title: AbdoPipe-3D: Volumetric Deep Learning for Multi-Organ Abdominal CT Segmentation on BTCV\n"
                        "1. Introduction: In this clinical engineering report, we do not propose any new neural network architecture. "
                        "Instead, we run the standard off-the-shelf nnU-Net baseline model and evaluate boundary distance only."
                    ),
                    2: (
                        "3. Quantitative Evaluation:\n"
                        "NOTE: We do NOT report Dice Similarity Coefficient (DSC) or IoU anywhere in this paper (`has_dice_metric = False`). "
                        "All evaluations are reported strictly in terms of 95th percentile Hausdorff Distance (HD95 in mm) using off-the-shelf nnU-Net.\n\n"
                        "Table 1: Boundary error on BTCV (HD95 in mm — lower is better; NO Dice scores reported).\n"
                        "| Model | BTCV Liver HD95 (mm) | BTCV Spleen HD95 (mm) |\n"
                        "| nnU-Net | 4.12 mm | 3.05 mm |\n"
                    ),
                },
            ),
        },
    ]


def run_defense_in_depth_test(provider: str = "gemini", model: str = "gemini-3.1-flash-lite", base_url: str | None = None) -> Dict[str, Any]:
    db = DatabaseManager(DEFAULT_DB_PATH)
    llm_cfg = LLMConfig(provider=provider, model=model, base_url=base_url)
    llm_client = LLMClient(llm_cfg)

    scout = ScoutAgent(
        llm_client=llm_client,
        db_manager=db,
        ignore_db_models_after_date="2025-04-01",
    )
    extractor = ExtractionAgent(llm_client=llm_client)
    auditor = AuditorAgent(db_manager=db)

    scenarios = build_test_scenarios()
    results: List[Dict[str, Any]] = []

    print("\n" + "=" * 80)
    print(f"🛡️  TESTING 3-STAGE DEFENSE-IN-DEPTH FUNNEL ({provider} / {model})")
    print("   [Stage 3: ScoutAgent (Abstract)] -> [ExtractionAgent (Full PDF)] -> [AuditorAgent ('Il Vigile')]")
    print("=" * 80)

    for idx, sc in enumerate(scenarios, 1):
        raw_cand: RawPaperCandidate = sc["raw_candidate"]
        print(f"\n[{idx}/{len(scenarios)}] {sc['scenario_label']}")
        print(f"   📄 Title: {raw_cand.title}")

        # Step 1: ScoutAgent Stage 3 Abstract Triage
        t0 = time.time()
        scout_cand = scout.screen_candidate_stage3(raw_cand)
        scout_time = round(time.time() - t0, 2)
        scout_promoted = scout_cand.final_tier in (ScoutTier.TIER_1_GENERALIST, ScoutTier.TIER_2_BENCHMARK_SOTA)
        scout_decision_str = "PROMOTE" if scout_promoted else "REJECT"

        print(f"   🔭 [1. ScoutAgent (Abstract Only)] -> {scout_decision_str} ({scout_cand.final_tier.value}) in {scout_time}s")
        if scout_cand.screening:
            print(
                f"      Tri-State: 3D={scout_cand.screening.is_3d_volumetric.value}, "
                f"Rad={scout_cand.screening.is_radiological.value}, "
                f"Public={scout_cand.screening.uses_public_benchmarks.value}"
            )
            print(f"      Scout CoT Summary: {scout_cand.screening.reasoning}")

        # Step 2: ExtractionAgent Full-Text Reading
        t1 = time.time()
        if sc["pdf_path"] is not None:
            manuscript = extractor.parser.parse(sc["pdf_path"])
        else:
            manuscript = sc["manuscript"]

        extracted_data = extractor.process_manuscript(manuscript)
        ext_time = round(time.time() - t1, 2)

        # Step 3: AuditorAgent Quality Gatekeeper
        audit_report = auditor.audit(extracted_data, manuscript=manuscript, benchmark_mode=False)
        final_accepted = audit_report.status != AuditStatus.REJECTED
        final_verdict_str = "ACCEPTED" if final_accepted else "REJECTED"

        print(
            f"   🔬 [2. ExtractionAgent (Full Text)] -> "
            f"is_included={extracted_data.screening.is_included} "
            f"(3D={extracted_data.screening.is_3d}, Rad={extracted_data.screening.is_radiological}, "
            f"Public={extracted_data.screening.is_public_dataset}, Dice={extracted_data.screening.has_dice_metric}) "
            f"| Extracted Scores={len(extracted_data.extracted_scores)} in {ext_time}s"
        )
        print(
            f"   👮 [3. AuditorAgent ('Il Vigile')] -> status={audit_report.status.value} ({final_verdict_str})"
        )
        if audit_report.rejection_reasons:
            print(f"      Rejection Reasons: {' | '.join(audit_report.rejection_reasons)}")

        screening_dump = extracted_data.screening.model_dump(mode="json")
        screening_dump["is_included"] = extracted_data.screening.is_included

        results.append(
            {
                "scenario_id": sc["scenario_id"],
                "scenario_label": sc["scenario_label"],
                "title": raw_cand.title,
                "abstract": raw_cand.abstract,
                "expected_scout_decision": sc["expected_scout_decision"],
                "actual_scout_decision": scout_decision_str,
                "scout_tier": scout_cand.final_tier.value,
                "scout_latency_s": scout_time,
                "scout_screening": scout_cand.screening.model_dump(mode="json") if scout_cand.screening else None,
                "extractor_latency_s": ext_time,
                "extractor_proposed_model_id": extracted_data.proposed_model_id,
                "extractor_thought_process": (
                    extracted_data.thought_process.model_dump(mode="json")
                    if extracted_data.thought_process
                    else None
                ),
                "extractor_fulltext_screening": screening_dump,
                "extractor_scores_count": len(extracted_data.extracted_scores),
                "extractor_scores_preview": [
                    s.model_dump(mode="json") for s in extracted_data.extracted_scores[:5]
                ],
                "auditor_status": audit_report.status.value,
                "auditor_rejection_reasons": audit_report.rejection_reasons,
                "auditor_anomalies": [a.model_dump(mode="json") for a in audit_report.anomalies],
                "expected_final_audit": sc["expected_final_audit"],
                "actual_final_audit": final_verdict_str,
                "chain_passed": (scout_decision_str == sc["expected_scout_decision"])
                and (final_verdict_str == sc["expected_final_audit"]),
            }
        )

    out_dir = LOGS_DIR / "scout_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = out_dir / "defense_in_depth_chain_trace.json"
    md_out = out_dir / "defense_in_depth_chain_trace.md"

    payload = {
        "timestamp": datetime.now().isoformat(),
        "provider": llm_client.config.provider,
        "model": llm_client.config.model,
        "all_passed": all(r["chain_passed"] for r in results),
        "scenarios": results,
    }
    json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_out.write_text(format_defense_in_depth_markdown(payload), encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"✅ DEFENSE-IN-DEPTH CHAIN VERIFICATION COMPLETE: {sum(1 for r in results if r['chain_passed'])}/{len(results)} scenarios matched!")
    print(f"   📄 Markdown CoT Trace Log : {md_out}")
    print(f"   💾 JSON Trace Artifact    : {json_out}")
    print("=" * 80 + "\n")
    return payload


def format_defense_in_depth_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# Defense-in-Depth Multi-Agent Funnel Trace (`ScoutAgent` $\\rightarrow$ `ExtractionAgent` $\\rightarrow$ `AuditorAgent`)",
        "",
        f"- **Timestamp**: `{payload['timestamp']}`",
        f"- **LLM Engine**: `{payload['provider']}` / `{payload['model']}`",
        f"- **Overall Verification Result**: **{'100% PASSED' if payload['all_passed'] else 'PARTIAL'}**",
        "",
        "## Executive Summary Table",
        "",
        "| Scenario | Stage 3 `ScoutAgent` (Abstract Only) | Full-Text `ExtractionAgent` (`is_included`) | Critic `AuditorAgent` Verdict | Database Action |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for r in payload["scenarios"]:
        sc_str = f"**{r['actual_scout_decision']}** (`{r['scout_tier']}`)"
        ext_inc = r["extractor_fulltext_screening"]["is_included"]
        ext_str = f"`is_included={ext_inc}` (`{r['extractor_scores_count']}` scores)"
        aud_str = f"**`{r['auditor_status']}`**"
        db_act = "✅ **INGEST INTO SQLITE**" if r["actual_final_audit"] == "ACCEPTED" else "🛑 **BLOCKED FROM DB**"
        lines.append(f"| {r['scenario_label']} | {sc_str} | {ext_str} | {aud_str} | {db_act} |")

    lines.extend(["", "---", "", "## Detailed Multi-Agent Chain-of-Thought (CoT) Traces", ""])

    for idx, r in enumerate(payload["scenarios"], 1):
        sc_data = r["scout_screening"] or {}
        sc_tp = sc_data.get("thought_process") or {}
        ext_tp = r["extractor_thought_process"] or {}
        ext_sc = r["extractor_fulltext_screening"] or {}

        lines.extend(
            [
                f"### {idx}. {r['scenario_label']}",
                "",
                f"- **Paper Title**: *{r['title']}*",
                f"- **Input Abstract**: > {r['abstract']}",
                "",
                "#### Step A — `ScoutAgent` (Abstract-Level Epistemic Triage & CoT)",
                f"- **Decision**: **{r['actual_scout_decision']}** (`{r['scout_tier']}`, Latency: `{r['scout_latency_s']}s`)",
                f"- **Epistemic Tri-State**: `is_3d_volumetric={sc_data.get('is_3d_volumetric')}`, `is_radiological={sc_data.get('is_radiological')}`, `uses_public_benchmarks={sc_data.get('uses_public_benchmarks')}`",
                "- **Structured `ScoutThoughtProcess` (CoT)**:",
                f"  1. *Task & Novelty Analysis*: {sc_tp.get('task_and_novelty_analysis', 'N/A')}",
                f"  2. *Dimensionality & Modality Analysis*: {sc_tp.get('dimensionality_and_modality_analysis', 'N/A')}",
                f"  3. *Benchmark & Paradigm Analysis*: {sc_tp.get('benchmark_and_paradigm_analysis', 'N/A')}",
                f"  4. *Executive Reasoning*: {sc_data.get('reasoning', 'N/A')}",
                "",
                "#### Step B — `ExtractionAgent` (Full-Text Reading & `ExtractionThoughtProcess` CoT)",
                f"- **Latency**: `{r['extractor_latency_s']}s` | **Proposed Model ID**: `{r['extractor_proposed_model_id']}` | **Extracted Quantitative Rows**: `{r['extractor_scores_count']}`",
                f"- **Full-Text Inclusion Booleans (`InclusionCriteriaCheck`)**: `is_included={ext_sc.get('is_included')}` (`is_3d={ext_sc.get('is_3d')}`, `is_radiological={ext_sc.get('is_radiological')}`, `is_public_dataset={ext_sc.get('is_public_dataset')}`, `has_dice_metric={ext_sc.get('has_dice_metric')}`)",
                "- **Structured `ExtractionThoughtProcess` (CoT)**:",
                f"  1. *Proposed Model Identification*: {ext_tp.get('proposed_model_identification', 'N/A')}",
                f"  2. *Primary Benchmark Tables*: {ext_tp.get('primary_benchmark_tables', 'N/A')}",
                f"  3. *Peak Performance Logic*: {ext_tp.get('peak_performance_logic', 'N/A')}",
                f"  4. *Ambiguities / Caveats*: {ext_tp.get('ambiguities_or_caveats', 'N/A')}",
                "",
                "#### Step C — `AuditorAgent` ('Il Vigile' — Final Gatekeeper Verdict)",
                f"- **Audit Status**: **`{r['auditor_status']}`** $\\rightarrow$ **{r['actual_final_audit']}**",
            ]
        )
        if r["auditor_rejection_reasons"]:
            lines.append("- **Rejection Reasons**:")
            for reason in r["auditor_rejection_reasons"]:
                lines.append(f"  - 🛑 `{reason}`")
        lines.extend(["", "---", ""])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test ScoutAgent -> ExtractionAgent -> AuditorAgent filter chain.")
    parser.add_argument("--provider", default="gemini", help="LLM provider (gemini, ollama, mufasa, mock)")
    parser.add_argument("--model", default="gemini-3.1-flash-lite", help="LLM model identifier")
    parser.add_argument("--base-url", default=None, help="Optional custom OpenAI-compatible base URL")
    args = parser.parse_args()

    run_defense_in_depth_test(provider=args.provider, model=args.model, base_url=args.base_url)


if __name__ == "__main__":
    main()
