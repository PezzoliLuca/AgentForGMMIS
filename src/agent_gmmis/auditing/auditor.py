"""AuditorAgent ('Il Vigile'): Autonomous critic and quality assurance engine."""

import re
from typing import Dict, List, Optional, Tuple
from agent_gmmis.auditing.schemas import (
    AuditAnomaly,
    AuditReport,
    AuditStatus,
    DiscrepancyNotice,
    MissingDataDiagnosis,
    OutlierDetectionResult,
)
from agent_gmmis.data.db_manager import DatabaseManager
from agent_gmmis.data.enums import ScoreCategory
from agent_gmmis.extraction.alias_resolver import DatasetAliasResolver
from agent_gmmis.extraction.pdf_parser import ParsedManuscript
from agent_gmmis.extraction.schemas import ExtractedPaperData, RawExtractedScore

# Ubiquitous task-specific baseline models across generalist segmentation literature
KNOWN_BASELINE_MODELS = {
    "nnu-net",
    "nnunet",
    "unetr",
    "swin-unetr",
    "swin unetr",
    "swinunetr",
    "transunet",
    "nnformer",
    "segresnet",
    "v-net",
    "u-net",
    "unet",
    "d-lka net",
    "med-sa",
    "sam3d",
}


class AuditorAgent:
    """
    Autonomous verification and quality control agent ('Il Vigile').
    Operates as the Critic in the Actor-Critic pattern, auditing extracted manuscript data
    against physical sanity checks, inclusion criteria, and historical SQLite distributions.
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()
        self.alias_resolver = DatasetAliasResolver(db_manager=self.db)

    def audit(
        self,
        extracted: ExtractedPaperData,
        manuscript: Optional[ParsedManuscript] = None,
        benchmark_mode: bool = False,
    ) -> AuditReport:
        """
        Performs comprehensive multi-dimensional audit of extracted paper data.

        Args:
            extracted: Raw extracted structured data from ExtractionAgent.
            manuscript: Optional ParsedManuscript to diagnose missing tables/figures.
            benchmark_mode: If True, compares extracted values against human survey
                ground truth in SQLite to compute evaluation discrepancies.
                If False (autonomous deploy mode for new literature / zero-shot DB construction),
                relies purely on intrinsic physical, architectural, self-consistency,
                and historical dataset distributions.

        Returns:
            AuditReport certifying or flagging the extraction.
        """
        anomalies: List[AuditAnomaly] = []
        outliers: List[OutlierDetectionResult] = []
        discrepancies: List[DiscrepancyNotice] = []
        rejection_reasons: List[str] = []
        data_diagnosis = MissingDataDiagnosis.NONE

        passed_sanity = True
        passed_screening = True

        # -------------------------------------------------------------
        # 1. Inclusion Criteria Screening Check
        # -------------------------------------------------------------
        if not extracted.screening.is_included:
            # Domain Knowledge Override / Critic Correction:
            # Check if the paper evaluates on known 3D volumetric benchmarks
            # (e.g. ACDC, BraTS, BTCV, AMOS, KiTS, LiTS, MSD, TotalSegmentator).
            # LLMs sometimes flag is_3d=False when a model performs 2D slice inference
            # on 3D volumetric scans (like KnowSAM or nnU-Net 2D on ACDC).
            eval_datasets = [s.dataset_name for s in extracted.extracted_scores]
            canon_datasets = [self.alias_resolver.resolve(d) or d for d in eval_datasets]
            known_3d_datasets = {"acdc", "brats", "btcv", "amos", "kits", "lits", "flare", "totalsegmentator", "synapse", "isles", "segthor"}
            evaluates_known_3d = any(c.lower() in known_3d_datasets for c in canon_datasets)

            if not extracted.screening.is_3d and evaluates_known_3d and extracted.screening.is_radiological and extracted.screening.is_public_dataset and extracted.screening.has_dice_metric:
                passed_screening = True
            else:
                passed_screening = False
                rejection_reasons.extend(extracted.screening.rejection_reasons)

        # -------------------------------------------------------------
        # 2. Numerical & Architectural Sanity Checks
        # -------------------------------------------------------------
        # 2a. Parameter count sanity & architectural scaling
        if extracted.millions_of_parameters is not None:
            params = extracted.millions_of_parameters
            if params <= 0.0:
                passed_sanity = False
                anomalies.append(
                    AuditAnomaly(
                        field="millions_of_parameters",
                        severity="critical",
                        message=f"Parameter count must be strictly positive, got {params}M",
                        extracted_value=params,
                    )
                )
            elif params > 50000.0:
                passed_sanity = False
                anomalies.append(
                    AuditAnomaly(
                        field="millions_of_parameters",
                        severity="critical",
                        message=f"Unrealistic parameter count > 50,000M ({params}M)",
                        extracted_value=params,
                    )
                )
            else:
                # Heuristic architectural family checks
                backbone_text = f"{extracted.visual_backbone or ''} {extracted.architecture or ''}".lower()
                if "vit-b" in backbone_text or "vit base" in backbone_text or "vit-base" in backbone_text:
                    if params < 30.0 or params > 350.0:
                        anomalies.append(
                            AuditAnomaly(
                                field="millions_of_parameters",
                                severity="warning",
                                message=f"ViT-B backbone reported with unusual parameter count {params}M (typical ~85-150M)",
                                extracted_value=params,
                                expected_range=(60.0, 220.0),
                            )
                        )
                elif "vit-l" in backbone_text or "vit large" in backbone_text or "vit-large" in backbone_text:
                    if params < 150.0 or params > 700.0:
                        anomalies.append(
                            AuditAnomaly(
                                field="millions_of_parameters",
                                severity="warning",
                                message=f"ViT-L backbone reported with unusual parameter count {params}M (typical ~300-450M)",
                                extracted_value=params,
                                expected_range=(200.0, 550.0),
                            )
                        )
                elif "vit-h" in backbone_text or "vit huge" in backbone_text or "vit-huge" in backbone_text:
                    if params < 350.0 or params > 1500.0:
                        anomalies.append(
                            AuditAnomaly(
                                field="millions_of_parameters",
                                severity="warning",
                                message=f"ViT-H backbone reported with unusual parameter count {params}M (typical ~600-800M)",
                                extracted_value=params,
                                expected_range=(450.0, 1000.0),
                            )
                        )

        # 2b. VRAM sanity
        if extracted.resources_total_vram_gb is not None:
            vram = extracted.resources_total_vram_gb
            if vram <= 0 or vram > 20000:
                anomalies.append(
                    AuditAnomaly(
                        field="resources_total_vram_gb",
                        severity="warning",
                        message=f"Anomalous total VRAM count ({vram} GB)",
                        extracted_value=vram,
                    )
                )

        # 2c. Dice score physical bounds
        for s in extracted.extracted_scores:
            if s.dice_score is None:
                continue
            if s.dice_score < 0.0 or s.dice_score > 1.0:
                passed_sanity = False
                anomalies.append(
                    AuditAnomaly(
                        field="dice_score",
                        severity="critical",
                        message=f"Dice score {s.dice_score} out of physical range [0.0, 1.0] on {s.dataset_name}",
                        extracted_value=s.dice_score,
                        expected_range=(0.0, 1.0),
                    )
                )
            elif s.dice_score >= 0.999:
                anomalies.append(
                    AuditAnomaly(
                        field="dice_score",
                        severity="warning",
                        message=f"Dice score {s.dice_score} on {s.dataset_name} is near-perfect (1.000), which is statistically improbable in 3D radiological segmentation",
                        extracted_value=s.dice_score,
                    )
                )

        # 2d. Baseline Competitor vs Proposed Model Confusion Check
        model_id_clean = extracted.proposed_model_id.lower().strip()
        title_clean = extracted.paper_title.lower().strip()
        if model_id_clean in KNOWN_BASELINE_MODELS and model_id_clean not in title_clean:
            passed_sanity = False
            anomalies.append(
                AuditAnomaly(
                    field="proposed_model_id",
                    severity="critical",
                    message=(
                        f"Extractor identified standard baseline '{extracted.proposed_model_id}' "
                        f"as the proposed model, but the paper title ('{extracted.paper_title}') introduces a distinct architecture."
                    ),
                    extracted_value=extracted.proposed_model_id,
                )
            )

        # 2e. Unflagged Baseline Row Detection
        for s in extracted.extracted_scores:
            if not s.is_baseline_comparison and s.evaluated_model_name:
                eval_name_clean = s.evaluated_model_name.lower().strip()
                if eval_name_clean in KNOWN_BASELINE_MODELS and eval_name_clean != model_id_clean:
                    anomalies.append(
                        AuditAnomaly(
                            field="extracted_scores",
                            severity="warning",
                            message=(
                                f"Score on {s.dataset_name} ({s.dice_score if s.dice_score is not None else 'N/A'}) for model '{s.evaluated_model_name}' "
                                f"was not flagged as baseline comparison despite being a known competitor."
                            ),
                            extracted_value=s.evaluated_model_name,
                        )
                    )

        # 2f. Internal Contradiction / Conflicting Duplicate Scores Check
        # Only flag conflicting duplicate scores if reported under identical table/section,
        # prompt setting, AND without differing data budgets or experimental conditions (e.g. semi-supervised ratios).
        proposed_scores_by_condition: Dict[Tuple[str, str, str, str, str], List[float]] = {}
        for s in extracted.extracted_scores:
            if not s.is_baseline_comparison and s.dice_score is not None:
                canon_d = self.alias_resolver.resolve(s.dataset_name) or s.dataset_name.strip()
                organ = (s.organ_or_task or "").strip().lower()
                prompt = (s.prompt_type or "default").strip().lower()
                table_ref = (s.table_or_section_ref or "").strip().lower()
                budget = (s.data_budget_or_setting or "").strip().lower()

                # Skip semi-supervised or ablation partitions if organ text already specifies the budget
                if any(k in organ for k in ("scarce", "label", "%", "percent", "few", "semi", "ablation", "round", "client")):
                    continue

                key = (canon_d.lower(), organ, prompt, table_ref, budget)
                proposed_scores_by_condition.setdefault(key, []).append(s.dice_score)

        for (d_k, organ_k, prompt_k, t_k, b_k), vals in proposed_scores_by_condition.items():
            unique_vals = sorted(list(set(vals)))
            # Only flag as conflicting if exactly 2 contradictory scores differ significantly (> 0.03 DSC) under identical conditions.
            # If 3 or more values exist, it represents an ablation series, training checkpoints, or data budget sweep.
            if len(unique_vals) == 2 and (max(unique_vals) - min(unique_vals)) > 0.03:
                anomalies.append(
                    AuditAnomaly(
                        field="extracted_scores",
                        severity="warning",
                        message=(
                            f"Conflicting duplicate Dice scores extracted for {d_k.upper()} (organ='{organ_k}', prompt='{prompt_k}'): "
                            f"{unique_vals}. Model produced inconsistent results under identical test conditions."
                        ),
                        extracted_value=unique_vals,
                    )
                )

        # 2g. Suspiciously Identical / Cloned Scores (Template Hallucination Check)
        has_cloned_template_hallucination = False
        scores_by_val: Dict[float, Set[Tuple[str, str]]] = {}
        for s in extracted.extracted_scores:
            if not s.is_baseline_comparison and s.dice_score is not None:
                val = round(s.dice_score, 4)
                d_canon = (self.alias_resolver.resolve(s.dataset_name) or s.dataset_name).strip().lower()
                task = (s.organ_or_task or "").strip().lower()
                scores_by_val.setdefault(val, set()).add((d_canon, task))

        for score_v, tasks_set in scores_by_val.items():
            if len(tasks_set) >= 4:
                passed_sanity = False
                has_cloned_template_hallucination = True
                anomalies.append(
                    AuditAnomaly(
                        field="extracted_scores",
                        severity="critical",
                        message=(
                            f"Suspiciously identical Dice score ({score_v:.4f}) repeated across {len(tasks_set)} "
                            f"different benchmarks/tasks. Strong indicator of LLM template hallucination."
                        ),
                        extracted_value=score_v,
                    )
                )

        # 2h. Hardware vs VRAM Consistency Check
        if extracted.resources and extracted.resources_total_vram_gb is not None:
            res_lower = extracted.resources.lower()
            vram = extracted.resources_total_vram_gb
            if re.search(r"\b1x?\s*(?:gpu|nvidia|rtx|tesla|a100|v100|3090|4090)\b", res_lower) and vram > 120:
                anomalies.append(
                    AuditAnomaly(
                        field="resources_total_vram_gb",
                        severity="warning",
                        message=f"Paper reports single GPU ('{extracted.resources}'), but extracted total VRAM is {vram} GB (typical single GPU max is 80-96 GB)",
                        extracted_value=vram,
                    )
                )

        # -------------------------------------------------------------
        # 3. Statistical Baseline Outlier Detection (SQLite Knowledge Base)
        # -------------------------------------------------------------
        for s in extracted.extracted_scores:
            if s.is_baseline_comparison or s.dice_score is None:
                continue

            canon_dataset = self.alias_resolver.resolve(s.dataset_name)
            if not canon_dataset:
                continue

            stats = self.db.get_dataset_statistics(canon_dataset, category=ScoreCategory.BEST)
            if not stats:
                continue

            score_val = s.dice_score
            # Check difficult anatomical targets (e.g. Pancreas tumor SOTA ~0.80)
            if ("pancreas" in canon_dataset.lower() or (s.organ_or_task and "pancreas" in s.organ_or_task.lower())) and score_val > 0.92:
                outliers.append(
                    OutlierDetectionResult(
                        dataset_name=canon_dataset,
                        extracted_score=score_val,
                        historical_min=stats["min"],
                        historical_max=stats["max"],
                        historical_mean=stats["mean"],
                        historical_count=stats["count"],
                        is_outlier=True,
                        reason=f"Pancreas score {score_val:.4f} is biologically improbable (current world benchmark max is {stats['max']:.4f})",
                    )
                )
            # Check if exceeding historical SOTA by > 4%
            elif score_val > stats["max"] + 0.04:
                outliers.append(
                    OutlierDetectionResult(
                        dataset_name=canon_dataset,
                        extracted_score=score_val,
                        historical_min=stats["min"],
                        historical_max=stats["max"],
                        historical_mean=stats["mean"],
                        historical_count=stats["count"],
                        is_outlier=True,
                        reason=f"Score {score_val:.4f} exceeds historical SOTA ({stats['max']:.4f}) by >4% DSC",
                    )
                )
            # Check abnormally low score
            elif score_val < 0.20 and stats["min"] > 0.40:
                outliers.append(
                    OutlierDetectionResult(
                        dataset_name=canon_dataset,
                        extracted_score=score_val,
                        historical_min=stats["min"],
                        historical_max=stats["max"],
                        historical_mean=stats["mean"],
                        historical_count=stats["count"],
                        is_outlier=True,
                        reason=f"Score {score_val:.4f} is abnormally low compared to historical distribution ({stats['min']:.4f} - {stats['max']:.4f})",
                    )
                )

        # -------------------------------------------------------------
        # 4. Human Survey Discrepancy Detection (Ground Truth Cross-Check)
        # Only active during offline evaluation benchmark to compare vs Moglia et al.
        # In autonomous deploy mode (new literature / DB construction), benchmark_mode is False.
        # -------------------------------------------------------------
        if benchmark_mode:
            gt_scores = self.db.get_scores_for_model(extracted.proposed_model_id)
            if gt_scores:
                # Group extracted scores by canonical dataset
                extracted_by_dataset: Dict[str, List[RawExtractedScore]] = {}
                for s in extracted.extracted_scores:
                    if not s.is_baseline_comparison and s.dice_score is not None:
                        c = self.alias_resolver.resolve(s.dataset_name) or s.dataset_name
                        extracted_by_dataset.setdefault(c, []).append(s)

                for d_name, d_scores in gt_scores.items():
                    canon_d = self.alias_resolver.resolve(d_name) or d_name
                    true_val = d_scores.get("primary") or d_scores.get("best")
                    if true_val is not None:
                        true_dice = round(true_val / 100.0 if true_val > 1.0 else true_val, 4)

                        # Match canon_d or handle Synapse/BTCV equivalence if the other is not in gt_scores
                        match_key = None
                        if canon_d in extracted_by_dataset:
                            match_key = canon_d
                        elif canon_d.lower() == "synapse" and "BTCV" in extracted_by_dataset and "BTCV" not in gt_scores:
                            match_key = "BTCV"
                        elif canon_d.lower() == "btcv" and "Synapse" in extracted_by_dataset and "Synapse" not in gt_scores:
                            match_key = "Synapse"

                        if match_key:
                            cand_scores = [s.dice_score for s in extracted_by_dataset[match_key] if s.dice_score is not None]
                            if cand_scores:
                                # Find closest score
                                closest = min(cand_scores, key=lambda val: abs(val - true_dice))
                                delta = round(closest - true_dice, 4)
                                if abs(delta) > 0.0005:
                                    discrepancies.append(
                                        DiscrepancyNotice(
                                            dataset_name=match_key,
                                            extracted_score=closest,
                                            survey_ground_truth=true_dice,
                                            delta=delta,
                                            message=(
                                                f"Discrepancy on {match_key}: paper reports {closest:.4f}, "
                                                f"human survey recorded {true_dice:.4f} (diff = {delta:+.4f})"
                                            ),
                                        )
                                    )

        # -------------------------------------------------------------
        # 5. Missing Data Diagnosis
        # -------------------------------------------------------------
        scalar_scores = [s for s in extracted.extracted_scores if s.dice_score is not None and not s.is_baseline_comparison]

        # Check if extracted scores are only textual ablations rather than formal benchmark tables
        # (e.g. BiomedParse, where the 14 main benchmarks are only in figures/radar plots and GitHub)
        has_canonical_benchmark_table = False
        if scalar_scores and not has_cloned_template_hallucination:
            for s in scalar_scores:
                canon = self.alias_resolver.resolve(s.dataset_name)
                # If it's a known canonical benchmark and comes from a table or repository artifact
                if canon and s.table_or_section_ref and any(t in s.table_or_section_ref.lower() for t in ("table", "readme", ".csv", ".json", "repo")):
                    has_canonical_benchmark_table = True
                    break

        if len(scalar_scores) == 0 or has_cloned_template_hallucination or (not has_canonical_benchmark_table and extracted.code_repository_url):
            figures_found = False
            if manuscript is not None:
                for text in manuscript.pages_text.values():
                    t_low = text.lower()
                    has_figs = bool(re.search(r"\b(?:figure|fig\.?)\s+(\d+|[ivx]+)\b", t_low))
                    has_chart = any(term in t_low for term in ("bar chart", "radar plot", "box plot", "performance curve"))
                    has_metrics = any(m in t_low for m in ("dice", "dsc", "mean dice"))
                    if (has_figs or has_chart) and has_metrics:
                        figures_found = True
                        break

            if (len(scalar_scores) == 0 or has_cloned_template_hallucination) and figures_found and not extracted.code_repository_url:
                data_diagnosis = MissingDataDiagnosis.MISSING_TABLE_FIGURES_FOUND
            elif extracted.code_repository_url:
                data_diagnosis = MissingDataDiagnosis.CODE_REPO_AVAILABLE
            elif len(scalar_scores) == 0 or has_cloned_template_hallucination:
                data_diagnosis = MissingDataDiagnosis.MISSING_TABLE_NO_DATA

        # -------------------------------------------------------------
        # 6. Status Certification & Confidence Calculation
        # -------------------------------------------------------------
        if not passed_screening:
            status = AuditStatus.REJECTED
        elif data_diagnosis == MissingDataDiagnosis.CODE_REPO_AVAILABLE:
            status = AuditStatus.NEEDS_REPO_INSPECTION
        elif data_diagnosis == MissingDataDiagnosis.MISSING_TABLE_FIGURES_FOUND:
            status = AuditStatus.NEEDS_OUTREACH
        elif data_diagnosis == MissingDataDiagnosis.MISSING_TABLE_NO_DATA:
            status = AuditStatus.REJECTED
            rejection_reasons.append("Manuscript contains no quantitative evaluation tables or benchmark scores.")
        elif any(a.severity == "critical" for a in anomalies):
            status = AuditStatus.REJECTED
            rejection_reasons.extend(a.message for a in anomalies if a.severity == "critical")
        elif len(anomalies) > 0 or len(outliers) > 0 or len(discrepancies) > 0:
            status = AuditStatus.FLAGGED_WARNING
        else:
            status = AuditStatus.VERIFIED

        # Compute confidence score
        confidence = 1.0
        confidence -= len([a for a in anomalies if a.severity == "critical"]) * 0.5
        # Cap warning penalty to max 0.35 so warnings alone cannot drag a non-rejected model to 0%
        warning_penalty = min(0.35, len([a for a in anomalies if a.severity == "warning"]) * 0.05)
        confidence -= warning_penalty
        confidence -= min(0.20, len(outliers) * 0.05)
        confidence -= min(0.15, len(discrepancies) * 0.05)
        if not passed_screening or any(a.severity == "critical" for a in anomalies):
            confidence = min(confidence, 0.40)
            if not passed_screening:
                confidence = 0.0
        confidence = max(0.0, min(1.0, round(confidence, 2)))

        # Build human-readable summary
        summary_lines = [
            f"=== Audit Certificate: {extracted.proposed_model_id} ===",
            f"Status: {status.value.upper()}",
            f"Screening Passed: {passed_screening}",
            f"Sanity Checks Passed: {passed_sanity}",
            f"Confidence Score: {confidence:.2f}",
        ]
        if rejection_reasons:
            summary_lines.append(f"Rejection Reasons: {'; '.join(rejection_reasons)}")
        if anomalies:
            summary_lines.append(f"Anomalies ({len(anomalies)}): " + "; ".join(a.message for a in anomalies))
        if outliers:
            summary_lines.append(f"Statistical Outliers ({len(outliers)}): " + "; ".join(o.reason or '' for o in outliers))
        if discrepancies:
            summary_lines.append(f"Survey Discrepancies ({len(discrepancies)}): " + "; ".join(d.message for d in discrepancies))
        if data_diagnosis != MissingDataDiagnosis.NONE:
            summary_lines.append(f"Data Diagnosis: {data_diagnosis.value}")

        return AuditReport(
            model_id=extracted.proposed_model_id,
            status=status,
            passed_sanity_checks=passed_sanity,
            passed_inclusion_screening=passed_screening,
            rejection_reasons=rejection_reasons,
            anomalies=anomalies,
            outliers=outliers,
            discrepancies=discrepancies,
            data_diagnosis=data_diagnosis,
            confidence_score=confidence,
            summary="\n".join(summary_lines),
        )
