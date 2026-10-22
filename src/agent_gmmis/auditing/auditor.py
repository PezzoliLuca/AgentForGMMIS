"""AuditorAgent ('Il Vigile'): Autonomous critic and quality assurance engine."""

import re
from typing import Dict, List, Optional, Set, Tuple
from agent_gmmis.auditing.schemas import (
    AuditAnomaly,
    AuditReport,
    AuditStatus,
    ConsensusAuditReport,
    ConsensusMetadata,
    ConsensusScore,
    ConsensusScreening,
    ConsensusStatus,
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

    STOPWORDS = {"and", "for", "with", "of", "in", "the", "a", "an", "on", "to", "from", "by", "as"}

    @classmethod
    def _is_model_name_compatible_with_paper(
        cls,
        model_id: str,
        paper_title: str,
        manuscript: Optional[ParsedManuscript] = None,
    ) -> bool:
        """Determines if a model name is genuinely the paper's proposed architecture rather than a baseline competitor.

        Evaluates multi-level semantic compatibility:
        1. Exact or punctuation-insensitive substring match in title.
        2. Acronym / abbreviation / prefix alignment with title words (excluding common stopwords).
        3. Explicit self-nomination in manuscript abstract/introduction ("we propose/introduce X", "our model, X").
        """
        if not model_id or not paper_title:
            return True

        m_clean = model_id.lower().strip()
        t_clean = paper_title.lower().strip()

        # 1. Direct or normalized substring match in title
        if m_clean in t_clean:
            return True
        m_alphanumeric = re.sub(r"[^a-z0-9]", "", m_clean)
        t_alphanumeric = re.sub(r"[^a-z0-9]", "", t_clean)
        if m_alphanumeric and m_alphanumeric in t_alphanumeric:
            return True

        # 2. Acronym / Abbreviation alignment with title words
        title_tokens = [w for w in re.findall(r"[a-z0-9]+", t_clean) if len(w) > 1 and w not in cls.STOPWORDS]
        model_tokens = [w for w in re.findall(r"[a-z0-9]+", m_clean) if w]

        if model_tokens and title_tokens:
            tokens_matched = 0
            for mt in model_tokens:
                # Check if mt is prefix of any title word (e.g. 'med' of 'medical')
                if any(tt.startswith(mt) for tt in title_tokens):
                    tokens_matched += 1
                # Check if mt matches initials of consecutive title words (e.g. 'sa' for 'sam adapter')
                elif any(
                    "".join(title_tokens[i : i + len(mt)]).startswith(mt)
                    or "".join(w[0] for w in title_tokens[i : i + len(mt)]) == mt
                    for i in range(len(title_tokens) - len(mt) + 1)
                ):
                    tokens_matched += 1

            if tokens_matched == len(model_tokens):
                return True

            # Check title word initials (e.g. "Medical SAM Adapter" initials: "msa", "STU-Net" in "stun")
            title_initials = "".join(w[0] for w in title_tokens)
            if m_alphanumeric in title_initials or (len(m_alphanumeric) >= 3 and title_initials.startswith(m_alphanumeric)):
                return True

        # 3. Contextual Self-Nomination in Manuscript (Abstract & First 3 Pages)
        if manuscript is not None and getattr(manuscript, "pages_text", None):
            first_text = " ".join(
                str(manuscript.pages_text.get(p, ""))
                for p in range(1, min(4, getattr(manuscript, "total_pages", 1) + 1))
            ).lower()

            # Pattern A: Title followed by acronym in parentheses, e.g. "Medical SAM Adapter (Med-SA)"
            if re.search(rf"\(\s*{re.escape(m_clean)}\s*\)", first_text):
                return True

            # Pattern B: Authors explicitly state "we propose/introduce/present/develop X" or "our model/framework/method X"
            self_nomination_pattern = rf"(?:we\s+(?:propose|introduce|present|develop)|our\s+(?:model|method|framework|approach|architecture))\s+(?:named|called\s+)?[\w\s-]{{0,30}}\b{re.escape(m_clean)}\b"
            if re.search(self_nomination_pattern, first_text):
                return True

        return False

    @staticmethod
    def normalize_data_budget(budget: Optional[str]) -> Optional[str]:
        """
        Normalizes data budget strings into canonical semantics (e.g. percentages or shots).
        E.g. '100% labeled', '100% training data (24 samples)', 'fully supervised' -> '100%'
             '25% labeled', '25% training data (6 samples)' -> '25%'
             '1-shot', '1 shot' -> '1-shot'
        """
        if not budget:
            return None
        b_lower = budget.strip().lower()
        # 1. Percentage check (e.g. 100%, 30%, 25%, 12%, 10%, 1%)
        pct_match = re.search(r"\b(\d{1,3})\s*%", b_lower)
        if pct_match:
            return f"{pct_match.group(1)}%"
        # 2. Shot check (e.g. 1-shot, 5-shot)
        shot_match = re.search(r"\b(\d{1,2})\s*-?\s*shots?\b", b_lower)
        if shot_match:
            return f"{shot_match.group(1)}-shot"
        # 3. Supervised synonyms
        if any(w in b_lower for w in ["fully supervised", "full data", "all data", "full training", "100 percent"]):
            return "100%"
        return b_lower

    def audit(
        self,
        extracted: ExtractedPaperData,
        manuscript: Optional[ParsedManuscript] = None,
        benchmark_mode: bool = False,
        repo_inspected: bool = False,
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
            repo_inspected: If True, indicates that the code repository was already inspected
                and yielded no quantitative evaluation tables, allowing transition to outreach.

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
                notes_text = " ".join([s.notes or "" for s in extracted.extracted_scores[:10]]).lower()
                backbone_text = f"{extracted.visual_backbone or ''} {extracted.architecture or ''} {extracted.proposed_model_id} {extracted.paper_title} {notes_text}".lower()
                PEFT_KEYWORDS = {"adapter", "lora", "peft", "prompt", "tuning", "frozen", "trainable", "efficient", "lean", "hybrid", "distill", "lightweight"}
                is_peft = any(k in backbone_text for k in PEFT_KEYWORDS)

                if "vit-b" in backbone_text or "vit base" in backbone_text or "vit-base" in backbone_text:
                    if (params < 20.0 and not is_peft) or (params > 350.0 and not any(k in backbone_text for k in ["multimodal", "text", "focal", "language"])):
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
                    if (params < 150.0 and not is_peft) or params > 700.0:
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
                    if (params < 350.0 and not is_peft) or params > 1500.0:
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
        if model_id_clean in KNOWN_BASELINE_MODELS and not self._is_model_name_compatible_with_paper(
            extracted.proposed_model_id, extracted.paper_title, manuscript
        ):
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
                resolved_d = self.alias_resolver.resolve(s.dataset_name) if s.dataset_name else None
                canon_d = (resolved_d or s.dataset_name or "Unknown").strip()
                organ = (s.organ_or_task or "").strip().lower()
                prompt = (s.prompt_type or "default").strip().lower()
                table_ref = (s.table_or_section_ref or "").strip().lower()
                budget = (s.data_budget_or_setting or "").strip().lower()

                # Skip semi-supervised or ablation partitions if organ text or budget already specifies the partition
                organ_and_setting = f"{organ} {prompt} {budget}"
                if any(k in organ_and_setting for k in ("scarce", "label", "%", "percent", "few", "semi", "ablation", "round", "client", "shot", "split")):
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
        # Template hallucinations only apply to scores extracted directly from manuscript text/tables,
        # not to authentic empirical evaluation artifacts recovered from code repositories.
        has_cloned_template_hallucination = False
        manuscript_scores = [
            s for s in extracted.extracted_scores
            if not s.is_baseline_comparison
            and s.dice_score is not None
            and not (s.table_or_section_ref and s.table_or_section_ref.startswith("repo:"))
        ]
        scores_by_val: Dict[float, Set[Tuple[str, str]]] = {}
        for s in manuscript_scores:
            val = round(s.dice_score, 4)
            resolved_d = self.alias_resolver.resolve(s.dataset_name) if s.dataset_name else None
            d_canon = (resolved_d or s.dataset_name or "Unknown").strip().lower()
            task = (s.organ_or_task or "").strip().lower()
            scores_by_val.setdefault(val, set()).add((d_canon, task))

        for score_v, tasks_set in scores_by_val.items():
            distinct_datasets = {d for d, _ in tasks_set if d and d != "unknown"}
            # A true template hallucination occurs when the identical score is copy-pasted across
            # multiple distinct benchmark datasets (e.g. BTCV and ACDC and ISLES all sharing 0.8500).
            # If multiple tasks/organs belong to the same dataset (e.g. 4 SegTHOR organs), it is legitimate.
            if len(distinct_datasets) >= 3 or len(tasks_set) >= 8:
                passed_sanity = False
                has_cloned_template_hallucination = True
                anomalies.append(
                    AuditAnomaly(
                        field="extracted_scores",
                        severity="critical",
                        message=(
                            f"Suspiciously identical Dice score ({score_v:.4f}) repeated across {len(distinct_datasets)} "
                            f"distinct datasets ({', '.join(sorted(distinct_datasets)[:4])}). Strong indicator of LLM template hallucination."
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

            if (len(scalar_scores) == 0 or has_cloned_template_hallucination):
                if extracted.code_repository_url and not repo_inspected:
                    data_diagnosis = MissingDataDiagnosis.CODE_REPO_AVAILABLE
                elif figures_found:
                    data_diagnosis = MissingDataDiagnosis.MISSING_TABLE_FIGURES_FOUND
                else:
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

    def audit_consensus(
        self,
        extracted_a: ExtractedPaperData,
        extracted_b: ExtractedPaperData,
        manuscript: Optional[ParsedManuscript] = None,
        benchmark_mode: bool = False,
        tolerance: float = 0.01,
    ) -> ConsensusAuditReport:
        """
        Performs dual-agent consensus auditing and epistemic arbitration across two independent extractions.

        Args:
            extracted_a: ExtractedPaperData from Agent A.
            extracted_b: ExtractedPaperData from Agent B.
            manuscript: Optional ParsedManuscript for table/figure diagnosis.
            benchmark_mode: If True, cross-checks against human ground truth in SQLite.
            tolerance: Maximum absolute Dice discrepancy considered CONCORDANT (default: 0.01 = 1% DSC).

        Returns:
            ConsensusAuditReport with inter-annotator agreement metrics and certified status.
        """
        # 1. Audit both extractions individually
        report_a = self.audit(extracted_a, manuscript=manuscript, benchmark_mode=benchmark_mode)
        report_b = self.audit(extracted_b, manuscript=manuscript, benchmark_mode=benchmark_mode)

        # 2. Reconcile Inclusion Screening
        sc_a = extracted_a.screening
        sc_b = extracted_b.screening

        screening_disagreements: List[str] = []
        if sc_a.is_3d != sc_b.is_3d:
            screening_disagreements.append(f"is_3d (A={sc_a.is_3d}, B={sc_b.is_3d})")
        if sc_a.is_radiological != sc_b.is_radiological:
            screening_disagreements.append(f"is_radiological (A={sc_a.is_radiological}, B={sc_b.is_radiological})")
        if sc_a.is_public_dataset != sc_b.is_public_dataset:
            screening_disagreements.append(f"is_public_dataset (A={sc_a.is_public_dataset}, B={sc_b.is_public_dataset})")
        if sc_a.has_dice_metric != sc_b.has_dice_metric:
            screening_disagreements.append(f"has_dice_metric (A={sc_a.has_dice_metric}, B={sc_b.has_dice_metric})")

        if not screening_disagreements:
            screening_status = ConsensusStatus.CONCORDANT
            is_included = report_a.passed_inclusion_screening and report_b.passed_inclusion_screening
            screening_notes = "Complete consensus on all inclusion screening criteria."
        else:
            screening_status = ConsensusStatus.DISPUTED
            is_included = report_a.passed_inclusion_screening or report_b.passed_inclusion_screening
            screening_notes = f"Dispute on screening criteria: {', '.join(screening_disagreements)}."

        consensus_screening = ConsensusScreening(
            status=screening_status,
            is_included=is_included,
            is_3d_a=sc_a.is_3d,
            is_3d_b=sc_b.is_3d,
            is_radiological_a=sc_a.is_radiological,
            is_radiological_b=sc_b.is_radiological,
            is_public_a=sc_a.is_public_dataset,
            is_public_b=sc_b.is_public_dataset,
            has_dice_a=sc_a.has_dice_metric,
            has_dice_b=sc_b.has_dice_metric,
            notes=screening_notes,
        )

        # 3. Reconcile Model Metadata & Parameters
        id_a = (extracted_a.proposed_model_id or "").strip()
        id_b = (extracted_b.proposed_model_id or "").strip()

        norm_a = re.sub(r"[^a-z0-9]", "", id_a.lower())
        norm_b = re.sub(r"[^a-z0-9]", "", id_b.lower())

        if norm_a == norm_b:
            model_id_status = ConsensusStatus.CONCORDANT
            resolved_model_id = id_a
        elif norm_a in norm_b or norm_b in norm_a:
            model_id_status = ConsensusStatus.CONCORDANT
            resolved_model_id = id_a if len(id_a) >= len(id_b) else id_b
        else:
            model_id_status = ConsensusStatus.DISPUTED
            if self._is_model_name_compatible_with_paper(id_a, extracted_a.paper_title, manuscript):
                resolved_model_id = id_a
            elif self._is_model_name_compatible_with_paper(id_b, extracted_b.paper_title, manuscript):
                resolved_model_id = id_b
            else:
                resolved_model_id = f"{id_a} / {id_b}"

        params_a = extracted_a.millions_of_parameters
        params_b = extracted_b.millions_of_parameters
        if params_a is not None and params_b is not None:
            max_p = max(params_a, params_b)
            p_diff = abs(params_a - params_b) / max_p if max_p > 0 else 0.0
            if p_diff <= 0.05:
                params_status = ConsensusStatus.CONCORDANT
                resolved_params_m = round((params_a + params_b) / 2.0, 1)
            else:
                params_status = ConsensusStatus.DISPUTED
                resolved_params_m = None
        elif params_a is not None or params_b is not None:
            params_status = ConsensusStatus.SINGLE_AGENT_UNCONFIRMED
            resolved_params_m = params_a if params_a is not None else params_b
        else:
            params_status = ConsensusStatus.CONCORDANT
            resolved_params_m = None

        resolved_backbone = extracted_a.visual_backbone or extracted_b.visual_backbone
        resolved_repo_url = extracted_a.code_repository_url or extracted_b.code_repository_url

        metadata_notes = []
        if model_id_status == ConsensusStatus.DISPUTED:
            metadata_notes.append(f"Model ID dispute: '{id_a}' vs '{id_b}'")
        if params_status == ConsensusStatus.DISPUTED:
            metadata_notes.append(f"Parameter count dispute: {params_a}M vs {params_b}M")

        consensus_metadata = ConsensusMetadata(
            model_id_status=model_id_status,
            resolved_model_id=resolved_model_id,
            model_id_a=id_a,
            model_id_b=id_b,
            params_status=params_status,
            resolved_params_m=resolved_params_m,
            params_m_a=params_a,
            params_m_b=params_b,
            visual_backbone_a=extracted_a.visual_backbone,
            visual_backbone_b=extracted_b.visual_backbone,
            resolved_visual_backbone=resolved_backbone,
            code_repository_url=resolved_repo_url,
            notes="; ".join(metadata_notes) if metadata_notes else None,
        )

        # 4. Reconcile Benchmark Scores (Semantic Alignment via DatasetAliasResolver)
        scores_a_by_canon: Dict[str, List[RawExtractedScore]] = {}
        for s in extracted_a.extracted_scores:
            if not s.is_baseline_comparison:
                resolved = self.alias_resolver.resolve(s.dataset_name) if s.dataset_name else None
                canon = (resolved or s.dataset_name or "Unknown").strip()
                scores_a_by_canon.setdefault(canon.lower(), []).append(s)

        scores_b_by_canon: Dict[str, List[RawExtractedScore]] = {}
        for s in extracted_b.extracted_scores:
            if not s.is_baseline_comparison:
                resolved = self.alias_resolver.resolve(s.dataset_name) if s.dataset_name else None
                canon = (resolved or s.dataset_name or "Unknown").strip()
                scores_b_by_canon.setdefault(canon.lower(), []).append(s)

        all_canon_keys = sorted(list(set(scores_a_by_canon.keys()) | set(scores_b_by_canon.keys())))

        consensus_scores: List[ConsensusScore] = []
        for canon_k in all_canon_keys:
            list_a = scores_a_by_canon.get(canon_k, [])
            list_b = scores_b_by_canon.get(canon_k, [])

            if list_a and not list_b:
                for s_a in list_a:
                    canon_a = (self.alias_resolver.resolve(s_a.dataset_name) if s_a.dataset_name else None) or s_a.dataset_name or "Unknown"
                    consensus_scores.append(
                        ConsensusScore(
                            dataset_name=s_a.dataset_name,
                            canonical_dataset_name=canon_a,
                            organ_or_task=s_a.organ_or_task,
                            status=ConsensusStatus.SINGLE_AGENT_UNCONFIRMED,
                            score_a=s_a.dice_score,
                            score_b=None,
                            resolved_score=s_a.dice_score,
                            delta=None,
                            table_ref_a=s_a.table_or_section_ref,
                            prompt_type=s_a.prompt_type,
                            data_budget_or_setting=s_a.data_budget_or_setting,
                            notes="Extracted only by Agent A",
                        )
                    )
                continue

            if list_b and not list_a:
                for s_b in list_b:
                    canon_b = (self.alias_resolver.resolve(s_b.dataset_name) if s_b.dataset_name else None) or s_b.dataset_name or "Unknown"
                    consensus_scores.append(
                        ConsensusScore(
                            dataset_name=s_b.dataset_name,
                            canonical_dataset_name=canon_b,
                            organ_or_task=s_b.organ_or_task,
                            status=ConsensusStatus.SINGLE_AGENT_UNCONFIRMED,
                            score_a=None,
                            score_b=s_b.dice_score,
                            resolved_score=s_b.dice_score,
                            delta=None,
                            table_ref_b=s_b.table_or_section_ref,
                            prompt_type=s_b.prompt_type,
                            data_budget_or_setting=s_b.data_budget_or_setting,
                            notes="Extracted only by Agent B",
                        )
                    )
                continue

            # Present in both -> match individual score rows using optimal greedy bipartite matching
            candidate_pairs = []
            for idx_a, s_a in enumerate(list_a):
                organ_a_norm = (s_a.organ_or_task or "").strip().lower()
                prompt_a_norm = (s_a.prompt_type or "").strip().lower()
                budget_a_norm = (s_a.data_budget_or_setting or "").strip().lower()
                sem_budget_a = self.normalize_data_budget(s_a.data_budget_or_setting)

                for idx_b, s_b in enumerate(list_b):
                    organ_b_norm = (s_b.organ_or_task or "").strip().lower()
                    prompt_b_norm = (s_b.prompt_type or "").strip().lower()
                    budget_b_norm = (s_b.data_budget_or_setting or "").strip().lower()
                    sem_budget_b = self.normalize_data_budget(s_b.data_budget_or_setting)

                    # If both explicitly specify contradictory data budget splits (e.g. 10% vs 20%), do not match as same test
                    if sem_budget_a and sem_budget_b and sem_budget_a != sem_budget_b:
                        continue

                    OVERALL_SYNONYMS = {"overall", "mean", "average", "avg", "multi-organ mean", "macro average", "all organs", "multi-organ", "total"}

                    is_overall_a = organ_a_norm in OVERALL_SYNONYMS
                    is_overall_b = organ_b_norm in OVERALL_SYNONYMS

                    diff_score = abs(s_a.dice_score - s_b.dice_score) if (s_a.dice_score is not None and s_b.dice_score is not None) else 1.0

                    # If one is an aggregated macro-average and the other is a specific organ, do not match as the same test
                    # unless both agents extracted only 1 row for this dataset or the scores match identically
                    if is_overall_a != is_overall_b and (organ_a_norm and organ_b_norm):
                        if not (len(list_a) == 1 and len(list_b) == 1) and diff_score > 0.01:
                            continue

                    match_pts = 0
                    if organ_a_norm == organ_b_norm or (not organ_a_norm and not organ_b_norm):
                        match_pts += 3
                    elif is_overall_a and is_overall_b:
                        match_pts += 3
                    elif organ_a_norm in organ_b_norm or organ_b_norm in organ_a_norm:
                        match_pts += 2

                    if prompt_a_norm and prompt_b_norm and prompt_a_norm == prompt_b_norm:
                        match_pts += 2

                    if sem_budget_a and sem_budget_b and sem_budget_a == sem_budget_b:
                        match_pts += 3
                    elif budget_a_norm and budget_b_norm and budget_a_norm == budget_b_norm:
                        match_pts += 3

                    if diff_score <= 0.01:
                        match_pts += 5
                    elif diff_score <= 0.05:
                        match_pts += 2

                    if len(list_b) == 1 and len(list_a) == 1:
                        match_pts += 1

                    if match_pts > 0:
                        candidate_pairs.append((match_pts, -diff_score, idx_a, idx_b))

            # Sort candidate pairs by highest match points, then smallest score difference
            candidate_pairs.sort(key=lambda x: (x[0], x[1]), reverse=True)

            matched_a_indices: Set[int] = set()
            matched_b_indices: Set[int] = set()

            for pts, _, idx_a, idx_b in candidate_pairs:
                if idx_a in matched_a_indices or idx_b in matched_b_indices:
                    continue
                matched_a_indices.add(idx_a)
                matched_b_indices.add(idx_b)

                s_a = list_a[idx_a]
                s_b = list_b[idx_b]
                canon_name = (self.alias_resolver.resolve(s_a.dataset_name) if s_a.dataset_name else None) or s_a.dataset_name or "Unknown"

                val_a = s_a.dice_score
                val_b = s_b.dice_score

                if val_a is not None and val_b is not None:
                    delta = abs(round(val_a - val_b, 4))
                    if delta <= tolerance:
                        status = ConsensusStatus.CONCORDANT
                        resolved_val = round((val_a + val_b) / 2.0, 4)
                        notes = f"Concordant agreement (|A - B| = {delta:.4f} <= {tolerance:.4f})"
                    else:
                        status = ConsensusStatus.DISPUTED
                        resolved_val = None
                        notes = f"DISPUTED: Agent A reported {val_a:.4f}, Agent B reported {val_b:.4f} (diff = {delta:+.4f} > {tolerance:.4f})"
                elif val_a is None and val_b is None:
                    delta = 0.0
                    status = ConsensusStatus.CONCORDANT
                    resolved_val = None
                    notes = "Both models identified qualitative mention without scalar Dice score."
                else:
                    delta = None
                    status = ConsensusStatus.DISPUTED
                    resolved_val = None
                    notes = f"DISPUTED: One agent extracted scalar value ({val_a if val_a is not None else val_b}), other did not."

                consensus_scores.append(
                    ConsensusScore(
                        dataset_name=s_a.dataset_name,
                        canonical_dataset_name=canon_name,
                        organ_or_task=s_a.organ_or_task or s_b.organ_or_task,
                        status=status,
                        score_a=val_a,
                        score_b=val_b,
                        resolved_score=resolved_val,
                        delta=delta,
                        table_ref_a=s_a.table_or_section_ref,
                        table_ref_b=s_b.table_or_section_ref,
                        prompt_type=s_a.prompt_type or s_b.prompt_type,
                        data_budget_or_setting=s_a.data_budget_or_setting or s_b.data_budget_or_setting,
                        notes=notes,
                    )
                )

            # Unmatched rows from Agent A
            for idx_a, s_a in enumerate(list_a):
                if idx_a not in matched_a_indices:
                    canon_name = (self.alias_resolver.resolve(s_a.dataset_name) if s_a.dataset_name else None) or s_a.dataset_name or "Unknown"
                    consensus_scores.append(
                        ConsensusScore(
                            dataset_name=s_a.dataset_name,
                            canonical_dataset_name=canon_name,
                            organ_or_task=s_a.organ_or_task,
                            status=ConsensusStatus.SINGLE_AGENT_UNCONFIRMED,
                            score_a=s_a.dice_score,
                            score_b=None,
                            resolved_score=s_a.dice_score,
                            delta=None,
                            table_ref_a=s_a.table_or_section_ref,
                            prompt_type=s_a.prompt_type,
                            data_budget_or_setting=s_a.data_budget_or_setting,
                            notes="Extracted only by Agent A",
                        )
                    )

            # Unmatched rows from Agent B
            for idx_b, s_b in enumerate(list_b):
                if idx_b not in matched_b_indices:
                    canon_name = (self.alias_resolver.resolve(s_b.dataset_name) if s_b.dataset_name else None) or s_b.dataset_name or "Unknown"
                    consensus_scores.append(
                        ConsensusScore(
                            dataset_name=s_b.dataset_name,
                            canonical_dataset_name=canon_name,
                            organ_or_task=s_b.organ_or_task,
                            status=ConsensusStatus.SINGLE_AGENT_UNCONFIRMED,
                            score_a=None,
                            score_b=s_b.dice_score,
                            resolved_score=s_b.dice_score,
                            delta=None,
                            table_ref_b=s_b.table_or_section_ref,
                            prompt_type=s_b.prompt_type,
                            data_budget_or_setting=s_b.data_budget_or_setting,
                            notes="Extracted only by Agent B",
                        )
                    )

        # 5. Compute Inter-Annotator Agreement (IAA) Metrics
        paired_scores = [cs for cs in consensus_scores if cs.score_a is not None and cs.score_b is not None]
        deltas = [cs.delta for cs in paired_scores if cs.delta is not None]
        inter_model_dice_mae = round(sum(deltas) / len(deltas), 4) if deltas else None

        concordant_count = len([cs for cs in consensus_scores if cs.status == ConsensusStatus.CONCORDANT])
        dispute_count = len([cs for cs in consensus_scores if cs.status == ConsensusStatus.DISPUTED])
        unconfirmed_count = len([cs for cs in consensus_scores if cs.status == ConsensusStatus.SINGLE_AGENT_UNCONFIRMED])
        paired_count = len(paired_scores)
        total_unique_benchmarks = len(consensus_scores)

        # Concordance rate is strictly calculated on common paired extractions (true Inter-Annotator Agreement)
        concordance_rate = (
            round(concordant_count / paired_count, 4)
            if paired_count > 0
            else (1.0 if (not dispute_count and concordant_count > 0) else 0.0)
        )

        # 6. Overall Certification Status & Confidence Score
        # In a dual-agent architecture, a single agent's critical anomaly (e.g. hallucinated training table rows in Agent A)
        # is neutralized if Agent B refutes it, it is rejected during deliberation, and none of those rows enter consensus.
        # Thus, score anomalies are only critical to the consensus if they persist across multiple concordant datasets.
        def is_consensus_critical(a: AuditAnomaly) -> bool:
            if a.severity != "critical":
                return False
            # Parameter anomalies or model identity anomalies are critical to the paper itself
            if a.field != "extracted_scores":
                return True
            # For template hallucination on extracted_scores, check if the identical score
            # appears across multiple distinct datasets in the CONCORDANT consensus scores
            if a.extracted_value is not None:
                try:
                    val = round(float(a.extracted_value), 4)
                    concordant_datasets_with_val = {
                        cs.canonical_dataset_name.lower()
                        for cs in consensus_scores
                        if cs.status == ConsensusStatus.CONCORDANT
                        and cs.resolved_score is not None
                        and round(cs.resolved_score, 4) == val
                    }
                    return len(concordant_datasets_with_val) >= 3
                except (ValueError, TypeError):
                    return True
            return True

        has_critical_anomaly = any(is_consensus_critical(a) for a in (report_a.anomalies + report_b.anomalies))
        has_repo_inspection = (
            (report_a.status == AuditStatus.NEEDS_REPO_INSPECTION or report_b.status == AuditStatus.NEEDS_REPO_INSPECTION)
            and total_unique_benchmarks == 0
        )
        has_outreach = (
            (report_a.status == AuditStatus.NEEDS_OUTREACH or report_b.status == AuditStatus.NEEDS_OUTREACH)
            and total_unique_benchmarks == 0
        )

        # Datasets where both models achieved concordant agreement
        concordant_canon_datasets = {
            cs.canonical_dataset_name.lower()
            for cs in consensus_scores
            if cs.status == ConsensusStatus.CONCORDANT and cs.canonical_dataset_name
        }

        # Filter out outliers that were simply exceeding historical SOTA if verified concordantly by both agents
        genuine_outliers_a = [
            o for o in report_a.outliers
            if not ("exceeds historical SOTA" in (o.reason or "") and o.dataset_name.lower() in concordant_canon_datasets)
        ]
        genuine_outliers_b = [
            o for o in report_b.outliers
            if not ("exceeds historical SOTA" in (o.reason or "") and o.dataset_name.lower() in concordant_canon_datasets)
        ]
        all_genuine_outliers = genuine_outliers_a + genuine_outliers_b

        # Unconfirmed singletons that belong to a dataset that is ALREADY concordant at the macro level
        # are merely anatomical sub-organ breakdowns, not missing benchmark datasets
        genuine_unconfirmed = [
            cs for cs in consensus_scores
            if cs.status == ConsensusStatus.SINGLE_AGENT_UNCONFIRMED
            and cs.canonical_dataset_name.lower() not in concordant_canon_datasets
        ]
        genuine_unconfirmed_count = len(genuine_unconfirmed)

        # Confidence calculation: bounded and proportional
        confidence = 1.0
        if has_critical_anomaly:
            confidence -= 0.50

        # Filter warnings: anomalies on competitor baselines (e.g. SwinUNETR missing baseline flag) do not penalize proposed model confidence
        all_warnings = [
            a for a in (report_a.anomalies + report_b.anomalies)
            if a.severity == "warning"
            and not (a.extracted_value and isinstance(a.extracted_value, str) and a.extracted_value.lower() in KNOWN_BASELINE_MODELS)
        ]
        confidence -= min(0.35, len(all_warnings) * 0.05)
        confidence -= min(0.20, len(all_genuine_outliers) * 0.05)

        # Boost for concordance: multi-model agreement gives strong confidence boost
        if concordant_count > 0 and dispute_count == 0:
            confidence += min(0.20, concordant_count * 0.03)

        # Bounded and relative penalties for disputes and genuine unconfirmed omissions
        if dispute_count > 0:
            if paired_count > 0:
                dispute_ratio = dispute_count / paired_count
                confidence -= min(0.25, round(dispute_ratio * 0.25, 2))
            else:
                confidence -= min(0.25, round(dispute_count * 0.05, 2))

        if genuine_unconfirmed_count > 0:
            # Modest penalty for actual benchmark dataset omissions (ignoring sub-organ breakdowns)
            confidence -= min(0.08, round(genuine_unconfirmed_count * 0.02, 2))

        if consensus_screening.status == ConsensusStatus.DISPUTED:
            confidence -= 0.15
        if consensus_metadata.model_id_status == ConsensusStatus.DISPUTED:
            confidence -= 0.10

        has_consensus_dispute = (
            dispute_count > 0
            or consensus_screening.status == ConsensusStatus.DISPUTED
            or consensus_metadata.model_id_status == ConsensusStatus.DISPUTED
        )

        if not consensus_screening.is_included or has_critical_anomaly:
            final_status = AuditStatus.REJECTED
        elif has_repo_inspection and total_unique_benchmarks == 0:
            final_status = AuditStatus.NEEDS_REPO_INSPECTION
        elif has_outreach and total_unique_benchmarks == 0:
            final_status = AuditStatus.NEEDS_OUTREACH
        elif has_consensus_dispute:
            final_status = AuditStatus.FLAGGED_WARNING
        elif len(all_genuine_outliers) > 0 or confidence < 0.80:
            final_status = AuditStatus.FLAGGED_WARNING
        else:
            final_status = AuditStatus.VERIFIED

        if final_status == AuditStatus.REJECTED:
            confidence = min(confidence, 0.40)
            if not consensus_screening.is_included:
                confidence = 0.0

        confidence = max(0.0, min(1.0, round(confidence, 2)))

        # 7. Formulate Comprehensive Summary
        summary_lines = [
            f"=== Dual-Agent Consensus Audit Certificate: {resolved_model_id} ===",
            f"Overall Status: {final_status.value.upper()}",
            f"Screening Consensus: {consensus_screening.status.value.upper()} (Included={consensus_screening.is_included})",
            f"Metadata Consensus: Model ID={consensus_metadata.model_id_status.value.upper()}, Params={consensus_metadata.params_status.value.upper()}",
            f"Inter-Model Dice MAE: {f'{inter_model_dice_mae:.4f}' if inter_model_dice_mae is not None else 'N/A'}",
            f"Concordance Rate on Paired: {concordance_rate * 100:.1f}% ({concordant_count} concordant, {dispute_count} disputed / {paired_count} paired | {unconfirmed_count} singletons of {total_unique_benchmarks} total)",
            f"Certified Confidence: {confidence * 100:.0f}% (Agent A={report_a.confidence_score * 100:.0f}%, Agent B={report_b.confidence_score * 100:.0f}%)",
        ]

        if dispute_count > 0:
            disputed_items = [cs for cs in consensus_scores if cs.status == ConsensusStatus.DISPUTED]
            summary_lines.append(f"Disputes ({len(disputed_items)}): " + "; ".join(f"{d.canonical_dataset_name} ({d.notes})" for d in disputed_items))

        if unconfirmed_count > 0:
            summary_lines.append(f"Unconfirmed Singletons: {unconfirmed_count} benchmarks verified by only one agent.")

        return ConsensusAuditReport(
            model_id=resolved_model_id,
            status=final_status,
            consensus_scores=consensus_scores,
            consensus_screening=consensus_screening,
            consensus_metadata=consensus_metadata,
            inter_model_dice_mae=inter_model_dice_mae,
            concordance_rate=concordance_rate,
            concordant_count=concordant_count,
            dispute_count=dispute_count,
            unconfirmed_count=unconfirmed_count,
            paired_count=paired_count,
            total_unique_benchmarks=total_unique_benchmarks,
            audit_report_a=report_a,
            audit_report_b=report_b,
            confidence_score=confidence,
            summary="\n".join(summary_lines),
        )
