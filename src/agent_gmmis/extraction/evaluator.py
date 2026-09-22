"""Evaluation engine: compares extracted manuscript data against verified database ground truth."""

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional
from agent_gmmis.data.db_manager import DatabaseManager
from agent_gmmis.extraction.schemas import ExtractedPaperData, RawExtractedScore


@dataclass
class ExtractionEvaluationReport:
    """Quantitative comparison report between extracted data and ground truth."""

    model_id: str
    parameter_error_pct: Optional[float] = None
    vram_error_gb: Optional[int] = None
    code_url_match: Optional[bool] = None
    matched_datasets_count: int = 0
    ground_truth_datasets_count: int = 0
    dataset_precision: float = 0.0
    dataset_recall: float = 0.0
    dataset_f1: float = 0.0
    dice_mae: Optional[float] = None
    score_comparisons: List[Dict[str, float]] = field(default_factory=list)

    def summary(self) -> str:
        """Generates a human-readable summary of the evaluation."""
        lines = [
            f"=== Extraction Evaluation Report: {self.model_id} ===",
            f"Dataset Retrieval: Recall={self.dataset_recall:.2%}, Precision={self.dataset_precision:.2%}, F1={self.dataset_f1:.2%}",
            f"Matched Datasets: {self.matched_datasets_count} / {self.ground_truth_datasets_count}",
        ]
        if self.dice_mae is not None:
            lines.append(f"Dice Score MAE: {self.dice_mae:.4f}")
        if self.parameter_error_pct is not None:
            lines.append(f"Parameter Count Error: {self.parameter_error_pct:.2f}%")
        if self.code_url_match is not None:
            lines.append(f"Code Repository URL Match: {self.code_url_match}")
        return "\n".join(lines)


class ExtractionEvaluator:
    """Benchmarks ExtractedPaperData against human-curated GMMIS ground truth."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()

    def evaluate(
        self,
        extracted: ExtractedPaperData,
        ground_truth_id: Optional[str] = None,
        compare_source: str = "primary",
    ) -> ExtractionEvaluationReport:
        """Compares extracted paper data with ground truth in database.sqlite."""
        model_id = ground_truth_id or extracted.proposed_model_id
        gt_model = self.db.get_model(model_id)
        if not gt_model:
            raise ValueError(f"Ground truth model not found in database: {model_id}")

        gt_scores = self.db.get_scores_for_model(model_id)

        # 1. Evaluate Model Metadata
        param_err = None
        if extracted.millions_of_parameters and gt_model.millions_of_parameters:
            param_err = abs(extracted.millions_of_parameters - gt_model.millions_of_parameters) / gt_model.millions_of_parameters * 100.0

        vram_err = None
        if extracted.resources_total_vram_gb and gt_model.resources_total_vram:
            vram_err = abs(extracted.resources_total_vram_gb - gt_model.resources_total_vram)

        code_match = None
        if extracted.code_repository_url and gt_model.code:
            code_match = extracted.code_repository_url.strip().lower() in gt_model.code.strip().lower() or gt_model.code.strip().lower() in extracted.code_repository_url.strip().lower()

        # 2. Evaluate Benchmark Scores
        # Group extracted scores by dataset for the proposed model
        extracted_by_dataset: Dict[str, List[RawExtractedScore]] = {}
        for s in extracted.extracted_scores:
            if not s.is_baseline_comparison:
                d_clean = s.dataset_name.strip().lower()
                extracted_by_dataset.setdefault(d_clean, []).append(s)

        # Prepare GT scores (normalized dataset names)
        gt_map: Dict[str, float] = {}
        for d_name, d_scores in gt_scores.items():
            # Use 'primary' (paper's own report) or 'best' (future re-evaluations)
            score_val = d_scores.get(compare_source)
            if score_val is None:
                # Fallback to alternate if specific source not populated
                score_val = d_scores.get("primary") if compare_source != "primary" else d_scores.get("best")
            if score_val is not None:
                # normalize score from [0, 100] to [0, 1] if needed
                val = score_val / 100.0 if score_val > 1.0 else score_val
                gt_map[d_name.strip().lower()] = val

        # Compare matching datasets
        matched = []
        dice_diffs = []
        comparisons = []

        for d_clean, scores_list in extracted_by_dataset.items():
            # Check direct or fuzzy match against GT
            match_gt_key = None
            for gt_k in gt_map.keys():
                if d_clean in gt_k or gt_k in d_clean:
                    match_gt_key = gt_k
                    break

            if match_gt_key:
                true_dice = gt_map[match_gt_key]
                # If multiple scores exist (e.g. 1 pt vs 10 pts), match appropriately
                selected_score = scores_list[0]
                if len(scores_list) > 1:
                    # If evaluating against primary, prefer 1-point prompt if available
                    for s in scores_list:
                        p_str = (s.prompt_type or "").lower()
                        if re.search(r"\b1\s*(?:point|pt)s?\b", p_str):
                            selected_score = s
                            break
                    else:
                        selected_score = min(scores_list, key=lambda s: abs(s.dice_score - true_dice))

                pred_dice = selected_score.dice_score
                diff = abs(pred_dice - true_dice)
                dice_diffs.append(diff)
                matched.append(match_gt_key)
                comparisons.append({
                    "dataset": match_gt_key,
                    "prompt_type": selected_score.prompt_type or "N/A",
                    "predicted": pred_dice,
                    "ground_truth": true_dice,
                    "diff": diff,
                })

        n_gt = len(gt_map)
        n_pred = len(extracted_by_dataset)
        n_matched = len(matched)

        precision = (n_matched / n_pred) if n_pred > 0 else 0.0
        recall = (n_matched / n_gt) if n_gt > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        mae = (sum(dice_diffs) / len(dice_diffs)) if dice_diffs else None

        return ExtractionEvaluationReport(
            model_id=model_id,
            parameter_error_pct=param_err,
            vram_error_gb=vram_err,
            code_url_match=code_match,
            matched_datasets_count=n_matched,
            ground_truth_datasets_count=n_gt,
            dataset_precision=precision,
            dataset_recall=recall,
            dataset_f1=f1,
            dice_mae=mae,
            score_comparisons=comparisons,
        )
