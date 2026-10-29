"""Evaluation engine: compares extracted manuscript data against verified database ground truth."""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple
from agent_gmmis.data.db_manager import DatabaseManager
from agent_gmmis.extraction.alias_resolver import DatasetAliasResolver
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
    dice_mae_primary: Optional[float] = None
    dice_mae_best: Optional[float] = None
    dice_mae_regime_agnostic: Optional[float] = None
    score_comparisons: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        """Generates a human-readable summary of the evaluation."""
        lines = [
            f"=== Extraction Evaluation Report: {self.model_id} ===",
            f"Dataset Retrieval: Recall={self.dataset_recall:.2%}, Precision={self.dataset_precision:.2%}, F1={self.dataset_f1:.2%}",
            f"Matched Datasets: {self.matched_datasets_count} / {self.ground_truth_datasets_count}",
        ]
        if self.dice_mae is not None:
            lines.append(f"Dice Score MAE: {self.dice_mae:.4f}")
        if self.dice_mae_primary is not None and self.dice_mae_best is not None:
            lines.append(f"Dual Regime MAE: Primary={self.dice_mae_primary:.4f}, Best={self.dice_mae_best:.4f}, Regime-Agnostic={self.dice_mae_regime_agnostic:.4f}")
        if self.parameter_error_pct is not None:
            lines.append(f"Parameter Count Error: {self.parameter_error_pct:.2f}%")
        if self.code_url_match is not None:
            lines.append(f"Code Repository URL Match: {self.code_url_match}")
        return "\n".join(lines)


class ExtractionEvaluator:
    """Benchmarks ExtractedPaperData against human-curated GMMIS ground truth."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()
        self.alias_resolver = DatasetAliasResolver(db_manager=self.db)

    def _match_dataset_to_gt(self, d_name: str, gt_keys: List[str]) -> Optional[str]:
        """Matches a raw extracted dataset name against Ground Truth canonical keys."""
        if not d_name:
            return None
        d_clean = d_name.strip()
        # 1. Direct resolved alias
        resolved = self.alias_resolver.resolve(d_clean)
        if resolved:
            for gt_k in gt_keys:
                if resolved.lower() == gt_k.lower():
                    return gt_k

        # 2. Case-insensitive exact match
        for gt_k in gt_keys:
            if d_clean.lower() == gt_k.lower():
                return gt_k

        # 3. Word-boundary substring match
        d_lower = d_clean.lower()
        for gt_k in gt_keys:
            gt_lower = gt_k.lower()
            pattern_gt = r"\b" + re.escape(gt_lower) + r"\b"
            pattern_d = r"\b" + re.escape(d_lower) + r"\b"
            if re.search(pattern_gt, d_lower) or re.search(pattern_d, gt_lower):
                return gt_k

        # 4. Cross-compatibility between Synapse and BTCV
        if d_lower in ("synapse", "btcv"):
            for gt_k in gt_keys:
                if gt_k.lower() in ("synapse", "btcv"):
                    return gt_k

        return None

    def get_ground_truth_map(self, model_id: str, compare_source: str = "primary") -> Dict[str, float]:
        """Returns normalized canonical GT scores for a given model and compare_source."""
        gt_scores = self.db.get_scores_for_model(model_id)
        gt_map: Dict[str, float] = {}
        has_any_primary = any(s.get("primary") is not None for s in gt_scores.values())

        for d_name, d_scores in gt_scores.items():
            score_val = d_scores.get(compare_source)
            if score_val is None:
                if compare_source == "primary" and has_any_primary:
                    continue
                score_val = d_scores.get("best") if compare_source == "primary" else d_scores.get("primary")
            if score_val is not None:
                val = score_val / 100.0 if score_val > 1.0 else score_val
                canon_gt = self.alias_resolver.resolve(d_name) or d_name.strip()
                gt_map[canon_gt] = val
        return gt_map

    def _select_candidate_score(
        self, scores_list: List[RawExtractedScore], compare_source: str = "primary"
    ) -> Tuple[float, str]:
        """Deterministically selects or macro-averages candidate scores without peeking at GT."""
        if not scores_list:
            return 0.0, "N/A"
        if len(scores_list) == 1:
            s = scores_list[0]
            return s.dice_score or 0.0, s.prompt_type or "N/A"

        # 1. Prefer overall average/mean row if table reports individual organs/classes + average
        avg_candidates = [
            s for s in scores_list
            if s.organ_or_task and any(
                k in s.organ_or_task.lower()
                for k in ("average", "mean", "overall", "all organs", "all classes", "mdice", "total")
            )
        ]

        if avg_candidates:
            if compare_source == "best":
                selected = max(avg_candidates, key=lambda s: s.dice_score if s.dice_score is not None else -1.0)
                return selected.dice_score or 0.0, selected.prompt_type or "Peak Overall Macro-Average"
            else:
                one_click = [
                    s for s in avg_candidates
                    if s.prompt_type and any(p in s.prompt_type.lower() for p in ("1 point", "1 click", "1-point", "1-click"))
                ]
                if one_click:
                    selected = one_click[0]
                    return selected.dice_score or 0.0, selected.prompt_type or "1-Point Prompt"
                else:
                    selected = max(avg_candidates, key=lambda s: s.dice_score if s.dice_score is not None else -1.0)
                    return selected.dice_score or 0.0, selected.prompt_type or "Overall Macro-Average"

        # 2. Macro-average across unique reported organs/tasks
        organ_map: Dict[str, List[float]] = {}
        for s in scores_list:
            if s.dice_score is not None:
                organ_key = (s.organ_or_task or "default").strip().lower()
                organ_map.setdefault(organ_key, []).append(s.dice_score)

        organ_vals = [
            max(vals) if compare_source == "best" else (sum(vals) / len(vals))
            for vals in organ_map.values()
        ]
        pred_dice = sum(organ_vals) / len(organ_vals) if organ_vals else 0.0
        return pred_dice, f"Macro-average of {len(organ_map)} reported organs/splits"

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

        # 2. Evaluate Benchmark Scores (Dual Primary and Best Baselines)
        gt_primary_map = self.get_ground_truth_map(model_id, compare_source="primary")
        gt_best_map = self.get_ground_truth_map(model_id, compare_source="best")
        gt_map = gt_best_map if compare_source == "best" else gt_primary_map

        # Union of all known canonical GT benchmarks
        gt_keys = list(dict.fromkeys(list(gt_map.keys()) + list(gt_primary_map.keys()) + list(gt_best_map.keys())))

        # Group candidate scores by matching GT benchmark (prevents duplicate matches inflating Recall > 100%)
        gt_matched_scores: Dict[str, List[RawExtractedScore]] = {gt_k: [] for gt_k in gt_keys}
        unmatched_extracted_datasets: set[str] = set()

        for s in extracted.extracted_scores:
            if s.is_baseline_comparison or s.dice_score is None:
                continue
            raw_d = (s.dataset_name or "").strip()
            if not raw_d:
                continue

            matched_gt = self._match_dataset_to_gt(raw_d, gt_keys)
            if matched_gt:
                gt_matched_scores[matched_gt].append(s)
            else:
                canon_unmatched = self.alias_resolver.resolve(raw_d) or raw_d
                unmatched_extracted_datasets.add(canon_unmatched)

        # Compare matching datasets strictly without Oracle Data Leakage
        dice_diffs = []
        dice_diffs_primary = []
        dice_diffs_best = []
        dice_diffs_regime = []
        comparisons = []
        matched_gt_datasets = [gt_k for gt_k in gt_keys if gt_matched_scores[gt_k]]

        for match_gt_key in matched_gt_datasets:
            scores_list = gt_matched_scores[match_gt_key]
            true_dice = gt_map.get(match_gt_key)
            true_primary = gt_primary_map.get(match_gt_key)
            true_best = gt_best_map.get(match_gt_key)

            # Deterministic Candidate Selection
            pred_dice_p, prompt_label_p = self._select_candidate_score(scores_list, compare_source="primary")
            pred_dice_b, prompt_label_b = self._select_candidate_score(scores_list, compare_source="best")

            pred_dice = pred_dice_b if compare_source == "best" else pred_dice_p
            prompt_label = prompt_label_b if compare_source == "best" else prompt_label_p

            diff = abs(pred_dice - true_dice) if true_dice is not None else None
            if diff is not None:
                dice_diffs.append(diff)

            diff_p = abs(pred_dice_p - true_primary) if true_primary is not None else None
            if diff_p is not None:
                dice_diffs_primary.append(diff_p)

            diff_b = abs(pred_dice_b - true_best) if true_best is not None else None
            if diff_b is not None:
                dice_diffs_best.append(diff_b)

            # Minimum regime-agnostic diff (distance from closest valid GT regime)
            diff_ra_candidates = [
                abs(pred_dice - v) for v in (true_primary, true_best) if v is not None
            ]
            diff_ra = min(diff_ra_candidates) if diff_ra_candidates else None
            if diff_ra is not None:
                dice_diffs_regime.append(diff_ra)

            comp_entry: Dict[str, Any] = {
                "dataset": match_gt_key,
                "prompt_type": prompt_label,
                "predicted": round(pred_dice, 4),
                "ground_truth": round(true_dice, 4) if true_dice is not None else None,
                "diff": round(diff, 4) if diff is not None else None,
                "ground_truth_primary": round(true_primary, 4) if true_primary is not None else None,
                "diff_primary": round(diff_p, 4) if diff_p is not None else None,
                "ground_truth_best": round(true_best, 4) if true_best is not None else None,
                "diff_best": round(diff_b, 4) if diff_b is not None else None,
            }
            comparisons.append(comp_entry)

        # Active compare_source matched stats
        active_matched = [gt_k for gt_k in matched_gt_datasets if gt_k in gt_map]
        extra_regime_matched = [gt_k for gt_k in matched_gt_datasets if gt_k not in gt_map]
        n_gt = len(gt_map)
        n_matched = len(active_matched)
        n_pred = n_matched + len(unmatched_extracted_datasets) + len(extra_regime_matched)

        precision = (n_matched / n_pred) if n_pred > 0 else 0.0
        recall = (n_matched / n_gt) if n_gt > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        mae = (sum(dice_diffs) / len(dice_diffs)) if dice_diffs else None
        mae_p = (sum(dice_diffs_primary) / len(dice_diffs_primary)) if dice_diffs_primary else None
        mae_b = (sum(dice_diffs_best) / len(dice_diffs_best)) if dice_diffs_best else None
        mae_ra = (sum(dice_diffs_regime) / len(dice_diffs_regime)) if dice_diffs_regime else None

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
            dice_mae_primary=mae_p,
            dice_mae_best=mae_b,
            dice_mae_regime_agnostic=mae_ra,
            score_comparisons=comparisons,
        )
