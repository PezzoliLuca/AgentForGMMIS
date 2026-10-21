from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from agent_gmmis.auditing.auditor import AuditorAgent
    from agent_gmmis.auditing.schemas import ConsensusAuditReport

from agent_gmmis.extraction.alias_resolver import DatasetAliasResolver
from agent_gmmis.extraction.extractor import ExtractionAgent
from agent_gmmis.extraction.pdf_parser import ParsedManuscript, PDFParser
from agent_gmmis.extraction.schemas import (
    DeliberationRoundSummary,
    ExtractedPaperData,
    RawExtractedScore,
    ReinterrogationScoreResponse,
)


def _find_matching_score_index(
    scores: List[RawExtractedScore],
    dataset_name: str,
    organ_or_task: Optional[str] = None,
    alias_resolver: Optional[DatasetAliasResolver] = None,
) -> Optional[int]:
    """Finds the index of the RawExtractedScore corresponding to dataset_name and organ_or_task."""
    norm_ds = dataset_name.lower().strip()
    canon_target = (alias_resolver.resolve(dataset_name) if alias_resolver else dataset_name).lower().strip()
    norm_organ = (organ_or_task or "").lower().strip()

    best_idx = None
    best_score = -1

    for idx, s in enumerate(scores):
        if s.is_baseline_comparison:
            continue
        s_ds = s.dataset_name.lower().strip()
        s_canon = (alias_resolver.resolve(s.dataset_name) if alias_resolver else s.dataset_name).lower().strip()

        ds_match = (s_ds == norm_ds) or (s_canon == canon_target)
        if not ds_match:
            continue

        s_organ = (s.organ_or_task or "").lower().strip()
        organ_match = (s_organ == norm_organ) or (not norm_organ) or (not s_organ)
        pts = 2 if organ_match else 1

        if pts > best_score:
            best_score = pts
            best_idx = idx

    return best_idx


class DualExtractionCoordinator:
    """
    Coordinates concurrent extraction of scientific manuscripts using two independent ExtractionAgent instances
    and orchestrates double-blind multi-round deliberation.
    """

    def __init__(
        self,
        agent_a: Optional[ExtractionAgent] = None,
        agent_b: Optional[ExtractionAgent] = None,
    ):
        self.agent_a = agent_a or ExtractionAgent()
        self.agent_b = agent_b or ExtractionAgent()
        self.parser = PDFParser()
        self.alias_resolver = DatasetAliasResolver()

    def extract_manuscript_parallel(
        self,
        manuscript: ParsedManuscript,
    ) -> Tuple[ExtractedPaperData, ExtractedPaperData]:
        """
        Executes parallel extraction of a pre-parsed manuscript using ThreadPoolExecutor.

        Args:
            manuscript: ParsedManuscript object extracted from the PDF.

        Returns:
            Tuple of (extracted_data_a, extracted_data_b).
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(self.agent_a.process_manuscript, manuscript)
            future_b = executor.submit(self.agent_b.process_manuscript, manuscript)

            # Retrieve results; re-raise exceptions with clear origin if either fails
            try:
                extracted_a = future_a.result()
            except Exception as exc:
                raise RuntimeError(f"Agent A extraction failed: {exc}") from exc

            try:
                extracted_b = future_b.result()
            except Exception as exc:
                raise RuntimeError(f"Agent B extraction failed: {exc}") from exc

        return extracted_a, extracted_b

    def extract_pdf_parallel(
        self,
        pdf_path: Union[str, Path],
    ) -> Tuple[ExtractedPaperData, ExtractedPaperData]:
        """
        Parses a PDF manuscript once and submits it to both extraction agents concurrently.

        Args:
            pdf_path: Filesystem path to the PDF manuscript.

        Returns:
            Tuple of (extracted_data_a, extracted_data_b).
        """
        manuscript = self.parser.parse(pdf_path)
        return self.extract_manuscript_parallel(manuscript)

    def extract_manuscript_with_deliberation(
        self,
        manuscript: ParsedManuscript,
        auditor: AuditorAgent,
        max_rounds: int = 3,
        tolerance: float = 0.01,
        benchmark_mode: bool = False,
        verbose: bool = True,
    ) -> Tuple[ExtractedPaperData, ExtractedPaperData, ConsensusAuditReport, List[DeliberationRoundSummary]]:
        """
        Executes parallel extraction and orchestrates double-blind iterative deliberation.

        Protocol:
        1. Round 1: Conduct initial independent parallel extraction with Agent A and Agent B.
        2. Audit consensus: identify DISPUTED benchmarks and SINGLE_AGENT_UNCONFIRMED omissions.
        3. If disputes/omissions exist and max_rounds > 1, execute iterative deliberation rounds:
           - For DISPUTED scores: re-interrogate both Agent A and Agent B blindly.
           - For UNCONFIRMED omissions: re-interrogate the omitting agent blindly.
           - Update extracted scores without sycophancy (agents maintain stances if verified).
           - Re-audit consensus.
           - Terminate early if all benchmarks become CONCORDANT or epistemic equilibrium is reached.

        Returns:
            Tuple of (extracted_a, extracted_b, final_audit_report, deliberation_trajectory).
        """
        from agent_gmmis.auditing.schemas import ConsensusStatus

        # Round 1: Initial parallel extraction
        extracted_a, extracted_b = self.extract_manuscript_parallel(manuscript)
        audit_report = auditor.audit_consensus(
            extracted_a,
            extracted_b,
            manuscript=manuscript,
            benchmark_mode=benchmark_mode,
            tolerance=tolerance,
        )

        deliberation_trajectory: List[DeliberationRoundSummary] = []

        if max_rounds <= 1 or (audit_report.dispute_count == 0 and audit_report.unconfirmed_count == 0):
            if verbose and max_rounds > 1:
                print(f"   ✨ Deliberation: Round 1 achieved full consensus ({audit_report.concordant_count} concordant, 0 disputes). No further rounds needed.")
            return extracted_a, extracted_b, audit_report, deliberation_trajectory

        resolved_model_id = (
            audit_report.consensus_metadata.resolved_model_id
            or extracted_a.proposed_model_id
            or (manuscript.title if manuscript else "Proposed Model")
        )

        for round_num in range(2, max_rounds + 1):
            disputes_before = audit_report.dispute_count
            unconfirmed_before = audit_report.unconfirmed_count
            concordant_before = audit_report.concordant_count
            concordance_rate_before = audit_report.concordance_rate
            mae_before = audit_report.inter_model_dice_mae

            if disputes_before == 0 and unconfirmed_before == 0:
                break

            if verbose:
                print(f"   🤝 [Deliberation Round {round_num}] Flagged: {disputes_before} disputes, {unconfirmed_before} unconfirmed (Concordance: {concordance_rate_before*100:.1f}%)")

            round_details: List[str] = []
            queries_to_run: List[Tuple[str, str, Any]] = []

            # Identify items to re-interrogate
            for cs in audit_report.consensus_scores:
                if cs.status == ConsensusStatus.DISPUTED:
                    # Query both agents blindly on discrepancy
                    queries_to_run.append(("dispute_a", cs.canonical_dataset_name, cs))
                    queries_to_run.append(("dispute_b", cs.canonical_dataset_name, cs))
                elif cs.status == ConsensusStatus.SINGLE_AGENT_UNCONFIRMED:
                    if cs.score_a is not None and cs.score_b is None:
                        # Agent B omitted
                        queries_to_run.append(("omission_b", cs.canonical_dataset_name, cs))
                    elif cs.score_b is not None and cs.score_a is None:
                        # Agent A omitted
                        queries_to_run.append(("omission_a", cs.canonical_dataset_name, cs))

            # Execute re-interrogations concurrently
            results: Dict[Tuple[str, str], ReinterrogationScoreResponse] = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, max(1, len(queries_to_run)))) as executor:
                future_map = {}
                for q_type, ds_key, cs in queries_to_run:
                    if q_type == "dispute_a":
                        f = executor.submit(
                            self.agent_a.reinterrogate_discrepancy,
                            manuscript,
                            cs.dataset_name,
                            cs.organ_or_task or "Overall",
                            cs.score_a,
                            cs.prompt_type,
                            resolved_model_id,
                        )
                    elif q_type == "dispute_b":
                        f = executor.submit(
                            self.agent_b.reinterrogate_discrepancy,
                            manuscript,
                            cs.dataset_name,
                            cs.organ_or_task or "Overall",
                            cs.score_b,
                            cs.prompt_type,
                            resolved_model_id,
                        )
                    elif q_type == "omission_a":
                        f = executor.submit(
                            self.agent_a.reinterrogate_omission,
                            manuscript,
                            cs.dataset_name,
                            resolved_model_id,
                        )
                    elif q_type == "omission_b":
                        f = executor.submit(
                            self.agent_b.reinterrogate_omission,
                            manuscript,
                            cs.dataset_name,
                            resolved_model_id,
                        )
                    else:
                        continue
                    future_map[f] = (q_type, ds_key, cs)

                for f in concurrent.futures.as_completed(future_map):
                    q_type, ds_key, cs = future_map[f]
                    try:
                        resp = f.result()
                        results[(q_type, ds_key)] = resp
                    except Exception as exc:
                        if verbose:
                            print(f"      ⚠️ Re-interrogation failed for {q_type} on '{ds_key}': {exc}")

            # Capture snapshot of scores before updates to check equilibrium
            scores_a_before = [(s.dataset_name, s.organ_or_task, s.dice_score) for s in extracted_a.extracted_scores]
            scores_b_before = [(s.dataset_name, s.organ_or_task, s.dice_score) for s in extracted_b.extracted_scores]

            # Apply responses to ExtractedPaperData
            # 1. Agent A Discrepancies
            for cs in audit_report.consensus_scores:
                if cs.status == ConsensusStatus.DISPUTED and ("dispute_a", cs.canonical_dataset_name) in results:
                    resp_a = results[("dispute_a", cs.canonical_dataset_name)]
                    idx_a = _find_matching_score_index(
                        extracted_a.extracted_scores,
                        cs.dataset_name,
                        cs.organ_or_task,
                        alias_resolver=self.alias_resolver,
                    )
                    if idx_a is not None:
                        if not resp_a.is_valid_benchmark:
                            removed = extracted_a.extracted_scores.pop(idx_a)
                            round_details.append(f"Agent A excluded {cs.canonical_dataset_name} ({resp_a.exclusion_reason or resp_a.rationale})")
                        else:
                            curr_score = extracted_a.extracted_scores[idx_a].dice_score
                            if resp_a.dice_score != curr_score:
                                extracted_a.extracted_scores[idx_a].dice_score = resp_a.dice_score
                                extracted_a.extracted_scores[idx_a].notes = resp_a.rationale
                                if resp_a.prompt_type:
                                    extracted_a.extracted_scores[idx_a].prompt_type = resp_a.prompt_type
                                round_details.append(f"Agent A updated {cs.canonical_dataset_name} {curr_score} -> {resp_a.dice_score} ({resp_a.rationale})")
                            else:
                                round_details.append(f"Agent A maintained {cs.canonical_dataset_name} at {curr_score}")

                # 2. Agent B Discrepancies
                if cs.status == ConsensusStatus.DISPUTED and ("dispute_b", cs.canonical_dataset_name) in results:
                    resp_b = results[("dispute_b", cs.canonical_dataset_name)]
                    idx_b = _find_matching_score_index(
                        extracted_b.extracted_scores,
                        cs.dataset_name,
                        cs.organ_or_task,
                        alias_resolver=self.alias_resolver,
                    )
                    if idx_b is not None:
                        if not resp_b.is_valid_benchmark:
                            removed = extracted_b.extracted_scores.pop(idx_b)
                            round_details.append(f"Agent B excluded {cs.canonical_dataset_name} ({resp_b.exclusion_reason or resp_b.rationale})")
                        else:
                            curr_score = extracted_b.extracted_scores[idx_b].dice_score
                            if resp_b.dice_score != curr_score:
                                extracted_b.extracted_scores[idx_b].dice_score = resp_b.dice_score
                                extracted_b.extracted_scores[idx_b].notes = resp_b.rationale
                                if resp_b.prompt_type:
                                    extracted_b.extracted_scores[idx_b].prompt_type = resp_b.prompt_type
                                round_details.append(f"Agent B updated {cs.canonical_dataset_name} {curr_score} -> {resp_b.dice_score} ({resp_b.rationale})")
                            else:
                                round_details.append(f"Agent B maintained {cs.canonical_dataset_name} at {curr_score}")

                # 3. Agent A Omissions
                if cs.status == ConsensusStatus.SINGLE_AGENT_UNCONFIRMED and cs.score_a is None and ("omission_a", cs.canonical_dataset_name) in results:
                    resp_a = results[("omission_a", cs.canonical_dataset_name)]
                    if resp_a.is_valid_benchmark and resp_a.dice_score is not None:
                        extracted_a.extracted_scores.append(
                            RawExtractedScore(
                                dataset_name=cs.dataset_name,
                                organ_or_task=resp_a.organ_or_task or cs.organ_or_task or "Overall",
                                dice_score=resp_a.dice_score,
                                prompt_type=resp_a.prompt_type or cs.prompt_type,
                                table_or_section_ref=resp_a.table_or_section_ref,
                                notes=f"Recovered during deliberation round {round_num}: {resp_a.rationale}",
                            )
                        )
                        round_details.append(f"Agent A confirmed omitted {cs.canonical_dataset_name} with score {resp_a.dice_score}")
                    else:
                        round_details.append(f"Agent A verified exclusion of {cs.canonical_dataset_name}: {resp_a.exclusion_reason or resp_a.rationale}")

                # 4. Agent B Omissions
                if cs.status == ConsensusStatus.SINGLE_AGENT_UNCONFIRMED and cs.score_b is None and ("omission_b", cs.canonical_dataset_name) in results:
                    resp_b = results[("omission_b", cs.canonical_dataset_name)]
                    if resp_b.is_valid_benchmark and resp_b.dice_score is not None:
                        extracted_b.extracted_scores.append(
                            RawExtractedScore(
                                dataset_name=cs.dataset_name,
                                organ_or_task=resp_b.organ_or_task or cs.organ_or_task or "Overall",
                                dice_score=resp_b.dice_score,
                                prompt_type=resp_b.prompt_type or cs.prompt_type,
                                table_or_section_ref=resp_b.table_or_section_ref,
                                notes=f"Recovered during deliberation round {round_num}: {resp_b.rationale}",
                            )
                        )
                        round_details.append(f"Agent B confirmed omitted {cs.canonical_dataset_name} with score {resp_b.dice_score}")
                    else:
                        round_details.append(f"Agent B verified exclusion of {cs.canonical_dataset_name}: {resp_b.exclusion_reason or resp_b.rationale}")

            # Re-evaluate consensus with updated extractions
            new_report = auditor.audit_consensus(
                extracted_a,
                extracted_b,
                manuscript=manuscript,
                benchmark_mode=benchmark_mode,
                tolerance=tolerance,
            )

            scores_a_after = [(s.dataset_name, s.organ_or_task, s.dice_score) for s in extracted_a.extracted_scores]
            scores_b_after = [(s.dataset_name, s.organ_or_task, s.dice_score) for s in extracted_b.extracted_scores]

            stalled = (scores_a_before == scores_a_after and scores_b_before == scores_b_after)
            resolved_count = max(0, (disputes_before + unconfirmed_before) - (new_report.dispute_count + new_report.unconfirmed_count))

            summary = DeliberationRoundSummary(
                round_number=round_num,
                disputes_before=disputes_before,
                unconfirmed_before=unconfirmed_before,
                concordant_before=concordant_before,
                concordance_rate_before=concordance_rate_before,
                inter_model_mae_before=mae_before,
                disputes_after=new_report.dispute_count,
                unconfirmed_after=new_report.unconfirmed_count,
                concordant_after=new_report.concordant_count,
                concordance_rate_after=new_report.concordance_rate,
                inter_model_mae_after=new_report.inter_model_dice_mae,
                resolved_count=resolved_count,
                stalled=stalled,
                details=round_details,
            )
            deliberation_trajectory.append(summary)
            audit_report = new_report

            if verbose:
                mae_str = f"{audit_report.inter_model_dice_mae:.4f}" if audit_report.inter_model_dice_mae is not None else "N/A"
                print(f"   🤝 [Round {round_num} Result] Concordance: {audit_report.concordance_rate*100:.1f}%, MAE: {mae_str} ({audit_report.concordant_count} concordant, {audit_report.dispute_count} disputed, {audit_report.unconfirmed_count} unconfirmed)")

            if audit_report.dispute_count == 0 and audit_report.unconfirmed_count == 0:
                if verbose:
                    print(f"   ✨ Deliberation converged: 100% consensus achieved on Round {round_num}!")
                break

            if stalled:
                if verbose:
                    print(f"   🛑 Deliberation reached epistemic equilibrium in Round {round_num} (no scores altered). Terminating early.")
                break

        return extracted_a, extracted_b, audit_report, deliberation_trajectory

    def extract_pdf_with_deliberation(
        self,
        pdf_path: Union[str, Path],
        auditor: AuditorAgent,
        max_rounds: int = 3,
        tolerance: float = 0.01,
        benchmark_mode: bool = False,
        verbose: bool = True,
    ) -> Tuple[ExtractedPaperData, ExtractedPaperData, ConsensusAuditReport, List[DeliberationRoundSummary]]:
        """
        Parses a PDF manuscript and runs extraction with multi-round double-blind deliberation.

        Args:
            pdf_path: Filesystem path to the PDF manuscript.
            auditor: AuditorAgent instance.
            max_rounds: Maximum deliberation rounds (default: 3).
            tolerance: Score difference tolerance for concordance.
            benchmark_mode: Whether to cross-reference ground truth.
            verbose: Whether to print round progress to console.

        Returns:
            Tuple of (extracted_a, extracted_b, final_audit_report, deliberation_trajectory).
        """
        manuscript = self.parser.parse(pdf_path)
        return self.extract_manuscript_with_deliberation(
            manuscript=manuscript,
            auditor=auditor,
            max_rounds=max_rounds,
            tolerance=tolerance,
            benchmark_mode=benchmark_mode,
            verbose=verbose,
        )

