"""Autonomous Literature ScoutAgent (Step 4) implementing the 4-Stage Filtering Funnel and Hybrid Agentic Discovery."""

from datetime import datetime
import json
from pathlib import Path
import re
import time
from typing import Dict, List, Optional, Set

from agent_gmmis.config import DEFAULT_DB_PATH, LOGS_DIR
from agent_gmmis.data.db_manager import DatabaseManager
from agent_gmmis.extraction.alias_resolver import DatasetAliasResolver
from agent_gmmis.llm.client import LLMClient, LLMConfig
from agent_gmmis.scouting.deduplicator import ScoutDeduplicator, normalize_title_key, title_token_similarity
from agent_gmmis.scouting.harvesters import (
    DEFAULT_SCOUT_CACHE_DIR,
    SYSTEMATIC_ARXIV_QUERIES,
    SYSTEMATIC_KEYWORD_QUERIES,
    ArxivHarvester,
    EuropePMCHarvester,
    GoogleScholarHarvester,
    OpenAlexHarvester,
    _is_within_date_window,
)
from agent_gmmis.scouting.prompts import (
    AGENTIC_QUERY_PLANNER_SYSTEM_PROMPT,
    SCOUT_TRIAGE_SYSTEM_PROMPT,
    build_query_planner_prompt,
    build_scout_triage_prompt,
)
from agent_gmmis.scouting.schemas import (
    DeduplicationStatus,
    RawPaperCandidate,
    ScoutCandidate,
    ScoutFunnelMetrics,
    ScoutQueryPlan,
    ScoutRunReport,
    ScoutScreeningResult,
    ScoutSource,
    ScoutTier,
    TriStateCriterion,
)


CANONICAL_3D_RADIOLOGICAL_BENCHMARKS = {
    "BraTS",
    "AMOS",
    "BTCV",
    "TotalSegmentator",
    "TotalSegmentator MRI",
    "FLARE",
    "KiTS",
    "LiTS",
    "MSD",
    "ACDC",
    "SegTHOR",
    "ISLES",
    "ATLAS v2.0",
    "ATLAS 2023",
    "WORD",
    "AbdomenCT-1K",
    "CHAOS",
    "M&Ms",
    "VerSe",
    "LUNA16",
    "AutoPET",
    "Hecktor",
}

GENERALIST_KEYWORDS = [
    "foundation model",
    "generalist",
    "segment anything",
    "sam",
    "sam2",
    "medsam",
    "promptable",
    "interactive segmentation",
    "point prompt",
    "box prompt",
    "text-prompted",
    "text-driven",
    "vision-language",
    "open-vocabulary",
    "universal",
    "in-context",
    "zero-shot",
    "multi-modality",
]


class ScoutAgent:
    """
    Autonomous 4-Stage Literature Discovery & Screening Agent:
    - Stage 1: Hybrid Multi-Source Harvesting (PRISMA Systematic + LLM Agentic Queries + Citation Snowballing)
    - Stage 2: Deterministic Cross-Source Deduplication, Lifecycle Tracking & Lexical Pre-Filtering
    - Stage 3: Abstract-Level LLM Triage (3-State Epistemic Logic + Tier 1/Tier 2 Classification)
    - Stage 4: Open-Access PDF Resolution & Prioritized Candidate Queue Generation
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        db_manager: Optional[DatabaseManager] = None,
        alias_resolver: Optional[DatasetAliasResolver] = None,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
        masked_model_ids: Optional[Set[str]] = None,
        ignore_db_models_after_date: Optional[str] = None,
    ):
        self.llm = llm_client or LLMClient(LLMConfig())
        self.db = db_manager or DatabaseManager(DEFAULT_DB_PATH)
        self.alias_resolver = alias_resolver or DatasetAliasResolver(self.db)
        self.cache_dir = cache_dir or DEFAULT_SCOUT_CACHE_DIR
        self.use_cache = use_cache

        self.arxiv = ArxivHarvester(cache_dir=self.cache_dir, use_cache=use_cache)
        self.openalex = OpenAlexHarvester(cache_dir=self.cache_dir, use_cache=use_cache)
        self.europepmc = EuropePMCHarvester(cache_dir=self.cache_dir, use_cache=use_cache)
        self.gscholar = GoogleScholarHarvester(cache_dir=self.cache_dir, use_cache=use_cache)

        self.deduplicator = ScoutDeduplicator(
            db_manager=self.db,
            masked_model_ids=masked_model_ids,
            ignore_db_models_after_date=ignore_db_models_after_date,
        )
        self._last_agentic_query_plan: Optional[ScoutQueryPlan] = None

    def plan_agentic_queries(
        self,
        user_objective: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> ScoutQueryPlan:
        """Uses the LLM to dynamically formulate targeted search queries based on DB state and emerging trends."""
        all_models = self.db.get_all_models()
        sorted_models = sorted(
            all_models,
            key=lambda m: m.first_publication_date or "",
            reverse=True,
        )
        recent_ids = [m.id for m in sorted_models[:20]]
        canonical_ds = list(CANONICAL_3D_RADIOLOGICAL_BENCHMARKS)

        prompt = build_query_planner_prompt(
            recent_db_models=recent_ids,
            canonical_benchmarks=canonical_ds,
            user_objective=user_objective,
            start_date=start_date,
            end_date=end_date,
        )
        return self.llm.generate_structured(
            prompt=prompt,
            response_model=ScoutQueryPlan,
            system_prompt=AGENTIC_QUERY_PLANNER_SYSTEM_PROMPT,
        )

    def harvest_stage1(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results_per_query: int = 35,
        enable_arxiv: bool = True,
        enable_openalex: bool = True,
        enable_europepmc: bool = True,
        enable_gscholar: bool = True,
        enable_citation_snowball: bool = True,
        enable_agentic_queries: bool = True,
        user_objective: Optional[str] = None,
        custom_queries: Optional[List[str]] = None,
    ) -> tuple[List[RawPaperCandidate], List[str], List[str], Dict[str, int]]:
        """
        Executes Stage 1 multi-source harvesting across configured bibliographic APIs.

        Returns:
            Tuple of (raw_candidates, systematic_queries_used, agentic_queries_used, counts_by_source).
        """
        raw_candidates: List[RawPaperCandidate] = []
        counts_by_source: Dict[str, int] = {
            ScoutSource.ARXIV.value: 0,
            ScoutSource.OPENALEX.value: 0,
            ScoutSource.EUROPE_PMC.value: 0,
            ScoutSource.GOOGLE_SCHOLAR.value: 0,
            ScoutSource.CITATION_SNOWBALL.value: 0,
        }

        systematic_queries = list(custom_queries) if custom_queries else list(SYSTEMATIC_KEYWORD_QUERIES)
        arxiv_queries = list(custom_queries) if custom_queries else list(SYSTEMATIC_ARXIV_QUERIES)

        agentic_queries: List[str] = []
        self._last_agentic_query_plan = None
        if enable_agentic_queries and not custom_queries:
            try:
                plan = self.plan_agentic_queries(
                    user_objective=user_objective,
                    start_date=start_date,
                    end_date=end_date,
                )
                self._last_agentic_query_plan = plan
                agentic_queries = [q.query_string for q in plan.queries if q.query_string.strip()]
            except Exception as exc:
                print(f"   ⚠️ [ScoutAgent] Agentic query planner warning: {exc}")

        all_keyword_queries = systematic_queries + agentic_queries

        # 1. arXiv API
        if enable_arxiv:
            for q in arxiv_queries + agentic_queries:
                res = self.arxiv.search(q, start_date=start_date, end_date=end_date, max_results=max_results_per_query)
                raw_candidates.extend(res)
                counts_by_source[ScoutSource.ARXIV.value] += len(res)

        # 2. OpenAlex API (Keyword Search + Citation Snowballing)
        if enable_openalex:
            for q in all_keyword_queries:
                res = self.openalex.search(q, start_date=start_date, end_date=end_date, max_results=max_results_per_query)
                raw_candidates.extend(res)
                counts_by_source[ScoutSource.OPENALEX.value] += len(res)

            if enable_citation_snowball:
                snowball_res = self.openalex.harvest_citations(
                    start_date=start_date,
                    end_date=end_date,
                    max_per_seed=max_results_per_query,
                )
                raw_candidates.extend(snowball_res)
                counts_by_source[ScoutSource.CITATION_SNOWBALL.value] += len(snowball_res)

        # 3. EuropePMC / PubMed API
        if enable_europepmc:
            for q in all_keyword_queries[:4]:
                res = self.europepmc.search(q, start_date=start_date, end_date=end_date, max_results=max_results_per_query)
                raw_candidates.extend(res)
                counts_by_source[ScoutSource.EUROPE_PMC.value] += len(res)

        # 4. Google Scholar Harvester (with automatic anti-CAPTCHA circuit breaker)
        if enable_gscholar:
            for q in all_keyword_queries[:2]:
                res = self.gscholar.search(q, start_date=start_date, end_date=end_date, max_results=20)
                raw_candidates.extend(res)
                counts_by_source[ScoutSource.GOOGLE_SCHOLAR.value] += len(res)

        return raw_candidates, systematic_queries, agentic_queries, counts_by_source

    def extract_canonical_datasets_from_text(
        self, title: str, abstract: str, llm_mentioned: Optional[List[str]] = None
    ) -> List[str]:
        """Resolves mentioned datasets in title/abstract to canonical GMMIS dataset names."""
        found: Set[str] = set()
        for raw_ds in llm_mentioned or []:
            resolved = self.alias_resolver.resolve(raw_ds)
            if resolved:
                found.add(resolved)

        combined_text = f"{title} {abstract}"
        for canonical in CANONICAL_3D_RADIOLOGICAL_BENCHMARKS:
            # Check exact canonical token or known alias in text
            pattern = rf"\b{re.escape(canonical)}\b"
            if re.search(pattern, combined_text, re.IGNORECASE):
                found.add(canonical)

        return sorted(found)

    def screen_candidate_stage3(self, raw_paper: RawPaperCandidate) -> ScoutCandidate:
        """
        Executes Stage 3 Abstract-Level LLM Triage on a single candidate paper,
        followed by deterministic epistemic guardrails and priority scoring.
        """
        pre_rank = self.deduplicator.compute_stage2_pre_rank(raw_paper)
        prompt = build_scout_triage_prompt(raw_paper)
        t0 = time.time()
        screening = self.llm.generate_structured(
            prompt=prompt,
            response_model=ScoutScreeningResult,
            system_prompt=SCOUT_TRIAGE_SYSTEM_PROMPT,
        )
        llm_latency = round(time.time() - t0, 3)

        canonical_datasets = self.extract_canonical_datasets_from_text(
            raw_paper.title,
            raw_paper.abstract,
            screening.mentioned_datasets,
        )
        overrides: List[str] = []

        # Deterministic Rescue Rule: if a canonical 3D radiological dataset (e.g. BraTS, BTCV, AMOS, TotalSegmentator)
        # is explicitly named in the abstract, override false-negative 2D/non-radiological flags.
        if any(ds in CANONICAL_3D_RADIOLOGICAL_BENCHMARKS for ds in canonical_datasets):
            if screening.is_3d_volumetric == TriStateCriterion.NO:
                screening.is_3d_volumetric = TriStateCriterion.YES
                overrides.append(
                    f"Canonical 3D Benchmark Rescue ({', '.join(canonical_datasets)}): overrode is_3d_volumetric NO -> YES"
                )
            if screening.is_radiological == TriStateCriterion.NO:
                screening.is_radiological = TriStateCriterion.YES
                overrides.append(
                    f"Canonical 3D Benchmark Rescue ({', '.join(canonical_datasets)}): overrode is_radiological NO -> YES"
                )
            if screening.uses_public_benchmarks == TriStateCriterion.IMPLICIT_UNKNOWN:
                screening.uses_public_benchmarks = TriStateCriterion.YES
                overrides.append(
                    f"Canonical 3D Benchmark Rescue ({', '.join(canonical_datasets)}): upgraded uses_public_benchmarks IMPLICIT_UNKNOWN -> YES"
                )

        # Deterministic Rejection Gatekeeping
        rejection_reasons: List[str] = []
        if not screening.is_novel_segmentation_model:
            rejection_reasons.append("Not a novel segmentation model (review, synthesis, or non-segmentation task).")

        # Guard against clinical application papers that merely apply an existing off-the-shelf baseline
        off_the_shelf_names = {
            "nnu-net",
            "3d nnu-net",
            "nnunet",
            "sam",
            "sam 2",
            "sam2",
            "segment anything model 2",
            "u-net",
            "3d u-net",
            "v-net",
            "swin unetr",
            "unetr",
        }
        prop_norm = (screening.proposed_model_name or "").strip().lower()
        if prop_norm in off_the_shelf_names or prop_norm.startswith("nnu-net"):
            msg = (
                f"Off-the-Shelf Baseline Guard: rejected clinical application of existing baseline "
                f"('{screening.proposed_model_name}')"
            )
            overrides.append(msg)
            rejection_reasons.append(
                f"Clinical application of existing off-the-shelf baseline ('{screening.proposed_model_name}') rather than a novel model architecture."
            )
        if screening.is_3d_volumetric == TriStateCriterion.NO:
            rejection_reasons.append("Explicitly restricted to 2D-only imaging.")
        if screening.is_radiological == TriStateCriterion.NO:
            rejection_reasons.append("Restricted to non-radiological domain (histopathology/retina/dermoscopy/endoscopy).")
        if screening.uses_public_benchmarks == TriStateCriterion.NO:
            rejection_reasons.append("Evaluated exclusively on private/proprietary clinical cohorts.")

        if rejection_reasons:
            final_tier = ScoutTier.REJECTED
            rejection_str = " | ".join(rejection_reasons)
        else:
            rejection_str = None
            if screening.tier in (ScoutTier.TIER_1_GENERALIST, ScoutTier.TIER_2_BENCHMARK_SOTA):
                final_tier = screening.tier
            else:
                # Infer tier if LLM passed all criteria but left tier ambiguous
                combined_low = f"{raw_paper.title} {raw_paper.abstract}".lower()
                is_gen = any(kw in combined_low for kw in GENERALIST_KEYWORDS)
                final_tier = ScoutTier.TIER_1_GENERALIST if is_gen else ScoutTier.TIER_2_BENCHMARK_SOTA
                overrides.append(f"Deterministic Tier Inference: upgraded ambiguous tier -> {final_tier.value}")
            screening.tier = final_tier

        priority = self._compute_priority_score(
            raw_paper=raw_paper,
            screening=screening,
            final_tier=final_tier,
            canonical_datasets=canonical_datasets,
        )

        pdf_ok = bool(raw_paper.arxiv_id or (raw_paper.pdf_url and raw_paper.pdf_url.startswith("http")))
        resolved_pdf = raw_paper.pdf_url or (
            f"https://arxiv.org/pdf/{raw_paper.arxiv_id}.pdf" if raw_paper.arxiv_id else None
        )

        return ScoutCandidate(
            paper=raw_paper,
            deduplication_status=DeduplicationStatus.NOVEL_CANDIDATE,
            rejection_reason=rejection_str,
            screening=screening,
            canonical_datasets_matched=canonical_datasets,
            final_tier=final_tier,
            priority_score=priority,
            pdf_accessible=pdf_ok,
            resolved_pdf_url=resolved_pdf,
            stage2_pre_rank_score=pre_rank,
            llm_latency_seconds=llm_latency,
            deterministic_overrides=overrides,
        )

    @staticmethod
    def _compute_priority_score(
        raw_paper: RawPaperCandidate,
        screening: ScoutScreeningResult,
        final_tier: ScoutTier,
        canonical_datasets: List[str],
    ) -> float:
        if final_tier == ScoutTier.REJECTED:
            return 0.0
        if final_tier == ScoutTier.LIFECYCLE_UPGRADE:
            return 0.90

        score = 0.75 if final_tier == ScoutTier.TIER_1_GENERALIST else 0.55

        if canonical_datasets:
            score += 0.10
        if (
            screening.is_3d_volumetric == TriStateCriterion.YES
            and screening.is_radiological == TriStateCriterion.YES
            and screening.uses_public_benchmarks == TriStateCriterion.YES
        ):
            score += 0.10
        if raw_paper.arxiv_id or raw_paper.pdf_url:
            score += 0.05

        return round(min(1.0, score), 4)

    def resolve_pdf_access_stage4(
        self, candidate: ScoutCandidate, allow_network_lookup: bool = True
    ) -> ScoutCandidate:
        """
        Stage 4 Open-Access PDF Resolver:
        1. Uses direct arXiv PDF URL if `arxiv_id` is known.
        2. Uses `pdf_url` from OpenAlex/EuropePMC/Scholar if present.
        3. If only a closed publisher DOI is available, queries arXiv by title to check for a preprint twin.
        """
        if candidate.paper.arxiv_id:
            candidate.resolved_pdf_url = f"https://arxiv.org/pdf/{candidate.paper.arxiv_id}.pdf"
            candidate.pdf_accessible = True
            return candidate

        if candidate.paper.pdf_url and candidate.paper.pdf_url.startswith("http"):
            candidate.resolved_pdf_url = candidate.paper.pdf_url
            candidate.pdf_accessible = True
            return candidate

        if allow_network_lookup and candidate.paper.title:
            try:
                title_clean = re.sub(r"[^a-zA-Z0-9\s]", " ", candidate.paper.title)
                arxiv_matches = self.arxiv.search(f'ti:"{title_clean[:80]}"', max_results=3)
                for match in arxiv_matches:
                    if title_token_similarity(candidate.paper.title, match.title) >= 0.80 and match.arxiv_id:
                        candidate.paper.arxiv_id = match.arxiv_id
                        candidate.resolved_pdf_url = f"https://arxiv.org/pdf/{match.arxiv_id}.pdf"
                        candidate.pdf_accessible = True
                        break
            except Exception:
                pass

        return candidate

    def follow_discovered_leads(
        self,
        promoted_candidates: List[ScoutCandidate],
        seen_titles: Set[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_leads: int = 5,
    ) -> List[RawPaperCandidate]:
        """
        Agentic 'Follow-the-Lead' Explorer:
        Extracts unknown model names mentioned in promoted abstracts and launches targeted
        follow-up queries on arXiv and OpenAlex to locate their primary manuscripts.
        """
        existing_db_ids = {m.id.lower() for m in self.db.get_all_models()}
        known_benchmarks = {b.lower() for b in CANONICAL_3D_RADIOLOGICAL_BENCHMARKS}

        leads_to_search: List[str] = []
        for cand in promoted_candidates:
            if not cand.screening:
                continue
            for lead in cand.screening.discovered_external_leads:
                lead_clean = lead.strip()
                if len(lead_clean) < 3 or len(lead_clean) > 40:
                    continue
                l_low = lead_clean.lower()
                if l_low in existing_db_ids or l_low in known_benchmarks:
                    continue
                if self.alias_resolver.resolve(lead_clean) is not None:
                    continue
                if lead_clean not in leads_to_search:
                    leads_to_search.append(lead_clean)

        followup_raw: List[RawPaperCandidate] = []
        for lead_name in leads_to_search[:max_leads]:
            q = f'"{lead_name}" medical image segmentation 3D'
            res_arxiv = self.arxiv.search(q, start_date=start_date, end_date=end_date, max_results=5)
            res_oa = self.openalex.search(q, start_date=start_date, end_date=end_date, max_results=5)
            for item in res_arxiv + res_oa:
                t_key = normalize_title_key(item.title)
                if t_key not in seen_titles:
                    item.sources = [ScoutSource.AGENTIC_LEAD_FOLLOWUP]
                    item.discovery_query = f"Lead Follow-up: {lead_name}"
                    followup_raw.append(item)
                    seen_titles.add(t_key)

        return followup_raw

    def run_campaign(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results_per_query: int = 30,
        max_llm_evaluations: int = 100,
        enable_arxiv: bool = True,
        enable_openalex: bool = True,
        enable_europepmc: bool = True,
        enable_gscholar: bool = True,
        enable_citation_snowball: bool = True,
        enable_agentic_queries: bool = True,
        enable_lead_followup: bool = True,
        user_objective: Optional[str] = None,
        custom_queries: Optional[List[str]] = None,
        pre_harvested_candidates: Optional[List[RawPaperCandidate]] = None,
    ) -> ScoutRunReport:
        """
        Runs the complete 4-Stage ScoutAgent pipeline and returns a structured `ScoutRunReport`.
        Can also operate on `pre_harvested_candidates` for deterministic offline testing.
        """
        run_id = datetime.now().strftime("scout_%Y%m%d_%H%M%S")
        metrics = ScoutFunnelMetrics()

        if pre_harvested_candidates is not None:
            raw_list = list(pre_harvested_candidates)
            sys_queries: List[str] = ["[Pre-harvested Batch]"]
            ag_queries: List[str] = []
            metrics.total_raw_harvested = len(raw_list)
        else:
            raw_list, sys_queries, ag_queries, by_source = self.harvest_stage1(
                start_date=start_date,
                end_date=end_date,
                max_results_per_query=max_results_per_query,
                enable_arxiv=enable_arxiv,
                enable_openalex=enable_openalex,
                enable_europepmc=enable_europepmc,
                enable_gscholar=enable_gscholar,
                enable_citation_snowball=enable_citation_snowball,
                enable_agentic_queries=enable_agentic_queries,
                user_objective=user_objective,
                custom_queries=custom_queries,
            )
            metrics.total_raw_harvested = len(raw_list)
            metrics.harvested_by_source = by_source

        # Stage 2: Cross-Source Merge, Date Window Enforcement, Database Check & Lexical Pre-Filter
        merged_list = self.deduplicator.merge_raw_candidates(raw_list)
        if start_date or end_date:
            merged_list = [
                c for c in merged_list if _is_within_date_window(c.publication_date, start_date, end_date)
            ]
        metrics.unique_after_cross_source_merge = len(merged_list)

        tier1_list: List[ScoutCandidate] = []
        tier2_list: List[ScoutCandidate] = []
        lifecycle_list: List[ScoutCandidate] = []
        rejected_list: List[ScoutCandidate] = []
        stage3_queue: List[RawPaperCandidate] = []

        seen_titles: Set[str] = {normalize_title_key(c.title) for c in merged_list}

        for cand in merged_list:
            db_status, matched_id, reason = self.deduplicator.check_against_database(cand)
            if db_status == DeduplicationStatus.ALREADY_IN_DB:
                metrics.skipped_already_in_db += 1
                continue
            if db_status == DeduplicationStatus.LIFECYCLE_UPGRADE:
                metrics.detected_lifecycle_upgrades += 1
                lifecycle_list.append(
                    ScoutCandidate(
                        paper=cand,
                        deduplication_status=DeduplicationStatus.LIFECYCLE_UPGRADE,
                        matched_db_model_id=matched_id,
                        rejection_reason=reason,
                        final_tier=ScoutTier.LIFECYCLE_UPGRADE,
                        priority_score=0.90,
                        pdf_accessible=bool(cand.arxiv_id or cand.pdf_url),
                        resolved_pdf_url=cand.pdf_url,
                    )
                )
                continue

            passed_lex, lex_reason = self.deduplicator.apply_lexical_prefilter(cand)
            if not passed_lex:
                metrics.rejected_stage2_lexical += 1
                rejected_list.append(
                    ScoutCandidate(
                        paper=cand,
                        deduplication_status=DeduplicationStatus.LEXICAL_REJECTED,
                        rejection_reason=lex_reason,
                        final_tier=ScoutTier.REJECTED,
                        priority_score=0.0,
                    )
                )
                continue

            stage3_queue.append(cand)

        # Pre-rank Stage 3 queue deterministically so top 3D/generalist candidates are screened first
        stage3_queue.sort(key=self.deduplicator.compute_stage2_pre_rank, reverse=True)

        # Stage 3: Abstract-Level LLM Triage
        for raw_cand in stage3_queue[:max_llm_evaluations]:
            try:
                screened = self.screen_candidate_stage3(raw_cand)
            except Exception as exc:
                print(f"   ⚠️ [ScoutAgent Stage 3] Skipping candidate '{raw_cand.title[:50]}...' due to LLM error: {exc}")
                continue
            metrics.evaluated_by_llm_stage3 += 1
            if screened.final_tier == ScoutTier.TIER_1_GENERALIST:
                metrics.promoted_tier1_generalist += 1
                tier1_list.append(screened)
            elif screened.final_tier == ScoutTier.TIER_2_BENCHMARK_SOTA:
                metrics.promoted_tier2_benchmark_sota += 1
                tier2_list.append(screened)
            else:
                metrics.rejected_stage3_llm += 1
                rejected_list.append(screened)

        # Agentic Follow-the-Lead Exploration (if enabled and not offline pre-harvested)
        if enable_lead_followup and pre_harvested_candidates is None and (tier1_list or tier2_list):
            followup_raw = self.follow_discovered_leads(
                promoted_candidates=tier1_list + tier2_list,
                seen_titles=seen_titles,
                start_date=start_date,
                end_date=end_date,
            )
            metrics.followup_leads_discovered = len(followup_raw)
            for f_cand in followup_raw:
                db_status, _, _ = self.deduplicator.check_against_database(f_cand)
                if db_status != DeduplicationStatus.NOVEL_CANDIDATE:
                    continue
                passed_lex, _ = self.deduplicator.apply_lexical_prefilter(f_cand)
                if not passed_lex:
                    continue
                metrics.evaluated_by_llm_stage3 += 1
                screened_f = self.screen_candidate_stage3(f_cand)
                if screened_f.final_tier == ScoutTier.TIER_1_GENERALIST:
                    metrics.promoted_tier1_generalist += 1
                    tier1_list.append(screened_f)
                elif screened_f.final_tier == ScoutTier.TIER_2_BENCHMARK_SOTA:
                    metrics.promoted_tier2_benchmark_sota += 1
                    tier2_list.append(screened_f)
                else:
                    metrics.rejected_stage3_llm += 1
                    rejected_list.append(screened_f)

        # Stage 4: Resolve Open-Access PDFs for promoted candidates
        allow_net = pre_harvested_candidates is None
        for cand in tier1_list + tier2_list + lifecycle_list:
            self.resolve_pdf_access_stage4(cand, allow_network_lookup=allow_net)
            if cand.pdf_accessible:
                metrics.pdfs_resolved_stage4 += 1

        # Sort promoted candidates by priority_score descending, then citation count
        tier1_list.sort(key=lambda c: (c.priority_score, c.paper.cited_by_count), reverse=True)
        tier2_list.sort(key=lambda c: (c.priority_score, c.paper.cited_by_count), reverse=True)

        return ScoutRunReport(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            start_date=start_date,
            end_date=end_date,
            llm_provider=self.llm.config.provider,
            llm_model=self.llm.config.model,
            agentic_query_plan=self._last_agentic_query_plan,
            agentic_queries_executed=ag_queries,
            systematic_queries_executed=sys_queries,
            metrics=metrics,
            tier1_candidates=tier1_list,
            tier2_candidates=tier2_list,
            lifecycle_upgrades=lifecycle_list,
            rejected_candidates=rejected_list,
        )

    @staticmethod
    def save_report(report: ScoutRunReport, output_dir: Optional[Path] = None) -> tuple[Path, Path]:
        """Persists `ScoutRunReport` as JSON, Executive Markdown, and Full LLM Chain-of-Thought Trace Markdown."""
        target_dir = output_dir or (LOGS_DIR / "scout_runs" / report.run_id)
        target_dir.mkdir(parents=True, exist_ok=True)

        json_path = target_dir / "scout_report.json"
        md_path = target_dir / "scout_report.md"
        cot_path = target_dir / "scout_cot_trace.md"

        json_path.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        md_path.write_text(ScoutAgent.format_markdown_report(report), encoding="utf-8")
        cot_path.write_text(ScoutAgent.format_cot_trace_report(report), encoding="utf-8")
        return json_path, md_path

    @staticmethod
    def format_markdown_report(report: ScoutRunReport) -> str:
        """Formats a comprehensive Markdown summary of the ScoutAgent campaign."""
        m = report.metrics
        lines = [
            f"# ScoutAgent Discovery & Screening Report (`{report.run_id}`)",
            "",
            f"- **Timestamp**: `{report.timestamp}`",
            f"- **Window**: `{report.start_date or 'All'}` to `{report.end_date or 'Present'}`",
            f"- **LLM Engine**: `{report.llm_provider}` / `{report.llm_model}`",
            "",
            "## 1. 4-Stage Filtering Funnel Telemetry",
            "",
            "| Funnel Stage | Metric | Count |",
            "| :--- | :--- | :---: |",
            f"| **Stage 1 (Harvesting)** | Total Raw Records Fetched | {m.total_raw_harvested} |",
            f"| **Stage 2 (Deduplication)** | Unique Candidates After Cross-Source Merge | {m.unique_after_cross_source_merge} |",
            f"| **Stage 2 (DB Check)** | Skipped (`ALREADY_IN_DB`) | {m.skipped_already_in_db} |",
            f"| **Stage 2 (Lifecycle)** | Preprint $\\rightarrow$ Journal Upgrades (`LIFECYCLE_UPGRADE`) | {m.detected_lifecycle_upgrades} |",
            f"| **Stage 2 (Lexical Filter)** | Deterministically Rejected (Surveys / 2D-Only) | {m.rejected_stage2_lexical} |",
            f"| **Stage 3 (LLM Triage)** | Evaluated on Title + Abstract | {m.evaluated_by_llm_stage3} |",
            f"| **Stage 3 (Promoted Tier 1)** | **`TIER_1_GENERALIST` (Foundation / Promptable 3D)** | **{m.promoted_tier1_generalist}** |",
            f"| **Stage 3 (Promoted Tier 2)** | **`TIER_2_BENCHMARK_SOTA` (Task-Specific SOTA 3D)** | **{m.promoted_tier2_benchmark_sota}** |",
            f"| **Stage 3 (Rejected)** | Rejected by LLM Epistemic Triage | {m.rejected_stage3_llm} |",
            f"| **Stage 4 (PDF Resolver)** | Open-Access PDFs Resolved | {m.pdfs_resolved_stage4} |",
            "",
        ]

        if report.agentic_queries_executed:
            lines.append("## 2. Agentic Queries Formulated by LLM")
            lines.append("")
            for q in report.agentic_queries_executed:
                lines.append(f"- `{q}`")
            lines.append("")

        lines.append(f"## 3. Promoted Tier 1 Candidates: Generalist & Foundation 3D Models ({len(report.tier1_candidates)})")
        lines.append("")
        if not report.tier1_candidates:
            lines.append("_No Tier 1 candidates found in this window._\n")
        else:
            lines.append("| Priority | Model Name | Title | Date | Venue | Matched Benchmarks | PDF |")
            lines.append("| :---: | :--- | :--- | :---: | :--- | :--- | :---: |")
            for c in report.tier1_candidates:
                m_name = (c.screening.proposed_model_name if c.screening else None) or "Unnamed"
                ds_str = ", ".join(c.canonical_datasets_matched) or "Implicit"
                pdf_str = f"[PDF]({c.resolved_pdf_url})" if c.resolved_pdf_url else "❌"
                lines.append(
                    f"| `{c.priority_score:.2f}` | **{m_name}** | {c.paper.title} | "
                    f"{c.paper.publication_date or 'N/A'} | {c.paper.venue or 'N/A'} | {ds_str} | {pdf_str} |"
                )
            lines.append("")

        lines.append(f"## 4. Promoted Tier 2 Candidates: 3D Benchmark SOTA Models ({len(report.tier2_candidates)})")
        lines.append("")
        if not report.tier2_candidates:
            lines.append("_No Tier 2 candidates found in this window._\n")
        else:
            lines.append("| Priority | Model Name | Title | Date | Venue | Matched Benchmarks | PDF |")
            lines.append("| :---: | :--- | :--- | :---: | :--- | :--- | :---: |")
            for c in report.tier2_candidates:
                m_name = (c.screening.proposed_model_name if c.screening else None) or "Unnamed"
                ds_str = ", ".join(c.canonical_datasets_matched) or "Implicit"
                pdf_str = f"[PDF]({c.resolved_pdf_url})" if c.resolved_pdf_url else "❌"
                lines.append(
                    f"| `{c.priority_score:.2f}` | **{m_name}** | {c.paper.title} | "
                    f"{c.paper.publication_date or 'N/A'} | {c.paper.venue or 'N/A'} | {ds_str} | {pdf_str} |"
                )
            lines.append("")

        if report.lifecycle_upgrades:
            lines.append(f"## 5. Detected Preprint-to-Journal Lifecycle Upgrades ({len(report.lifecycle_upgrades)})")
            lines.append("")
            for u in report.lifecycle_upgrades:
                lines.append(
                    f"- **`{u.matched_db_model_id}`**: {u.paper.title} $\\rightarrow$ *{u.paper.venue}* "
                    f"(DOI: `{u.paper.doi or 'N/A'}`, Date: `{u.paper.publication_date or 'N/A'}`)"
                )
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def format_cot_trace_report(report: ScoutRunReport) -> str:
        """
        Formats a rich, structured Chain-of-Thought (CoT) and decision-trace audit log
        covering Stage 0 (Agentic Query Planner CoT), Stage 2 (Deterministic Pre-Filtering),
        and Stage 3 (Full Per-Candidate LLM Chain-of-Thought, Tri-State Matrix, and Guard Overrides).
        """
        lines = [
            f"# ScoutAgent Structured Chain-of-Thought (CoT) & Decision Trace (`{report.run_id}`)",
            "",
            f"- **Timestamp**: `{report.timestamp}`",
            f"- **Publication Window**: `{report.start_date or 'All'}` to `{report.end_date or 'Present'}`",
            f"- **LLM Engine**: `{report.llm_provider}` / `{report.llm_model}`",
            "",
            "---",
            "",
            "## 1. Stage 0 — Agentic Query Planner Chain-of-Thought",
            "",
        ]

        if report.agentic_query_plan:
            plan = report.agentic_query_plan
            lines.extend(
                [
                    "### LLM Strategic Planning Rationale (`thought_process`)",
                    "",
                    f"> {plan.thought_process}",
                    "",
                    "### Formulated Search Queries",
                    "",
                    "| # | Search Query String | Target Scope | LLM Rationale |",
                    "| :---: | :--- | :---: | :--- |",
                ]
            )
            for idx, q in enumerate(plan.queries, 1):
                lines.append(
                    f"| {idx} | `{q.query_string}` | `{q.target_tier}` | {q.rationale or 'N/A'} |"
                )
            lines.append("")
        else:
            lines.append("_Agentic Query Planner was not invoked or ran in pre-harvested batch mode._\n")

        lines.extend(
            [
                "---",
                "",
                "## 2. Stage 3 — Full LLM Chain-of-Thought (CoT) Triage Cards",
                "",
            ]
        )

        llm_Rejected = [
            c for c in report.rejected_candidates if c.deduplication_status != DeduplicationStatus.LEXICAL_REJECTED
        ]
        all_llm_screened = (
            [(c, "PROMOTED (TIER 1 GENERALIST)") for c in report.tier1_candidates]
            + [(c, "PROMOTED (TIER 2 BENCHMARK SOTA)") for c in report.tier2_candidates]
            + [(c, "REJECTED BY STAGE 3") for c in llm_Rejected]
        )

        if not all_llm_screened:
            lines.append("_No candidates were evaluated by the Stage 3 LLM in this run._\n")
        else:
            for idx, (cand, status_label) in enumerate(all_llm_screened, 1):
                sc = cand.screening
                tp = sc.thought_process if sc else None
                m_name = (sc.proposed_model_name if sc else None) or "None"
                sources_str = ", ".join(s.value for s in cand.paper.sources) if cand.paper.sources else "N/A"
                abstract_snippet = (cand.paper.abstract or "[No abstract provided]").strip()

                lines.extend(
                    [
                        f"### [{idx}/{len(all_llm_screened)}] {cand.paper.title}",
                        "",
                        f"- **Final Decision**: **{status_label}** (`{cand.final_tier.value}`)",
                        f"- **Proposed Model Identified**: `{m_name}`",
                        f"- **Identifiers**: arXiv: `{cand.paper.arxiv_id or 'N/A'}` | DOI: `{cand.paper.doi or 'N/A'}` | Date: `{cand.paper.publication_date or 'N/A'}`",
                        f"- **Harvested From**: `{sources_str}` (Query: `{cand.paper.discovery_query or 'N/A'}`)",
                        f"- **Telemetry**: Stage 2 Pre-Rank = `{cand.stage2_pre_rank_score:.1f}` | LLM Latency = `{cand.llm_latency_seconds:.2f}s` | Priority Score = `{cand.priority_score:.2f}`",
                        "",
                        "#### Input Abstract",
                        f"> {abstract_snippet}",
                        "",
                        "#### Structured LLM Chain-of-Thought (`ScoutThoughtProcess`)",
                    ]
                )
                if tp:
                    lines.extend(
                        [
                            f"1. **Task & Novelty Analysis**: {tp.task_and_novelty_analysis or 'N/A'}",
                            f"2. **Dimensionality & Modality Analysis**: {tp.dimensionality_and_modality_analysis or 'N/A'}",
                            f"3. **Benchmark & Paradigm Analysis**: {tp.benchmark_and_paradigm_analysis or 'N/A'}",
                        ]
                    )
                else:
                    lines.append("_Structured `thought_process` not populated._")

                lines.extend(
                    [
                        f"4. **Executive Reasoning Summary (`reasoning`)**: {sc.reasoning if sc else 'N/A'}",
                        "",
                        "#### Epistemic Tri-State Matrix & Metadata",
                    ]
                )
                if sc:
                    lines.extend(
                        [
                            "| Criterion | Value |",
                            "| :--- | :--- |",
                            f"| `is_novel_segmentation_model` | `{sc.is_novel_segmentation_model}` |",
                            f"| `is_3d_volumetric` | `{sc.is_3d_volumetric.value}` |",
                            f"| `is_radiological` | `{sc.is_radiological.value}` |",
                            f"| `uses_public_benchmarks` | `{sc.uses_public_benchmarks.value}` |",
                            f"| `mentioned_datasets` (Raw LLM) | `{', '.join(sc.mentioned_datasets) or 'None'}` |",
                            f"| `canonical_datasets_matched` (AliasResolver) | `{', '.join(cand.canonical_datasets_matched) or 'None'}` |",
                            f"| `imaging_modalities` | `{', '.join(sc.imaging_modalities) or 'None'}` |",
                            f"| `discovered_external_leads` | `{', '.join(sc.discovered_external_leads) or 'None'}` |",
                            "",
                        ]
                    )

                if cand.deterministic_overrides:
                    lines.append("#### Deterministic Guard / Rescue Overrides Triggered")
                    for ov in cand.deterministic_overrides:
                        lines.append(f"- ⚡ **{ov}**")
                    lines.append("")

                if cand.rejection_reason:
                    lines.append(f"- ❌ **Rejection Reason**: `{cand.rejection_reason}`\n")
                elif cand.resolved_pdf_url:
                    lines.append(f"- 📄 **Resolved PDF URL**: {cand.resolved_pdf_url}\n")

                lines.append("---\n")

        lex_rejected = [
            c for c in report.rejected_candidates if c.deduplication_status == DeduplicationStatus.LEXICAL_REJECTED
        ]
        if lex_rejected:
            lines.extend(
                [
                    f"## 3. Stage 2 — Zero-LLM Deterministic Lexical Exclusions ({len(lex_rejected)})",
                    "",
                    "| # | Title | Date | Exclusion Trigger |",
                    "| :---: | :--- | :---: | :--- |",
                ]
            )
            for idx, c in enumerate(lex_rejected, 1):
                lines.append(
                    f"| {idx} | {c.paper.title} | {c.paper.publication_date or 'N/A'} | `{c.rejection_reason}` |"
                )
            lines.append("")

        return "\n".join(lines)
