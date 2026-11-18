"""Pydantic v2 data contracts for the Autonomous Literature ScoutAgent (Step 4)."""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ScoutSource(str, Enum):
    """Bibliographic data sources interrogated by the ScoutAgent."""

    ARXIV = "arxiv"
    OPENALEX = "openalex"
    EUROPE_PMC = "europe_pmc"
    GOOGLE_SCHOLAR = "google_scholar"
    CITATION_SNOWBALL = "citation_snowball"
    AGENTIC_LEAD_FOLLOWUP = "agentic_lead_followup"


class TriStateCriterion(str, Enum):
    """
    Three-valued epistemic evaluation for abstract-level inclusion criteria.
    Prevents false-negative rejections when an abstract omits explicit mention
    of metrics (e.g. Dice) or full dataset lists that only appear in the PDF body.
    """

    YES = "YES"
    IMPLICIT_UNKNOWN = "IMPLICIT_UNKNOWN"
    NO = "NO"


class ScoutTier(str, Enum):
    """Taxonomic classification and priority tier assigned by the ScoutAgent."""

    TIER_1_GENERALIST = "TIER_1_GENERALIST"
    TIER_2_BENCHMARK_SOTA = "TIER_2_BENCHMARK_SOTA"
    LIFECYCLE_UPGRADE = "LIFECYCLE_UPGRADE"
    REJECTED = "REJECTED"


class DeduplicationStatus(str, Enum):
    """Outcome of Stage 2 deterministic cross-check against database.sqlite."""

    NOVEL_CANDIDATE = "NOVEL_CANDIDATE"
    ALREADY_IN_DB = "ALREADY_IN_DB"
    LIFECYCLE_UPGRADE = "LIFECYCLE_UPGRADE"
    LEXICAL_REJECTED = "LEXICAL_REJECTED"


class RawPaperCandidate(BaseModel):
    """Normalized bibliographic metadata harvested in Stage 1 before LLM screening."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    title: str = Field(..., description="Full title of the paper")
    abstract: str = Field(default="", description="Abstract text")
    authors: List[str] = Field(default_factory=list, description="List of author names")
    publication_date: Optional[str] = Field(
        default=None, description="Publication or preprint date (YYYY-MM-DD or YYYY-MM)"
    )
    venue: Optional[str] = Field(
        default=None, description="Journal, conference, or preprint server (e.g. arXiv, IEEE TMI, MICCAI)"
    )
    arxiv_id: Optional[str] = Field(default=None, description="Canonical arXiv ID without version (e.g. 2504.03600)")
    doi: Optional[str] = Field(default=None, description="Digital Object Identifier (DOI)")
    pmid: Optional[str] = Field(default=None, description="PubMed ID if indexed in MEDLINE")
    url: Optional[str] = Field(default=None, description="Primary landing page URL")
    pdf_url: Optional[str] = Field(default=None, description="Direct open-access PDF link if available")
    cited_by_count: int = Field(default=0, description="Citation count reported by OpenAlex / Scholar")
    sources: List[ScoutSource] = Field(
        default_factory=list, description="All bibliographic sources that returned this candidate"
    )
    discovery_query: Optional[str] = Field(
        default=None, description="The query or seed paper that discovered this candidate"
    )


class GeneratedSearchQuery(BaseModel):
    """A single targeted search query formulated by the Agentic Query Planner."""

    query_string: str = Field(
        ..., description="Search query string combining medical segmentation keywords and architectural terms"
    )
    target_tier: ScoutTier = Field(
        default=ScoutTier.TIER_1_GENERALIST,
        description="Whether this query hunts for TIER_1_GENERALIST or TIER_2_BENCHMARK_SOTA models",
    )
    rationale: str = Field(..., description="Why this query targets an emerging paradigm or gap in the database")


class ScoutQueryPlan(BaseModel):
    """Structured output produced by the LLM Agentic Query Planner before harvesting."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    thought_process: str = Field(
        ...,
        description="Step-by-step analysis of recent medical segmentation trends, architectures, and benchmarks to target.",
    )
    queries: List[GeneratedSearchQuery] = Field(
        default_factory=list,
        description="List of 3 to 6 targeted search queries to execute across arXiv, OpenAlex, EuropePMC, and Google Scholar.",
    )


class ScoutThoughtProcess(BaseModel):
    """Structured multi-step Chain-of-Thought (CoT) reasoning executed by the LLM before emitting triage labels."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    task_and_novelty_analysis: str = Field(
        default="",
        description=(
            "Step 1: Analyze whether the paper introduces a genuinely novel medical image segmentation model/architecture "
            "vs a review/survey, non-segmentation task (synthesis/registration/classification/report-gen), or an "
            "off-the-shelf baseline application (e.g. applying standard nnU-Net or SAM 2)."
        ),
    )
    dimensionality_and_modality_analysis: str = Field(
        default="",
        description=(
            "Step 2: Analyze evidence for 3D volumetric radiological imaging (CT, MRI, PET, 3D US) vs 2D-only or "
            "non-radiological modalities (histopathology, retinal fundus, dermoscopy, endoscopy), noting if "
            "dimensionality is implicit (`IMPLICIT_UNKNOWN`)."
        ),
    )
    benchmark_and_paradigm_analysis: str = Field(
        default="",
        description=(
            "Step 3: Identify public benchmarks or clinical datasets (or note if dataset availability is "
            "`IMPLICIT_UNKNOWN` in the abstract) and determine whether the architecture is a Foundation/Generalist/"
            "Promptable model (`TIER_1_GENERALIST`) or a Task-Specific 3D SOTA architecture (`TIER_2_BENCHMARK_SOTA`)."
        ),
    )


class ScoutScreeningResult(BaseModel):
    """
    Structured output produced by Stage 3 (Abstract-Level LLM Triage).
    Note: `thought_process` and `reasoning` are placed first to enforce Schema-Embedded Chain-of-Thought.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    thought_process: Optional[ScoutThoughtProcess] = Field(
        default=None,
        description="Mandatory structured 3-step Chain-of-Thought analyzing task novelty, 3D radiological modality, and benchmark/tier classification.",
    )
    reasoning: str = Field(
        ...,
        description=(
            "Concise step-by-step rationale evaluating whether the paper proposes a 3D radiological "
            "medical image segmentation model, its paradigm (Generalist/Foundation vs Task-Specific SOTA), "
            "and any explicit exclusion triggers."
        ),
    )
    proposed_model_name: Optional[str] = Field(
        default=None,
        description="Short acronym or name of the proposed model (e.g. 'MedSAM-2', 'SegAnyPET', 'STU-Net'), or None if unnamed/not a model paper.",
    )
    is_novel_segmentation_model: bool = Field(
        ...,
        description="True if the paper proposes or adapts a medical image segmentation model/framework; False if it is a literature review/survey, pure classification, image synthesis/generation, or clinical study without a segmentation model.",
    )
    is_3d_volumetric: TriStateCriterion = Field(
        ...,
        description="YES if 3D/volumetric segmentation (CT, MRI, PET, 3D ultrasound) is mentioned; IMPLICIT_UNKNOWN if medical segmentation is discussed without restricting to 2D; NO if explicitly restricted only to 2D modalities.",
    )
    is_radiological: TriStateCriterion = Field(
        ...,
        description="YES if radiological modalities (CT, MRI, PET, Ultrasound, X-Ray) are used; IMPLICIT_UNKNOWN if general medical imaging is stated; NO if restricted exclusively to non-radiological microscopy, histopathology WSI, retinal fundus, dermoscopy, or 2D colonoscopy video.",
    )
    uses_public_benchmarks: TriStateCriterion = Field(
        ...,
        description="YES if public benchmarks/datasets are named or stated; IMPLICIT_UNKNOWN if dataset availability is not specified in the abstract; NO if the abstract explicitly states evaluation is exclusively on a private/internal clinical cohort.",
    )
    tier: ScoutTier = Field(
        ...,
        description=(
            "TIER_1_GENERALIST for 3D foundation, universal, multi-organ/multi-modality, promptable (SAM/box/click/text) or zero/few-shot models; "
            "TIER_2_BENCHMARK_SOTA for 3D task-specific architectures (CNN/Transformer/Mamba) evaluated on standard benchmarks; "
            "REJECTED if out of scope."
        ),
    )
    mentioned_datasets: List[str] = Field(
        default_factory=list,
        description="Dataset or challenge names explicitly mentioned in the title or abstract (e.g. ['AMOS', 'BTCV', 'BraTS', 'TotalSegmentator']).",
    )
    imaging_modalities: List[str] = Field(
        default_factory=list,
        description="Imaging modalities mentioned (e.g. ['CT', 'MRI', 'PET']).",
    )
    discovered_external_leads: List[str] = Field(
        default_factory=list,
        description="Names of other recent competing 3D medical segmentation models or new public benchmarks explicitly mentioned in the abstract that could be searched as follow-up leads.",
    )

    @field_validator("thought_process", mode="before")
    @classmethod
    def coerce_thought_process(cls, v):
        if v is None:
            return None
        if isinstance(v, ScoutThoughtProcess):
            return v
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            return ScoutThoughtProcess(task_and_novelty_analysis=v)
        return None

    @field_validator("is_novel_segmentation_model", mode="before")
    @classmethod
    def coerce_bool(cls, v):
        if v is None:
            return False
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", "t")
        return bool(v)

    @field_validator("is_3d_volumetric", "is_radiological", "uses_public_benchmarks", mode="before")
    @classmethod
    def coerce_tristate(cls, v):
        if isinstance(v, TriStateCriterion):
            return v
        if v is None:
            return TriStateCriterion.IMPLICIT_UNKNOWN
        s = str(v).strip().upper()
        if s in ("YES", "TRUE", "1"):
            return TriStateCriterion.YES
        if s in ("NO", "FALSE", "0"):
            return TriStateCriterion.NO
        return TriStateCriterion.IMPLICIT_UNKNOWN

    @field_validator("tier", mode="before")
    @classmethod
    def coerce_tier(cls, v):
        if isinstance(v, ScoutTier):
            return v
        if v is None:
            return ScoutTier.REJECTED
        s = str(v).strip().upper()
        for item in ScoutTier:
            if item.value == s or item.name == s:
                return item
        if "TIER_1" in s or "GENERALIST" in s or "FOUNDATION" in s:
            return ScoutTier.TIER_1_GENERALIST
        if "TIER_2" in s or "SOTA" in s or "TASK" in s or "BENCHMARK" in s:
            return ScoutTier.TIER_2_BENCHMARK_SOTA
        if "LIFECYCLE" in s or "UPGRADE" in s:
            return ScoutTier.LIFECYCLE_UPGRADE
        return ScoutTier.REJECTED


class ScoutCandidate(BaseModel):
    """Final enriched candidate produced after Stages 1–4 of the ScoutAgent funnel."""

    paper: RawPaperCandidate
    deduplication_status: DeduplicationStatus
    matched_db_model_id: Optional[str] = Field(
        default=None,
        description="If ALREADY_IN_DB or LIFECYCLE_UPGRADE, the ID of the matching model in database.sqlite",
    )
    rejection_reason: Optional[str] = Field(
        default=None, description="Deterministic or LLM reason if rejected in Stage 2 or Stage 3"
    )
    screening: Optional[ScoutScreeningResult] = None
    canonical_datasets_matched: List[str] = Field(
        default_factory=list,
        description="Canonical GMMIS datasets resolved via DatasetAliasResolver from the abstract",
    )
    final_tier: ScoutTier = ScoutTier.REJECTED
    priority_score: float = Field(
        default=0.0, description="Deterministic priority score in [0.0, 1.0] for extraction queue ordering"
    )
    pdf_accessible: bool = Field(
        default=False, description="True if a direct open-access PDF URL or arXiv ID was resolved in Stage 4"
    )
    resolved_pdf_url: Optional[str] = Field(
        default=None, description="Resolved direct PDF download link (arXiv, PMC, or preprint)"
    )
    stage2_pre_rank_score: float = Field(
        default=0.0, description="Deterministic Stage 2 pre-rank score assigned before LLM triage"
    )
    llm_latency_seconds: float = Field(
        default=0.0, description="Wall-clock seconds spent in Stage 3 LLM evaluation"
    )
    deterministic_overrides: List[str] = Field(
        default_factory=list,
        description="Log of any deterministic rescue or guard rules triggered in Stage 3",
    )


class ScoutFunnelMetrics(BaseModel):
    """Telemetry metrics across all 4 stages of a ScoutAgent campaign."""

    total_raw_harvested: int = 0
    harvested_by_source: Dict[str, int] = Field(default_factory=dict)
    unique_after_cross_source_merge: int = 0
    skipped_already_in_db: int = 0
    detected_lifecycle_upgrades: int = 0
    rejected_stage2_lexical: int = 0
    evaluated_by_llm_stage3: int = 0
    promoted_tier1_generalist: int = 0
    promoted_tier2_benchmark_sota: int = 0
    rejected_stage3_llm: int = 0
    followup_leads_discovered: int = 0
    pdfs_resolved_stage4: int = 0


class ScoutRunReport(BaseModel):
    """Complete structured artifact summarizing a standalone ScoutAgent execution."""

    run_id: str
    timestamp: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    llm_provider: str
    llm_model: str
    agentic_query_plan: Optional[ScoutQueryPlan] = Field(
        default=None, description="Full structured output and CoT from Stage 0 Agentic Query Planner"
    )
    agentic_queries_executed: List[str] = Field(default_factory=list)
    systematic_queries_executed: List[str] = Field(default_factory=list)
    metrics: ScoutFunnelMetrics
    tier1_candidates: List[ScoutCandidate] = Field(default_factory=list)
    tier2_candidates: List[ScoutCandidate] = Field(default_factory=list)
    lifecycle_upgrades: List[ScoutCandidate] = Field(default_factory=list)
    rejected_candidates: List[ScoutCandidate] = Field(default_factory=list)

