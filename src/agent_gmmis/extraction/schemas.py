"""Pydantic schemas specifically designed for paper extraction and LLM structured outputs."""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from agent_gmmis.data.models import InclusionCriteriaCheck


class RawExtractedScore(BaseModel):
    """A raw evaluation score extracted directly from paper text or tables."""

    dataset_name: str = Field(..., description="Name of dataset as mentioned in the paper (e.g. BraTS, BTCV, AMOS)")
    organ_or_task: str = Field(..., description="Target anatomy or lesion (e.g. Liver, Brain Tumor, Multi-organ mean)")
    dice_score: Optional[float] = Field(
        None,
        description="Dice Similarity Coefficient, normalized to [0.0, 1.0]. None if only reported graphically without numerical value.",
    )
    table_or_section_ref: Optional[str] = Field(None, description="Source table or section (e.g. Table 2, Table 3)")
    data_budget_or_setting: Optional[str] = Field(
        None,
        description="Data budget or experimental setting if semi-supervised, few-shot, or ablation (e.g. '10% labeled', '1% labeled', 'centralized')",
    )
    is_baseline_comparison: bool = Field(
        False,
        description="True if this row reports a baseline/competitor score; False if it is the proposed model",
    )
    evaluated_model_name: Optional[str] = Field(
        None,
        description="Name of model this score belongs to (e.g. MedSAM, nnU-Net)",
    )
    prompt_type: Optional[str] = Field(
        None,
        description="Prompt interaction setting if applicable (e.g. '1 point', '3 points', '5 points', '10 points', 'bounding box', 'automatic')",
    )
    notes: Optional[str] = Field(
        None,
        description="Qualitative extraction notes or deliberation rationale",
    )

    @field_validator("dice_score", mode="before")
    @classmethod
    def normalize_score(cls, v) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip().lower()
            if not s or any(phrase in s for phrase in ("not", "n/a", "none", "null", "unknown", "-")):
                return None
            import re
            m = re.search(r"[-+]?\d*\.?\d+", s)
            if m:
                try:
                    v = float(m.group(0))
                except ValueError:
                    return None
            else:
                return None
        try:
            val = float(v)
        except (ValueError, TypeError):
            return None
        if val > 1.0 and val <= 100.0:
            val = val / 100.0
        if val < 0.0 or val > 1.0:
            raise ValueError(f"Dice score must be in range [0.0, 1.0], got {val}")
        return round(val, 4)


class ExtractedPaperData(BaseModel):
    """Complete structured extraction result produced by ExtractionAgent."""

    paper_title: str = Field(..., description="Title of the paper")
    proposed_model_id: str = Field(..., description="Short identifier of the proposed model (e.g. MedSAM, BioSAM-2)")
    proposed_model_full_title: Optional[str] = Field(None, description="Full publication title")
    authors_affiliations: Optional[str] = Field(None, description="Author institutions/countries")
    code_repository_url: Optional[str] = Field(None, description="GitHub or code repository URL if disclosed")
    framework: Optional[str] = Field(None, description="Generalist vs Task-Specific or DL Framework")
    architecture: Optional[str] = Field(None, description="High-level architecture (e.g. SAM, Transformer with Convolutions)")
    visual_backbone: Optional[str] = Field(None, description="Vision encoder (e.g. ViT-B, Swin-B, ResNet)")
    millions_of_parameters: Optional[float] = Field(None, description="Parameter count in millions (M)")
    number_of_gflops: Optional[float] = Field(None, description="GFLOPs complexity count")
    resources: Optional[str] = Field(None, description="Hardware training resources (e.g. 8x NVIDIA A100 80GB)")
    resources_total_vram_gb: Optional[float] = Field(None, description="Total VRAM in GB across all training GPUs")

    # Screening & Evaluation
    screening: InclusionCriteriaCheck = Field(..., description="Scientific inclusion/exclusion screening check")
    extracted_scores: List[RawExtractedScore] = Field(
        default_factory=list,
        description="All quantitative Dice scores extracted from tables and text",
    )
    extraction_notes: Optional[str] = Field(None, description="Qualitative notes or ambiguities encountered")

    @field_validator("millions_of_parameters", "number_of_gflops", "resources_total_vram_gb", mode="before")
    @classmethod
    def parse_optional_numeric(cls, v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if not s or any(phrase in s for phrase in ("not", "n/a", "none", "null", "unknown", "unspecified", "undisclosed")):
                return None
            import re
            m = re.search(r"[-+]?\d*\.?\d+", s)
            if m:
                try:
                    return float(m.group(0))
                except ValueError:
                    return None
            return None
        return None


class ReinterrogationScoreResponse(BaseModel):
    """Structured response from an agent during double-blind deliberation re-interrogation."""

    dataset_name: str = Field(..., description="Canonical or reported dataset name")
    organ_or_task: str = Field(..., description="Target anatomy or evaluation task")
    is_valid_benchmark: bool = Field(
        True,
        description="True if this is a valid 3D radiological public evaluation of the proposed model; False if it should be excluded",
    )
    dice_score: Optional[float] = Field(
        None,
        description="Peak (maximum) Dice Similarity Coefficient for the proposed model. None if excluded or graphical-only",
    )
    prompt_type: Optional[str] = Field(
        None,
        description="Prompt regime achieving the peak score (e.g. 'bounding box', '5 points', '1 point', 'automated')",
    )
    table_or_section_ref: Optional[str] = Field(None, description="Source table or section reference in the manuscript")
    maintained_original: bool = Field(
        True,
        description="True if the agent maintained its previous stance; False if it updated or corrected the score",
    )
    rationale: str = Field(..., description="Concise explanation grounded in manuscript text/tables")
    exclusion_reason: Optional[str] = Field(
        None,
        description="Reason for exclusion if is_valid_benchmark=False (e.g. '2D modality', 'baseline model', 'private data')",
    )

    @field_validator("dice_score", mode="before")
    @classmethod
    def normalize_score(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        val = float(v)
        if val > 1.0 and val <= 100.0:
            val = val / 100.0
        if val < 0.0 or val > 1.0:
            raise ValueError(f"Dice score must be in range [0.0, 1.0], got {val}")
        return round(val, 4)


class DeliberationRoundSummary(BaseModel):
    """Summary record of a single deliberation round."""

    round_number: int
    disputes_before: int
    unconfirmed_before: int
    concordant_before: int
    concordance_rate_before: float
    inter_model_mae_before: Optional[float]
    disputes_after: int
    unconfirmed_after: int
    concordant_after: int
    concordance_rate_after: float
    inter_model_mae_after: Optional[float]
    resolved_count: int
    stalled: bool = False
    details: List[str] = Field(default_factory=list)

