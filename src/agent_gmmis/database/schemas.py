"""Pydantic v2 schemas and action contracts for DatabaseAgent."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class CurationActionType(str, Enum):
    """Types of actions performed during database curation."""

    INSERT_NEW_MODEL = "insert_new_model"
    UPDATE_MODEL_METADATA = "update_model_metadata"
    EVOLVE_SCHEMA_ADD_DATASET = "evolve_schema_add_dataset"
    INSERT_SCORE = "insert_score"
    UPDATE_PEAK_SCORE = "update_peak_score"
    PRESERVE_EXISTING_PEAK = "preserve_existing_peak"
    SKIP_DISPUTED_SCORE = "skip_disputed_score"
    SKIP_UNCONFIRMED_SCORE = "skip_unconfirmed_score"
    RECORD_PROVENANCE_COMMENT = "record_provenance_comment"


class ScoreCurationDiff(BaseModel):
    """Detailed record of a score evaluation, decision, and SQL mutation."""

    dataset_name: str = Field(..., description="Canonical dataset benchmark name")
    target_table: str = Field(..., description="'results_best' or 'results_primary'")
    action: CurationActionType
    previous_score: Optional[float] = Field(None, description="Previous score in DB (percentage [0, 100])")
    new_score: Optional[float] = Field(None, description="Candidate new score (percentage [0, 100])")
    delta_improvement_pct: Optional[float] = Field(None, description="Relative improvement percentage Δ%")
    consensus_status: str = Field("unknown", description="'concordant', 'disputed', or 'single_agent_unconfirmed'")
    comment: Optional[str] = Field(None, description="Provenance or caveat note")
    source_reference: Optional[str] = Field(None, description="Citation / paper URL reference")
    sql_statement: Optional[str] = Field(None, description="Exact SQL query generated for this score")


class DatasetSchemaChange(BaseModel):
    """Details of a dynamic schema evolution for novel public benchmarks."""

    dataset_name: str
    is_public: bool = True
    registered_in_metadata: bool = True
    columns_added: List[str] = Field(default_factory=list, description="List of tables altered with new column")
    sql_statements: List[str] = Field(default_factory=list, description="DDL ALTER TABLE statements")
    reason: str = ""


class CurationPlan(BaseModel):
    """Comprehensive plan of all actions and SQL statements for a given paper."""

    model_id: str
    paper_title: Optional[str] = None
    audit_status: str
    audit_confidence: float
    model_action: CurationActionType
    model_metadata_diff: Dict[str, Any] = Field(default_factory=dict)
    schema_changes: List[DatasetSchemaChange] = Field(default_factory=list)
    score_diffs: List[ScoreCurationDiff] = Field(default_factory=list)
    generated_sql: List[str] = Field(default_factory=list, description="Ordered executable SQL statements")
    summary: str = ""
    can_commit: bool = True
    rejection_reason: Optional[str] = None


class CurationResult(BaseModel):
    """Execution result of a curation action."""

    plan: CurationPlan
    committed: bool = False
    execution_timestamp: str = ""
    error: Optional[str] = None


class CurationHistoryEntry(BaseModel):
    """Record inserted into curation_history table for immutable scientific audit trail."""

    history_id: Optional[int] = None
    timestamp: str
    model_id: str
    action_type: str
    status: str  # 'COMMITTED', 'DRY_RUN', 'REVERTED'
    audit_status: Optional[str] = None
    audit_confidence: Optional[float] = None
    diff_payload: Optional[str] = None  # JSON serialized
    notes: Optional[str] = None
