"""Auditing and curation module ('Il Vigile') for AgentForGMMIS."""

from agent_gmmis.auditing.auditor import AuditorAgent
from agent_gmmis.auditing.outreach import OutreachAgent
from agent_gmmis.auditing.repo_inspector import RepoInspector
from agent_gmmis.auditing.schemas import (
    AuditAnomaly,
    AuditReport,
    AuditStatus,
    AuthorContactInfo,
    ConsensusAuditReport,
    ConsensusMetadata,
    ConsensusScore,
    ConsensusScreening,
    ConsensusStatus,
    DiscrepancyNotice,
    MissingDataDiagnosis,
    OutlierDetectionResult,
    OutreachDraft,
    OutreachStatus,
    RepoInspectionResult,
)

__all__ = [
    "AuditorAgent",
    "AuditReport",
    "AuditStatus",
    "AuditAnomaly",
    "OutlierDetectionResult",
    "DiscrepancyNotice",
    "MissingDataDiagnosis",
    "RepoInspector",
    "RepoInspectionResult",
    "OutreachAgent",
    "AuthorContactInfo",
    "OutreachDraft",
    "OutreachStatus",
    "ConsensusStatus",
    "ConsensusScore",
    "ConsensusScreening",
    "ConsensusMetadata",
    "ConsensusAuditReport",
]
