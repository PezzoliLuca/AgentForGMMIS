"""Scouting package for autonomous 3D medical image segmentation literature discovery and screening."""

from agent_gmmis.scouting.deduplicator import ScoutDeduplicator
from agent_gmmis.scouting.harvesters import (
    ArxivHarvester,
    EuropePMCHarvester,
    GoogleScholarHarvester,
    OpenAlexHarvester,
)
from agent_gmmis.scouting.schemas import (
    DeduplicationStatus,
    GeneratedSearchQuery,
    RawPaperCandidate,
    ScoutCandidate,
    ScoutFunnelMetrics,
    ScoutQueryPlan,
    ScoutRunReport,
    ScoutScreeningResult,
    ScoutSource,
    ScoutThoughtProcess,
    ScoutTier,
    TriStateCriterion,
)
from agent_gmmis.scouting.scout_agent import ScoutAgent

__all__ = [
    "ArxivHarvester",
    "DeduplicationStatus",
    "EuropePMCHarvester",
    "GeneratedSearchQuery",
    "GoogleScholarHarvester",
    "OpenAlexHarvester",
    "RawPaperCandidate",
    "ScoutAgent",
    "ScoutCandidate",
    "ScoutDeduplicator",
    "ScoutFunnelMetrics",
    "ScoutQueryPlan",
    "ScoutRunReport",
    "ScoutScreeningResult",
    "ScoutSource",
    "ScoutThoughtProcess",
    "ScoutTier",
    "TriStateCriterion",
]

