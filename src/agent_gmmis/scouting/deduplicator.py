"""Stage 2 Deterministic Deduplication, Lifecycle Tracking, and Lexical Pre-Filtering."""

import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

from agent_gmmis.data.db_manager import DatabaseManager
from agent_gmmis.data.models import ModelMetadata
from agent_gmmis.scouting.harvesters import _extract_arxiv_id, _infer_date_from_arxiv_id, _normalize_date
from agent_gmmis.scouting.schemas import DeduplicationStatus, RawPaperCandidate


SURVEY_TITLE_PATTERNS = [
    r"\ba\s+(?:comprehensive\s+|systematic\s+|scoping\s+|literature\s+)?(?:survey|review)\b",
    r"\bsystematic\s+review\b",
    r"\bmeta-analysis\b",
    r"\bscoping\s+review\b",
    r"\bsurvey\s+of\b",
    r"\bsurvey\s+on\b",
    r"\breview\s+of\b",
    r"\bstate-of-the-art\s+review\b",
    r"\bbenchmarking\s+odyssey\b",
    r"\bcomparative\s+(?:study|evaluation|analysis)\s+of\b",
]

EXPLICIT_2D_NON_RADIOLOGICAL_TERMS = [
    "retinal vessel",
    "retinal fundus",
    "fundus image",
    "color fundus",
    "dermoscopy",
    "dermoscopic",
    "skin lesion segmentation",
    "histopathology",
    "histopathological",
    "whole slide image",
    "whole-slide image",
    "digital pathology",
    "fluorescence microscopy",
    "cell microscopy",
    "cell nuclei segmentation",
    "polyp segmentation in colonoscopy",
    "colonoscopy frames",
    "optical colonoscopy",
    "2d colonoscopy",
    "2d endoscopic",
    "surgical instrument segmentation",
]

EXPLICIT_PRIVATE_ONLY_PATTERNS = [
    r"\b(?:exclusively|solely)\s+on\s+(?:an?\s+)?(?:internal|in-house|private|proprietary)\b",
    r"\brestricted\s+to\s+a\s+confidential\s+private\s+cohort\b",
    r"\bwith\s+no\s+public\s+dataset\s+evaluation\b",
    r"\bwithout\s+public\s+benchmark\s+comparisons\b",
]

RADIOLOGICAL_OR_3D_RESCUE_TERMS = [
    "3d",
    "volumetric",
    "ct",
    "computed tomography",
    "mri",
    "magnetic resonance",
    "pet",
    "positron emission",
    "3d ultrasound",
    "cbct",
    "braTS",
    "amos",
    "btcv",
    "totalsegmentator",
    "flare",
    "kits",
    "lits",
    "multi-modality",
    "multimodal",
    "generalist",
    "universal",
]

SEGMENTATION_INDICATORS = [
    "segment",
    "segmentation",
    "mask",
    "delineat",
    "parcellat",
    "contour",
    "sam",
    "totalsegmentator",
]


def normalize_title_key(title: str) -> str:
    """Normalizes a paper title via NFKC and strips non-alphanumeric characters."""
    norm = unicodedata.normalize("NFKC", title or "").lower()
    norm = re.sub(r"[^a-z0-9\s]", " ", norm)
    return re.sub(r"\s+", " ", norm).strip()


def title_token_similarity(title_a: str, title_b: str) -> float:
    """Computes Jaccard token similarity on normalized titles (ignoring short stopwords)."""
    stopwords = {"the", "a", "an", "in", "on", "of", "for", "and", "with", "to", "via", "by", "from"}
    tokens_a = {t for t in normalize_title_key(title_a).split() if len(t) > 1 and t not in stopwords}
    tokens_b = {t for t in normalize_title_key(title_b).split() if len(t) > 1 and t not in stopwords}
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


def _extract_doi(link_or_doi: Optional[str]) -> Optional[str]:
    """Extracts normalized lowercase DOI from a URL or string (excluding arXiv pseudo-DOIs)."""
    if not link_or_doi:
        return None
    s = link_or_doi.strip()
    m = re.search(r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", s)
    if not m:
        return None
    doi = m.group(1).lower().rstrip(".")
    if "arxiv" in doi:
        return None
    return doi


class ScoutDeduplicator:
    """
    Executes Stage 2 of the ScoutAgent funnel:
    1. Cross-source candidate merging (arXiv + OpenAlex + EuropePMC + Google Scholar)
    2. Database entity lookup (`ALREADY_IN_DB` vs `LIFECYCLE_UPGRADE` vs `NOVEL_CANDIDATE`)
    3. Zero-LLM deterministic lexical pre-filtering
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        masked_model_ids: Optional[Set[str]] = None,
        ignore_db_models_after_date: Optional[str] = None,
    ):
        """
        Args:
            db_manager: SQLite DatabaseManager instance.
            masked_model_ids: Optional set of model IDs to hide from DB checks (for backtesting).
            ignore_db_models_after_date: Optional YYYY-MM-DD cutoff; models published on/after
                this date are treated as not yet in DB (for retrospective time-machine tests).
        """
        self.db_manager = db_manager
        self.masked_model_ids = {m.lower() for m in (masked_model_ids or set())}
        self.ignore_after = _normalize_date(ignore_db_models_after_date)
        self._db_models: List[ModelMetadata] = self._load_active_db_models()

    def _load_active_db_models(self) -> List[ModelMetadata]:
        all_models = self.db_manager.get_all_models()
        active: List[ModelMetadata] = []
        for m in all_models:
            if m.id.lower() in self.masked_model_ids:
                continue
            if self.ignore_after and m.first_publication_date:
                m_date = _normalize_date(m.first_publication_date)
                if m_date and m_date >= self.ignore_after:
                    continue
            active.append(m)
        return active

    @staticmethod
    def merge_raw_candidates(candidates: List[RawPaperCandidate]) -> List[RawPaperCandidate]:
        """Merges duplicate records fetched across multiple bibliographic sources."""
        merged_list: List[RawPaperCandidate] = []
        by_arxiv: Dict[str, int] = {}
        by_doi: Dict[str, int] = {}
        by_title: Dict[str, int] = {}

        for cand in candidates:
            arxiv_id = _extract_arxiv_id(cand.arxiv_id) or _extract_arxiv_id(cand.doi) or _extract_arxiv_id(cand.url)
            cand.arxiv_id = arxiv_id
            if arxiv_id and (not cand.publication_date or cand.publication_date.endswith("-01-01")):
                inferred = _infer_date_from_arxiv_id(arxiv_id)
                if inferred:
                    cand.publication_date = inferred
            if cand.pdf_url and "arxiv.org/pdf/" in cand.pdf_url and not arxiv_id:
                cand.pdf_url = None
            doi_clean = _extract_doi(cand.doi) or _extract_doi(cand.url)
            if doi_clean:
                cand.doi = doi_clean
            t_key = normalize_title_key(cand.title)

            existing_idx: Optional[int] = None
            if arxiv_id and arxiv_id in by_arxiv:
                existing_idx = by_arxiv[arxiv_id]
            elif doi_clean and doi_clean in by_doi:
                existing_idx = by_doi[doi_clean]
            elif t_key and t_key in by_title:
                existing_idx = by_title[t_key]

            if existing_idx is None:
                idx = len(merged_list)
                merged_list.append(cand.model_copy(deep=True))
                if arxiv_id:
                    by_arxiv[arxiv_id] = idx
                if doi_clean:
                    by_doi[doi_clean] = idx
                if t_key:
                    by_title[t_key] = idx
            else:
                target = merged_list[existing_idx]
                # Keep richest abstract
                if len(cand.abstract or "") > len(target.abstract or ""):
                    target.abstract = cand.abstract
                if not target.authors and cand.authors:
                    target.authors = cand.authors
                if not target.arxiv_id and cand.arxiv_id:
                    target.arxiv_id = cand.arxiv_id
                    by_arxiv[cand.arxiv_id] = existing_idx
                if not target.doi and cand.doi:
                    target.doi = cand.doi
                    by_doi[cand.doi] = existing_idx
                if not target.pmid and cand.pmid:
                    target.pmid = cand.pmid
                # Prefer peer-reviewed venue name over generic "arXiv" or "Google Scholar"
                if cand.venue and cand.venue not in ("arXiv", "Google Scholar"):
                    if not target.venue or target.venue in ("arXiv", "Google Scholar"):
                        target.venue = cand.venue
                if not target.pdf_url and cand.pdf_url:
                    target.pdf_url = cand.pdf_url
                if target.arxiv_id and not target.pdf_url:
                    target.pdf_url = f"https://arxiv.org/pdf/{target.arxiv_id}.pdf"
                target.cited_by_count = max(target.cited_by_count, cand.cited_by_count)
                for src in cand.sources:
                    if src not in target.sources:
                        target.sources.append(src)

        return merged_list

    def check_against_database(
        self, candidate: RawPaperCandidate
    ) -> Tuple[DeduplicationStatus, Optional[str], Optional[str]]:
        """
        Cross-checks a candidate against `database.sqlite`.

        Returns:
            Tuple of (DeduplicationStatus, matched_model_id, explanation).
        """
        cand_arxiv = _extract_arxiv_id(candidate.arxiv_id) or _extract_arxiv_id(candidate.url)
        cand_doi = _extract_doi(candidate.doi) or _extract_doi(candidate.url)
        cand_title_norm = normalize_title_key(candidate.title)

        for db_model in self._db_models:
            db_arxiv_first = _extract_arxiv_id(db_model.first_publication_link)
            db_arxiv_last = _extract_arxiv_id(db_model.last_publication_link)
            db_doi_first = _extract_doi(db_model.first_publication_link)
            db_doi_last = _extract_doi(db_model.last_publication_link)

            exact_id_match = False
            if cand_arxiv and cand_arxiv in (db_arxiv_first, db_arxiv_last):
                exact_id_match = True
            if cand_doi and cand_doi in (db_doi_first, db_doi_last):
                exact_id_match = True

            # Check title similarity or acronym prefix match
            title_sim = title_token_similarity(candidate.title, db_model.title or "")
            db_title_norm = normalize_title_key(db_model.title or "")
            title_match = (
                (cand_title_norm and cand_title_norm == db_title_norm)
                or title_sim >= 0.85
            )

            if exact_id_match or title_match:
                # Check if this represents a Preprint -> Peer-Reviewed Journal/Conference upgrade
                was_only_arxiv = (
                    (db_model.first_publisher or "").lower() == "arxiv"
                    and not db_doi_last
                    and (not db_model.last_publisher or db_model.last_publisher.lower() == "arxiv")
                )
                has_new_peer_reviewed_venue = bool(
                    cand_doi
                    and cand_doi != db_doi_first
                    and (not candidate.venue or candidate.venue.lower() not in ("arxiv", "google scholar"))
                )
                if was_only_arxiv and has_new_peer_reviewed_venue:
                    return (
                        DeduplicationStatus.LIFECYCLE_UPGRADE,
                        db_model.id,
                        f"Peer-reviewed publication upgrade detected for existing arXiv model '{db_model.id}' "
                        f"(new DOI: {cand_doi}, venue: {candidate.venue or 'Journal'}).",
                    )
                return (
                    DeduplicationStatus.ALREADY_IN_DB,
                    db_model.id,
                    f"Matches existing model '{db_model.id}' in database.sqlite.",
                )

        return (DeduplicationStatus.NOVEL_CANDIDATE, None, None)

    @staticmethod
    def apply_lexical_prefilter(candidate: RawPaperCandidate) -> Tuple[bool, Optional[str]]:
        """
        Applies fast, deterministic Zero-LLM lexical rules to filter out obvious non-candidates.

        Returns:
            Tuple of (passed_prefilter: bool, rejection_reason: Optional[str]).
        """
        title_low = (candidate.title or "").lower()
        abstract_low = (candidate.abstract or "").lower()
        combined = f"{title_low} {abstract_low}"

        # 1. Exclude pure review / survey / meta-analysis papers based on title
        for pat in SURVEY_TITLE_PATTERNS:
            if re.search(pat, title_low):
                return (False, f"Stage 2 Lexical Filter: Title indicates a survey/review paper ('{candidate.title}').")

        # 2. Must mention segmentation/mask/delineation/SAM in title or abstract (excluding "without ... segmentation")
        sanitized_for_seg = re.sub(r"without\s+(?:performing\s+)?(?:semantic\s+|instance\s+)?segmentation(?:\s+masks)?", "", combined)
        if sanitized_for_seg.strip() and not any(ind in sanitized_for_seg for ind in SEGMENTATION_INDICATORS):
            return (False, "Stage 2 Lexical Filter: Neither title nor abstract mentions segmentation or masks.")

        # 3. Exclude papers explicitly restricted to 2D non-radiological microscopy/optics when no 3D/radiological terms appear
        matched_2d_terms = [t for t in EXPLICIT_2D_NON_RADIOLOGICAL_TERMS if t in combined]
        if matched_2d_terms:
            has_3d_or_rad = any(
                re.search(rf"\b{re.escape(r.lower())}\b", combined)
                for r in RADIOLOGICAL_OR_3D_RESCUE_TERMS
            )
            if not has_3d_or_rad:
                return (
                    False,
                    f"Stage 2 Lexical Filter: Restricted to 2D non-radiological domain ({', '.join(matched_2d_terms)}) "
                    "with no 3D or radiological modalities.",
                )

        return (True, None)

    @staticmethod
    def compute_stage2_pre_rank(candidate: RawPaperCandidate) -> float:
        """
        Computes a fast, deterministic Zero-LLM relevance pre-score so that the highest-signal
        3D generalist and benchmark SOTA papers are prioritized first in the Stage 3 LLM queue.
        """
        title_low = (candidate.title or "").lower()
        abs_low = (candidate.abstract or "").lower()
        combined = f"{title_low} {abs_low}"
        score = 0.0

        # Bonus for explicit '<ModelAcronym>: ...' title pattern (e.g. 'CDPDNet:', 'TAGS:', 'MedSAM2:')
        if re.match(r"^[A-Za-z0-9\-\+]{2,18}:\s+", candidate.title or ""):
            score += 3.5

        # High-priority title signals (Generalist / Foundation / SAM / Universal / 3D / Text-Guided)
        for kw in [
            "medsam",
            "sam",
            "segment anything",
            "foundation model",
            "generalist",
            "universal",
            "promptable",
            "text-driven",
            "text-prompted",
            "text guidance",
            "open-vocabulary",
            "hybrid vision encoder",
        ]:
            if kw in title_low:
                score += 3.0
            elif kw in abs_low:
                score += 1.0

        for kw3d in ["3d", "volumetric", "ct", "mri", "pet"]:
            if re.search(rf"\b{kw3d}\b", title_low):
                score += 2.5
            elif re.search(rf"\b{kw3d}\b", abs_low):
                score += 1.0

        for bench in ["totalsegmentator", "amos", "btcv", "synapse", "brats", "flare", "kits", "lits", "autopet"]:
            if bench in combined:
                score += 1.5

        # Multi-source corroboration bonus
        score += 1.0 * max(0, len(candidate.sources) - 1)
        return score
