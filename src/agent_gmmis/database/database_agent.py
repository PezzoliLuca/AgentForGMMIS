"""DatabaseAgent (Database Manager Agent): Autonomous curator for GMMIS knowledge base."""

import datetime
import json
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple, Union

from agent_gmmis.auditing.schemas import (
    AuditReport,
    AuditStatus,
    ConsensusAuditReport,
    ConsensusStatus,
)
from agent_gmmis.data.db_manager import DatabaseManager
from agent_gmmis.data.models import DatasetMetadata, ModelMetadata
from agent_gmmis.database.schemas import (
    CurationActionType,
    CurationPlan,
    CurationResult,
    DatasetSchemaChange,
    ScoreCurationDiff,
)
from agent_gmmis.extraction.alias_resolver import DatasetAliasResolver


class DatabaseAgent:
    """
    Autonomous Database Manager Agent responsible for gating, entity resolution,
    monotonic score curation, wide-table schema evolution, and ACID transactions.
    """

    PRIVATE_COHORT_KEYWORDS = {
        "internal",
        "in-house",
        "private",
        "hospital",
        "clinic",
        "cohort",
        "proprietary",
        "custom",
        "inhouse",
    }

    ORGAN_BLACKLIST = {
        "liver", "spleen", "kidney", "right kidney", "left kidney",
        "aorta", "pancreas", "gallbladder", "stomach", "heart",
        "whole heart", "brain", "lung", "right lung", "left lung",
        "prostate", "bladder", "esophagus", "trachea", "adrenal gland",
        "colon", "duodenum", "bowel", "rectum", "femur", "bone",
        "spine", "vertebrae", "eye", "optic chiasm", "thyroid",
        "ventricle", "myocardium", "atrium", "rib", "clavicle",
        "tumor", "lesion", "vessel", "artery", "vein", "portal vein",
        "inferior vena cava", "ivc", "caval vein", "splenic vein",
    }

    MODALITY_BLACKLIST = {
        "ct", "mri", "mr", "pet", "x-ray", "xray", "ultrasound", "us",
        "flair", "t1", "t2", "t1c", "t1ce", "t1w", "t2w", "dwi", "adc",
        "oct", "fundus", "endoscopy", "dermoscopy", "contrast", "non-contrast",
    }

    GENERIC_PLACEHOLDER_BLACKLIST = {
        "dataset", "dataset 1", "dataset 2", "dataset 3", "dataset 4", "dataset 5",
        "test", "test set", "testset", "testing set", "validation", "val", "val set",
        "train", "train set", "training", "training set", "benchmark", "benchmarks",
        "task", "task 1", "task 2", "task 3", "task 4", "task 5",
        "ours", "proposed", "baseline", "model", "overall", "average", "mean",
        "metric", "dice", "score", "external", "internal", "public", "private",
        "unseen", "seen", "challenge", "cohort", "data", "results",
    }

    GENERIC_MODEL_NAMES = {
        "ours", "proposed", "baseline", "model", "our method",
        "proposed method", "our model", "the proposed model", "architecture",
        "method", "framework", "approach",
    }

    WIDE_EVALUATION_TABLES = [
        "results_primary",
        "results_best",
        "results_best_source",
        "results_primary_comments",
        "results_best_comments",
    ]

    def __init__(
        self,
        db_manager: DatabaseManager,
        single_agent_min_conf: float = 0.85,
        default_source_ref: Optional[str] = None,
    ):
        self.db = db_manager
        self.single_agent_min_conf = single_agent_min_conf
        self.default_source_ref = default_source_ref
        self.alias_resolver = DatasetAliasResolver(db_manager)
        self.db.init_curation_history_table()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plan_curation(
        self,
        audit_report: Union[AuditReport, ConsensusAuditReport],
        paper_metadata: Optional[Dict[str, Any]] = None,
    ) -> CurationPlan:
        """
        Analyzes an audited paper against the current database state and produces
        an ordered CurationPlan containing all decisions, diffs, and exact SQL queries.
        """
        model_id = self._resolve_model_id(audit_report, paper_metadata)
        paper_title = self._resolve_paper_title(audit_report, paper_metadata)
        confidence = float(audit_report.confidence_score)
        status_str = audit_report.status.value if hasattr(audit_report.status, "value") else str(audit_report.status)

        plan = CurationPlan(
            model_id=model_id,
            paper_title=paper_title,
            audit_status=status_str,
            audit_confidence=confidence,
            model_action=CurationActionType.INSERT_NEW_MODEL,
        )

        # 1. Gatekeeping Verification
        can_proceed, rejection_reason = self._gatekeep_audit(audit_report)
        if not can_proceed:
            plan.can_commit = False
            plan.rejection_reason = rejection_reason
            plan.summary = f"Curation blocked: {rejection_reason}"
            return plan

        # 2. Entity Resolution & Model Ingestion SQL
        existing_model = self.db.get_model(model_id)
        if existing_model:
            model_id = existing_model.id
            plan.model_id = model_id
            plan.model_action = CurationActionType.UPDATE_MODEL_METADATA
            model_sql, diff_dict = self._generate_model_update_sql(existing_model, paper_metadata)
            plan.model_metadata_diff = diff_dict
            if model_sql:
                plan.generated_sql.append(model_sql)
        else:
            plan.model_action = CurationActionType.INSERT_NEW_MODEL
            model_sql_list = self._generate_new_model_sql(model_id, paper_title, paper_metadata)
            plan.generated_sql.extend(model_sql_list)

        # 3. Dynamic Schema Evolution & Score Ingestion
        source_ref = self._resolve_source_reference(paper_metadata)
        existing_best_cols = set(self.db.get_table_columns("results_best"))

        # In-memory tracking of running peak scores for this paper
        planned_best: Dict[str, float] = {}
        planned_primary: Dict[str, float] = {}

        # Extract score items based on report type and deduplicate
        raw_score_items = self._extract_score_items(audit_report)
        score_items = self._deduplicate_score_items(raw_score_items)

        for item in score_items:
            raw_dataset_name = self._clean_str(item.get("dataset_name", ""))
            resolved = self.alias_resolver.resolve(raw_dataset_name)
            canonical_name = self._clean_str(item.get("canonical_name") or resolved or raw_dataset_name)
            is_private = self._is_private_cohort(raw_dataset_name, item.get("notes", ""))

            # A. Epistemic Boundary: Reject private cohorts from wide-table schema
            if is_private:
                diff = ScoreCurationDiff(
                    dataset_name=raw_dataset_name,
                    target_table="results_best_comments",
                    action=CurationActionType.RECORD_PROVENANCE_COMMENT,
                    consensus_status=item.get("status", "unknown"),
                    comment=f"[Private In-House Cohort] Evaluated on {raw_dataset_name}: {item.get('score')}",
                    sql_statement=self._generate_comment_append_sql(
                        model_id,
                        f"[Private Cohort] {raw_dataset_name}: {item.get('score')}",
                    ),
                )
                plan.score_diffs.append(diff)
                if diff.sql_statement:
                    plan.generated_sql.append(diff.sql_statement)
                continue

            # B. Public Benchmark Schema Evolution
            # Schema evolution can ONLY occur if:
            # 1. Dataset is not private
            # 2. Status is CONCORDANT / AGREEMENT
            # 3. Name is a valid public benchmark (not organ/modality/placeholder)
            is_concordant = item.get("status") in (
                ConsensusStatus.CONCORDANT.value,
                "concordant",
                "agreement",
                "discrepancy_resolved",
            )
            is_valid_name = self._is_valid_new_benchmark_name(canonical_name)

            if canonical_name not in existing_best_cols:
                if is_concordant and is_valid_name:
                    schema_change = self._plan_schema_evolution(canonical_name, item)
                    plan.schema_changes.append(schema_change)
                    plan.generated_sql.extend(schema_change.sql_statements)
                    existing_best_cols.add(canonical_name)
                else:
                    # Non-qualifying candidate: do NOT alter DB schema!
                    # Divert to provenance comments in results_best_comments
                    reason = "Single-agent unconfirmed" if not is_concordant else f"Invalid benchmark name '{canonical_name}'"
                    score_val = item.get("score")
                    diff = ScoreCurationDiff(
                        dataset_name=canonical_name,
                        target_table="results_best_comments",
                        action=CurationActionType.RECORD_PROVENANCE_COMMENT,
                        consensus_status=item.get("status", "unknown"),
                        comment=f"[Schema Guard: {reason}] {canonical_name}: {score_val}",
                        sql_statement=self._generate_comment_append_sql(
                            model_id,
                            f"[Schema Guard: {reason}] {canonical_name}: {score_val}",
                        ),
                    )
                    plan.score_diffs.append(diff)
                    if diff.sql_statement:
                        plan.generated_sql.append(diff.sql_statement)
                    continue  # Do not attempt to UPDATE non-existent column in wide tables!

            # C. Epistemic Arbitration & Score Evaluation
            score_diffs = self._evaluate_and_generate_score_sql(
                model_id=model_id,
                canonical_name=canonical_name,
                item=item,
                source_ref=source_ref,
                audit_confidence=confidence,
                planned_best=planned_best,
                planned_primary=planned_primary,
            )
            for diff in score_diffs:
                plan.score_diffs.append(diff)
                if diff.sql_statement:
                    plan.generated_sql.append(diff.sql_statement)

        # 4. Generate Curation History Audit Trail Statement
        history_sql = self._generate_history_log_sql(plan)
        plan.generated_sql.append(history_sql)

        # 5. Executive Summary
        plan.summary = self._build_executive_summary(plan)
        return plan

    def curate(
        self,
        audit_report: Union[AuditReport, ConsensusAuditReport],
        paper_metadata: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
    ) -> CurationResult:
        """
        Executes curation for a paper. If dry_run=True, returns the plan and SQL
        without modifying the database. If dry_run=False, executes all SQL queries
        inside an atomic SQLite transaction.
        """
        plan = self.plan_curation(audit_report, paper_metadata)
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()

        if dry_run or not plan.can_commit:
            return CurationResult(
                plan=plan,
                committed=False,
                execution_timestamp=now_str,
            )

        # Atomic Transaction Execution
        try:
            self.db.execute_sql_script(plan.generated_sql)
            return CurationResult(
                plan=plan,
                committed=True,
                execution_timestamp=now_str,
            )
        except Exception as e:
            return CurationResult(
                plan=plan,
                committed=False,
                execution_timestamp=now_str,
                error=str(e),
            )

    # ------------------------------------------------------------------
    # Gatekeeping & Entity Resolution
    # ------------------------------------------------------------------
    def _gatekeep_audit(self, audit_report: Union[AuditReport, ConsensusAuditReport]) -> Tuple[bool, Optional[str]]:
        """Verifies if the audit certificate satisfies ingestion quality policies."""
        status = audit_report.status

        # Status check
        if status == AuditStatus.REJECTED:
            reasons = "; ".join(audit_report.rejection_reasons) if audit_report.rejection_reasons else "Sanity checks failed"
            return False, f"Paper rejected by Auditor ({reasons})"

        if status in (AuditStatus.NEEDS_OUTREACH, AuditStatus.NEEDS_REPO_INSPECTION):
            return False, f"Paper requires delegation before database ingestion ({status.value})"

        # Gated confidence for FLAGGED_WARNING
        if status == AuditStatus.FLAGGED_WARNING and float(audit_report.confidence_score) < 0.60:
            return False, f"Certified confidence {audit_report.confidence_score:.2f} is below minimum threshold (0.60)"

        return True, None

    def _clean_str(self, val: Any) -> str:
        """Normalizes Unicode (NFKC) and strips whitespace."""
        if val is None:
            return ""
        return unicodedata.normalize("NFKC", str(val)).strip()

    def _resolve_model_id(
        self,
        audit_report: Union[AuditReport, ConsensusAuditReport],
        paper_metadata: Optional[Dict[str, Any]],
    ) -> str:
        """
        Extracts, normalizes (Unicode NFKC), and resolves the canonical model identifier.
        Prevents placeholder names like 'Ours' or 'Proposed' and links to existing DB models.
        """
        candidates: List[str] = []

        if paper_metadata and paper_metadata.get("expected_model_id"):
            candidates.append(self._clean_str(paper_metadata["expected_model_id"]))

        if hasattr(audit_report, "consensus_metadata") and audit_report.consensus_metadata:
            resolved = audit_report.consensus_metadata.resolved_model_id
            if resolved:
                candidates.append(self._clean_str(resolved))

        if paper_metadata and paper_metadata.get("proposed_model_id"):
            candidates.append(self._clean_str(paper_metadata["proposed_model_id"]))

        if hasattr(audit_report, "model_id") and audit_report.model_id:
            candidates.append(self._clean_str(audit_report.model_id))

        # Filter out generic placeholder names
        valid_candidates = [
            c for c in candidates
            if c and c.lower() not in self.GENERIC_MODEL_NAMES
        ]

        # 1. Direct match against DB
        for cand in valid_candidates:
            existing = self.db.get_model(cand)
            if existing:
                return existing.id

        # 2. Case-insensitive and NFKC search against all DB models
        all_models = self.db.get_all_models()
        for cand in valid_candidates:
            cand_norm = cand.lower()
            for m in all_models:
                if self._clean_str(m.id).lower() == cand_norm:
                    return m.id

        # 3. Paper title matching against DB models
        paper_title = self._clean_str(paper_metadata.get("paper_title") if paper_metadata else None)
        if paper_title and len(paper_title) >= 12:
            clean_title = re.sub(r"[^\w\s]", "", paper_title.lower())
            for m in all_models:
                m_title_raw = getattr(m, "title", None)
                if m_title_raw:
                    m_title = re.sub(r"[^\w\s]", "", self._clean_str(m_title_raw).lower())
                    if m_title and (m_title == clean_title or (len(m_title) > 15 and m_title in clean_title) or (len(clean_title) > 15 and clean_title in m_title)):
                        return m.id

        # 4. Fallback to first non-generic candidate
        if valid_candidates:
            return valid_candidates[0]

        # 5. Fallback to audit_report.model_id if non-empty, else paper title
        if hasattr(audit_report, "model_id") and audit_report.model_id:
            raw = self._clean_str(audit_report.model_id)
            if raw and raw.lower() not in self.GENERIC_MODEL_NAMES:
                return raw

        if paper_title:
            words = paper_title.split()[:4]
            return "-".join(words)

        return "Unknown-Model"

    def _resolve_paper_title(
        self,
        audit_report: Union[AuditReport, ConsensusAuditReport],
        paper_metadata: Optional[Dict[str, Any]],
    ) -> str:
        """Extracts the paper title."""
        if paper_metadata and paper_metadata.get("paper_title"):
            return self._clean_str(paper_metadata["paper_title"])
        return self._resolve_model_id(audit_report, paper_metadata)

    def _resolve_source_reference(self, paper_metadata: Optional[Dict[str, Any]]) -> str:
        """Constructs the citation or publication link for results_best_source."""
        if self.default_source_ref:
            return self.default_source_ref
        if not paper_metadata:
            return "Manuscript extraction"

        url = paper_metadata.get("first_publication_link") or paper_metadata.get("code_repository_url")
        title = self._clean_str(paper_metadata.get("paper_title") or "")
        if url:
            return f"{title} ({url})".strip()
        return title or "Manuscript extraction"

    # ------------------------------------------------------------------
    # Epistemic Boundary (Private vs Public Datasets)
    # ------------------------------------------------------------------
    def _is_private_cohort(self, dataset_name: str, notes: str) -> bool:
        """
        Determines whether a dataset is a private in-house hospital cohort.
        Private datasets MUST NEVER trigger wide-table schema evolution.
        """
        # If it matches an established public canonical benchmark, it is public
        if self.alias_resolver.resolve(dataset_name) is not None:
            return False

        name_lower = dataset_name.lower()
        notes_lower = notes.lower()

        for kw in self.PRIVATE_COHORT_KEYWORDS:
            if re.search(rf"\b{re.escape(kw)}\b", name_lower) or re.search(rf"\b{re.escape(kw)}\b", notes_lower):
                return True

        return False

    def _is_valid_new_benchmark_name(self, name: str) -> bool:
        """
        Validates whether a dataset name qualifies as a legitimate novel 3D public benchmark.
        Rejects anatomical organs, generic imaging modalities, and placeholder labels.
        """
        if not name or not isinstance(name, str):
            return False

        name_clean = self._clean_str(name)
        if len(name_clean) < 3 or len(name_clean) > 80:
            return False

        # Must not be purely numeric or punctuation
        if re.match(r"^[\d\W_]+$", name_clean):
            return False

        name_lower = name_clean.lower()

        # Check exact matches against blacklists
        if name_lower in self.ORGAN_BLACKLIST:
            return False
        if name_lower in self.MODALITY_BLACKLIST:
            return False
        if name_lower in self.GENERIC_PLACEHOLDER_BLACKLIST:
            return False

        # Check regex patterns (e.g. "dataset 1", "task 4", "benchmark 2")
        if re.match(r"^(dataset|task|benchmark|cohort|split|group)\s*\d+$", name_lower):
            return False

        # Modality phrases: "ct scans", "abdominal ct scans", "chest x-ray", "ct test set", etc.
        if re.match(r"^(abdominal|chest|head|body|pelvic|brain)?\s*(ct|mri|x-ray|mr|pet)\s*(scan|scans|test set|test|dataset)?$", name_lower):
            return False

        # Check if the name is purely an organ name plus generic suffix (e.g. "liver tumor", "spleen segmentation")
        for organ in self.ORGAN_BLACKLIST:
            if re.match(rf"^{re.escape(organ)}\s*(segmentation|task|subset|mask|roi|eval)?$", name_lower):
                return False

        return True

    def _deduplicate_score_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Groups score items by canonical_name and selects the single most representative item
        per dataset to avoid duplicate or conflicting UPDATE statements on the same cell.
        Prefers Overall / Macro rows, then highest concordant score (peak), resolving organ sub-scores.
        """
        if not items:
            return []

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in items:
            raw_name = self._clean_str(item.get("dataset_name", ""))
            resolved = self.alias_resolver.resolve(raw_name)
            canon = self._clean_str(item.get("canonical_name") or resolved or raw_name)
            grouped.setdefault(canon, []).append(item)

        deduped: List[Dict[str, Any]] = []
        for canon, group in grouped.items():
            if len(group) == 1:
                deduped.append(group[0])
                continue

            # Check if any overall/macro row exists
            overall_rows = [
                x for x in group
                if self._clean_str(x.get("organ_or_task", "")).lower() in (
                    "overall", "average", "mean", "macro", "macro-average", "all", "total"
                )
            ]

            if overall_rows:
                # Prefer concordant overall rows, then pick peak score
                rep = max(
                    overall_rows,
                    key=lambda x: (
                        1 if x.get("status") in (ConsensusStatus.CONCORDANT.value, "concordant", "agreement") else 0,
                        float(x.get("score") or 0.0),
                    ),
                )
                rep = dict(rep)
            else:
                # Multiple sub-organ scores: calculate macro mean
                numeric_scores = [float(x["score"]) for x in group if x.get("score") is not None]
                if numeric_scores:
                    macro_val = round(sum(numeric_scores) / len(numeric_scores), 4)
                    rep = dict(group[0])
                    rep["organ_or_task"] = "Overall"
                    rep["score"] = macro_val
                    rep["notes"] = f"Synthesized macro mean from {len(numeric_scores)} organ sub-scores"
                else:
                    rep = dict(group[0])

            # Check if there is an explicit primary baseline candidate in the group
            primary_cands = [x for x in group if self._is_primary_setting(x.get("prompt_type", ""))]
            if primary_cands:
                best_prim = max(primary_cands, key=lambda x: float(x.get("score") or 0.0))
                rep["primary_score"] = best_prim.get("score")
                rep["primary_prompt_type"] = best_prim.get("prompt_type")

            deduped.append(rep)

        return deduped

    # ------------------------------------------------------------------
    # Dynamic Wide-Table Schema Evolution
    # ------------------------------------------------------------------
    def _plan_schema_evolution(self, canonical_name: str, item: Dict[str, Any]) -> DatasetSchemaChange:
        """Generates safe transactional DDL statements to alter all 5 wide tables."""
        statements = []
        tables_altered = []

        # 1. Register metadata into datasets table
        dataset_meta_sql = self._generate_dataset_metadata_sql(canonical_name, item)
        statements.append(dataset_meta_sql)

        # 2. ALTER TABLE for each wide evaluation table
        esc_col = self._escape_ident(canonical_name)
        for table in self.WIDE_EVALUATION_TABLES:
            col_type = "REAL" if "comments" not in table and "source" not in table else "TEXT"
            esc_table = self._escape_ident(table)
            sql = f'ALTER TABLE "{esc_table}" ADD COLUMN "{esc_col}" {col_type} DEFAULT NULL;'
            statements.append(sql)
            tables_altered.append(table)

        return DatasetSchemaChange(
            dataset_name=canonical_name,
            is_public=True,
            registered_in_metadata=True,
            columns_added=tables_altered,
            sql_statements=statements,
            reason=f"Novel public 3D radiological benchmark '{canonical_name}' evaluated in manuscript.",
        )

    def _generate_dataset_metadata_sql(self, canonical_name: str, item: Dict[str, Any]) -> str:
        """Generates an INSERT OR IGNORE INTO datasets query."""
        organ = item.get("organ_or_task", "Anatomical Target")
        notes = self._escape_sql(item.get("notes", "Public benchmark"))
        esc_name = self._escape_sql(canonical_name)
        return (
            f'INSERT OR IGNORE INTO datasets ("Name", "Full Name", "Description", "Imaging Modality", "Objects") '
            f"VALUES ('{esc_name}', '{esc_name}', '{notes}', '3D Volumetric', '{self._escape_sql(organ)}');"
        )

    # ------------------------------------------------------------------
    # Score Extraction & Epistemic Arbitration Engine
    # ------------------------------------------------------------------
    def _extract_score_items(self, audit_report: Union[AuditReport, ConsensusAuditReport]) -> List[Dict[str, Any]]:
        """Normalizes extracted scores from single-agent or dual-consensus reports."""
        items = []

        if isinstance(audit_report, ConsensusAuditReport):
            for cs in audit_report.consensus_scores:
                # Prefer resolved score, fallback to available score
                score_val = cs.resolved_score
                if score_val is None:
                    score_val = cs.score_a if cs.score_a is not None else cs.score_b

                items.append({
                    "dataset_name": cs.dataset_name,
                    "canonical_name": cs.canonical_dataset_name,
                    "organ_or_task": cs.organ_or_task,
                    "status": cs.status.value if hasattr(cs.status, "value") else str(cs.status),
                    "score": score_val,
                    "score_a": cs.score_a,
                    "score_b": cs.score_b,
                    "delta": cs.delta,
                    "prompt_type": cs.prompt_type,
                    "table_ref": cs.table_ref_a or cs.table_ref_b,
                    "notes": cs.notes or "",
                })
        else:
            # Single-agent report: extract scores from validated_scores
            for s in getattr(audit_report, "validated_scores", []):
                if s.dice_score is not None and not s.is_baseline_comparison:
                    canon_name = self.alias_resolver.resolve(s.dataset_name) if s.dataset_name else None
                    items.append({
                        "dataset_name": s.dataset_name,
                        "canonical_name": canon_name or s.dataset_name,
                        "organ_or_task": s.organ_or_task or "Overall",
                        "status": "single_agent_unconfirmed",
                        "score": s.dice_score,
                        "score_a": s.dice_score,
                        "score_b": None,
                        "delta": None,
                        "prompt_type": s.prompt_type,
                        "table_ref": s.table_or_section_ref,
                        "notes": s.notes or "",
                    })

        return items

    def _evaluate_and_generate_score_sql(
        self,
        model_id: str,
        canonical_name: str,
        item: Dict[str, Any],
        source_ref: str,
        audit_confidence: float,
        planned_best: Dict[str, float],
        planned_primary: Dict[str, float],
    ) -> List[ScoreCurationDiff]:
        """
        Applies epistemic policies (Concordant, Single-Agent Gated, Disputed)
        and monotonic peak improvement rules to generate score SQL.
        """
        diffs = []
        status = item.get("status", ConsensusStatus.CONCORDANT.value)
        raw_score = item.get("score")
        table_ref = item.get("table_ref") or "Manuscript Table"
        prompt_type = item.get("prompt_type") or "Standard"

        # Safe escape
        esc_id = self._escape_sql(model_id)
        esc_source = self._escape_sql(source_ref)
        esc_col = self._escape_ident(canonical_name)

        # -------------------------------------------------------------
        # POLICY 1: DISPUTED SCORES
        # -------------------------------------------------------------
        if status == ConsensusStatus.DISPUTED.value:
            comment_text = (
                f"[DISPUTE BLOCKED] Model A: {item.get('score_a')} vs Model B: {item.get('score_b')} "
                f"(Δ={item.get('delta')}); Ref: {table_ref}"
            )
            esc_comment = self._escape_sql(comment_text)
            sql = f'UPDATE "results_best_comments" SET "{esc_col}" = \'{esc_comment}\' WHERE ID = \'{esc_id}\';'

            diffs.append(ScoreCurationDiff(
                dataset_name=canonical_name,
                target_table="results_best_comments",
                action=CurationActionType.SKIP_DISPUTED_SCORE,
                consensus_status=status,
                comment=comment_text,
                sql_statement=sql,
            ))
            return diffs

        # -------------------------------------------------------------
        # POLICY 2: SINGLE-AGENT UNCONFIRMED SCORES
        # -------------------------------------------------------------
        if status == ConsensusStatus.SINGLE_AGENT_UNCONFIRMED.value:
            if audit_confidence < self.single_agent_min_conf:
                comment_text = (
                    f"[UNCONFIRMED OMITTED (conf={audit_confidence:.2f} < {self.single_agent_min_conf:.2f})] "
                    f"Score: {raw_score}; Ref: {table_ref}"
                )
                esc_comment = self._escape_sql(comment_text)
                sql = f'UPDATE "results_best_comments" SET "{esc_col}" = \'{esc_comment}\' WHERE ID = \'{esc_id}\';'

                diffs.append(ScoreCurationDiff(
                    dataset_name=canonical_name,
                    target_table="results_best_comments",
                    action=CurationActionType.SKIP_UNCONFIRMED_SCORE,
                    consensus_status=status,
                    comment=comment_text,
                    sql_statement=sql,
                ))
                return diffs

        if raw_score is None:
            return diffs

        # Normalize score to percentage format for SQLite [0.0, 100.0]
        score_pct = round(float(raw_score) * 100.0, 2) if float(raw_score) <= 1.0 else round(float(raw_score), 2)

        # Construct provenance comment
        provenance_prefix = (
            f"[Concordant Dual-Agent: Δ={item.get('delta') or 0.0000}]"
            if status == ConsensusStatus.CONCORDANT.value
            else f"[Unconfirmed Single-Agent: Conf {audit_confidence:.2f}]"
        )
        base_comment = f"{provenance_prefix} {prompt_type} setting; Ref: {table_ref}".strip()

        # -------------------------------------------------------------
        # POLICY 3: MONOTONIC PEAK IMPROVEMENT (results_best)
        # -------------------------------------------------------------
        current_best = planned_best.get(canonical_name)
        if current_best is None:
            current_best = self.db.get_score_raw(model_id, canonical_name, "results_best")

        if current_best is None:
            # First time record for this benchmark
            planned_best[canonical_name] = score_pct
            esc_comment = self._escape_sql(base_comment)
            best_sql = (
                f'UPDATE "results_best" SET "{esc_col}" = {score_pct} WHERE ID = \'{esc_id}\';\n'
                f'UPDATE "results_best_source" SET "{esc_col}" = \'{esc_source}\' WHERE ID = \'{esc_id}\';\n'
                f'UPDATE "results_best_comments" SET "{esc_col}" = \'{esc_comment}\' WHERE ID = \'{esc_id}\';'
            )
            diffs.append(ScoreCurationDiff(
                dataset_name=canonical_name,
                target_table="results_best",
                action=CurationActionType.INSERT_SCORE,
                previous_score=None,
                new_score=score_pct,
                consensus_status=status,
                comment=base_comment,
                source_reference=source_ref,
                sql_statement=best_sql,
            ))
        elif score_pct > current_best:
            # Monotonic improvement strictly verified
            delta_pct = round(((score_pct - current_best) / current_best) * 100.0, 2) if current_best > 0 else 100.0
            comment_with_gain = f"{base_comment} (Peak updated from {current_best:.2f}%, gain: +{delta_pct}%)"
            esc_comment = self._escape_sql(comment_with_gain)
            planned_best[canonical_name] = score_pct

            best_sql = (
                f'UPDATE "results_best" SET "{esc_col}" = {score_pct} WHERE ID = \'{esc_id}\';\n'
                f'UPDATE "results_best_source" SET "{esc_col}" = \'{esc_source}\' WHERE ID = \'{esc_id}\';\n'
                f'UPDATE "results_best_comments" SET "{esc_col}" = \'{esc_comment}\' WHERE ID = \'{esc_id}\';'
            )
            diffs.append(ScoreCurationDiff(
                dataset_name=canonical_name,
                target_table="results_best",
                action=CurationActionType.UPDATE_PEAK_SCORE,
                previous_score=current_best,
                new_score=score_pct,
                delta_improvement_pct=delta_pct,
                consensus_status=status,
                comment=comment_with_gain,
                source_reference=source_ref,
                sql_statement=best_sql,
            ))
        else:
            # Score is lower than or equal to existing peak -> Preserve existing peak!
            alt_note = f"Alternative evaluation: {score_pct}% ({prompt_type}, {table_ref})"
            esc_note = self._escape_sql(alt_note)
            preserve_sql = (
                f'UPDATE "results_best_comments" SET "{esc_col}" = '
                f'COALESCE("{esc_col}" || \' | \', \'\') || \'{esc_note}\' '
                f"WHERE ID = '{esc_id}';"
            )
            diffs.append(ScoreCurationDiff(
                dataset_name=canonical_name,
                target_table="results_best",
                action=CurationActionType.PRESERVE_EXISTING_PEAK,
                previous_score=current_best,
                new_score=score_pct,
                consensus_status=status,
                comment=f"Preserved existing peak ({current_best:.2f}% >= {score_pct}%). Logged alternative evaluation.",
                sql_statement=preserve_sql,
            ))

        # -------------------------------------------------------------
        # POLICY 4: PRIMARY STANDARDIZED BASELINE (results_primary)
        # -------------------------------------------------------------
        current_primary = planned_primary.get(canonical_name)
        if current_primary is None:
            current_primary = self.db.get_score_raw(model_id, canonical_name, "results_primary")

        primary_score_raw = item.get("primary_score") if item.get("primary_score") is not None else raw_score
        primary_pt = item.get("primary_prompt_type") if item.get("primary_prompt_type") is not None else prompt_type

        if current_primary is None and self._is_primary_setting(primary_pt) and primary_score_raw is not None:
            score_primary_pct = round(float(primary_score_raw) * 100.0, 2) if float(primary_score_raw) <= 1.0 else round(float(primary_score_raw), 2)
            planned_primary[canonical_name] = score_primary_pct
            esc_comment = self._escape_sql(f"{base_comment} (Standardized primary baseline)")
            primary_sql = (
                f'UPDATE "results_primary" SET "{esc_col}" = {score_primary_pct} WHERE ID = \'{esc_id}\';\n'
                f'UPDATE "results_primary_comments" SET "{esc_col}" = \'{esc_comment}\' WHERE ID = \'{esc_id}\';'
            )
            diffs.append(ScoreCurationDiff(
                dataset_name=canonical_name,
                target_table="results_primary",
                action=CurationActionType.INSERT_SCORE,
                previous_score=None,
                new_score=score_primary_pct,
                consensus_status=status,
                comment="Primary standardized baseline inserted",
                sql_statement=primary_sql,
            ))

        return diffs

    def _is_primary_setting(self, prompt_type: Optional[str]) -> bool:
        """Determines if the prompt setting qualifies as the standardized primary baseline."""
        if not prompt_type or not isinstance(prompt_type, str):
            return False
        pt = prompt_type.lower()
        return "1-point" in pt or "one-prompt" in pt or "1 click" in pt or "standard" in pt or "zero-shot" in pt

    # ------------------------------------------------------------------
    # SQL Statement Builders
    # ------------------------------------------------------------------
    def _generate_new_model_sql(
        self,
        model_id: str,
        paper_title: str,
        meta: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Generates insertion statements for models and initializes all wide tables."""
        m = meta or {}
        now_date = datetime.datetime.now().strftime("%Y-%m-%d 00:00:00")
        esc_id = self._escape_sql(model_id)
        esc_title = self._escape_sql(paper_title)
        esc_affil = self._escape_sql(m.get("authors_affiliations", "Unknown Affiliation"))
        esc_code = self._escape_sql(m.get("code_repository_url", ""))
        esc_arch = self._escape_sql(m.get("architecture", "Generalist Foundation Model"))
        esc_backbone = self._escape_sql(m.get("visual_backbone", "Not specified"))
        params = float(m["millions_of_parameters"]) if m.get("millions_of_parameters") else "NULL"
        gflops = float(m["number_of_gflops"]) if m.get("number_of_gflops") else "NULL"
        vram = int(m["resources_total_vram_gb"]) if m.get("resources_total_vram_gb") else "NULL"

        statements = [
            # 1. models table
            (
                f'INSERT OR IGNORE INTO "models" ('
                f'ID, Title, "Major Affiliations", "First Publication Date", "First Publisher", '
                f'"First Publication Link", Code, Framework, Architecture, "Visual Backbone", '
                f'"Millions of Parameters", "Number of GFlops", "Resources Total V-RAM"'
                f") VALUES ("
                f"'{esc_id}', '{esc_title}', '{esc_affil}', '{now_date}', 'arXiv / Conference', "
                f"'{self._escape_sql(m.get('first_publication_link', ''))}', '{esc_code}', 'PyTorch', "
                f"'{esc_arch}', '{esc_backbone}', {params}, {gflops}, {vram}"
                f");"
            ),
            # 2. results_primary
            f'INSERT OR IGNORE INTO "results_primary" (ID, "Related Paper", Date) VALUES (\'{esc_id}\', \'{esc_id}\', \'{now_date}\');',
            # 3. results_best
            f'INSERT OR IGNORE INTO "results_best" (ID, "Related Paper", Date) VALUES (\'{esc_id}\', \'{esc_id}\', \'{now_date}\');',
            # 4. results_best_source
            f'INSERT OR IGNORE INTO "results_best_source" (ID) VALUES (\'{esc_id}\');',
            # 5. results_primary_comments
            f'INSERT OR IGNORE INTO "results_primary_comments" (ID, "Related Paper", Date) VALUES (\'{esc_id}\', \'{esc_id}\', \'{now_date}\');',
            # 6. results_best_comments
            f'INSERT OR IGNORE INTO "results_best_comments" (ID, "Related Paper", Date) VALUES (\'{esc_id}\', \'{esc_id}\', \'{now_date}\');',
        ]
        return statements

    def _generate_model_update_sql(
        self,
        existing: ModelMetadata,
        meta: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Generates UPDATE statement for existing model metadata if new fields are provided."""
        if not meta:
            return None, {}

        updates = []
        diffs = {}
        esc_id = self._escape_sql(existing.id)

        # Code repo
        new_code = meta.get("code_repository_url")
        if new_code and not existing.code:
            updates.append(f'Code = \'{self._escape_sql(new_code)}\'')
            diffs["code"] = {"old": None, "new": new_code}

        # Parameters
        new_params = meta.get("millions_of_parameters")
        if new_params and existing.millions_of_parameters is None:
            updates.append(f'"Millions of Parameters" = {float(new_params)}')
            diffs["parameters_m"] = {"old": None, "new": float(new_params)}

        # Visual Backbone
        new_bb = meta.get("visual_backbone")
        if new_bb and (not existing.visual_backbone or existing.visual_backbone == "Not specified"):
            updates.append(f'"Visual Backbone" = \'{self._escape_sql(new_bb)}\'')
            diffs["visual_backbone"] = {"old": existing.visual_backbone, "new": new_bb}

        if updates:
            sql = f'UPDATE "models" SET {", ".join(updates)} WHERE ID = \'{esc_id}\';'
            return sql, diffs

        return None, {}

    def _generate_comment_append_sql(self, model_id: str, note_text: str) -> str:
        """Appends a note to commentOnPaper in results_best_comments."""
        esc_id = self._escape_sql(model_id)
        esc_note = self._escape_sql(note_text)
        return (
            f'UPDATE "results_best_comments" SET commentOnPaper = '
            f'COALESCE(commentOnPaper || \' | \', \'\') || \'{esc_note}\' '
            f"WHERE ID = '{esc_id}';"
        )

    def _generate_history_log_sql(self, plan: CurationPlan) -> str:
        """Generates an audit trail insertion into curation_history."""
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        esc_id = self._escape_sql(plan.model_id)
        action_type = plan.model_action.value
        status = "COMMITTED"
        audit_status = plan.audit_status
        confidence = plan.audit_confidence

        payload = {
            "model_action": action_type,
            "schema_changes": [c.dataset_name for c in plan.schema_changes],
            "scores_updated": len([d for d in plan.score_diffs if d.action == CurationActionType.UPDATE_PEAK_SCORE]),
            "scores_inserted": len([d for d in plan.score_diffs if d.action == CurationActionType.INSERT_SCORE]),
            "scores_disputed": len([d for d in plan.score_diffs if d.action == CurationActionType.SKIP_DISPUTED_SCORE]),
        }
        esc_payload = self._escape_sql(json.dumps(payload))
        esc_notes = self._escape_sql(plan.summary[:200])

        return (
            f"INSERT INTO curation_history ("
            f"timestamp, model_id, action_type, status, audit_status, audit_confidence, diff_payload, notes"
            f") VALUES ("
            f"'{now_str}', '{esc_id}', '{action_type}', '{status}', '{audit_status}', {confidence}, '{esc_payload}', '{esc_notes}'"
            f");"
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _escape_sql(self, text: Optional[str]) -> str:
        """Sanitizes single quotes for SQL string literals."""
        if text is None:
            return ""
        return str(text).replace("'", "''")

    def _escape_ident(self, ident: Optional[str]) -> str:
        """Sanitizes double quotes and control characters for SQL identifiers (tables and columns)."""
        if ident is None:
            return ""
        s = unicodedata.normalize("NFKC", str(ident))
        s = s.replace("\x00", "").replace(";", "").replace("\n", " ").replace("\r", " ").strip()
        return s.replace('"', '""')

    def _build_executive_summary(self, plan: CurationPlan) -> str:
        """Constructs an executive markdown summary of the curation plan."""
        inserted = sum(1 for d in plan.score_diffs if d.action == CurationActionType.INSERT_SCORE)
        updated = sum(1 for d in plan.score_diffs if d.action == CurationActionType.UPDATE_PEAK_SCORE)
        preserved = sum(1 for d in plan.score_diffs if d.action == CurationActionType.PRESERVE_EXISTING_PEAK)
        disputed = sum(1 for d in plan.score_diffs if d.action == CurationActionType.SKIP_DISPUTED_SCORE)
        unconfirmed = sum(1 for d in plan.score_diffs if d.action == CurationActionType.SKIP_UNCONFIRMED_SCORE)

        lines = [
            f"### Curation Plan for `{plan.model_id}` ({plan.audit_status.upper()}, Conf: {plan.audit_confidence:.2f})",
            f"* **Model Action**: `{plan.model_action.value}`",
            f"* **Novel Benchmarks Added**: {len(plan.schema_changes)} ({', '.join(c.dataset_name for c in plan.schema_changes) or 'None'})",
            f"* **Scores**: {inserted} inserted, {updated} updated (peak gain), {preserved} preserved (existing peak >= new), {disputed} disputed (blocked), {unconfirmed} unconfirmed (omitted)",
            f"* **Total Generated SQL Statements**: {len(plan.generated_sql)}",
        ]
        return "\n".join(lines)


def generate_curation_markdown_report(
    plans: List[CurationPlan],
    dry_run: bool = True,
    sql_file_path: Optional[str] = None,
) -> str:
    """Generates an executive markdown report summarizing database curation decisions across all papers."""
    total_papers = len(plans)
    insert_models = sum(1 for p in plans if p.model_action == CurationActionType.INSERT_NEW_MODEL and p.can_commit)
    update_models = sum(1 for p in plans if p.model_action == CurationActionType.UPDATE_MODEL_METADATA and p.can_commit)
    blocked_models = sum(1 for p in plans if not p.can_commit)

    total_schema_changes = sum(len(p.schema_changes) for p in plans)
    novel_benchmarks = list(dict.fromkeys(c.dataset_name for p in plans for c in p.schema_changes))

    inserted_scores = sum(sum(1 for d in p.score_diffs if d.action == CurationActionType.INSERT_SCORE) for p in plans)
    updated_scores = sum(sum(1 for d in p.score_diffs if d.action == CurationActionType.UPDATE_PEAK_SCORE) for p in plans)
    preserved_scores = sum(sum(1 for d in p.score_diffs if d.action == CurationActionType.PRESERVE_EXISTING_PEAK) for p in plans)
    disputed_scores = sum(sum(1 for d in p.score_diffs if d.action == CurationActionType.SKIP_DISPUTED_SCORE) for p in plans)
    unconfirmed_scores = sum(sum(1 for d in p.score_diffs if d.action == CurationActionType.SKIP_UNCONFIRMED_SCORE) for p in plans)

    sql_link = f"[`{sql_file_path}`]({sql_file_path})" if sql_file_path else "`database_curation.sql`"

    lines = [
        "# DatabaseAgent Curation Executive Report",
        "",
        f"> **Execution Mode**: `{'DRY RUN (Preview Only)' if dry_run else 'APPLIED (Committed)'}`  ",
        f"> **Total Papers Evaluated**: {total_papers}  ",
        f"> **SQL Script Export**: {sql_link}  ",
        "",
        "## 1. High-Level Curation Summary",
        "",
        "| Metric | Count | Description |",
        "| :--- | :--- | :--- |",
        f"| **New Models Inserted** | **{insert_models}** | Unindexed architectures added to `models` and initialized in wide tables |",
        f"| **Models Updated** | **{update_models}** | Existing models enriched with new publication or code metadata |",
        f"| **Papers Blocked** | **{blocked_models}** | Rejected or held in staging due to audit status |",
        f"| **Novel Benchmarks Added** | **{len(novel_benchmarks)}** | Public 3D datasets triggering wide-table schema evolution (`{', '.join(novel_benchmarks) or 'None'}`) |",
        f"| **New Benchmark Scores** | **{inserted_scores}** | Newly established baseline or peak evaluations |",
        f"| **Peak Scores Improved** | **{updated_scores}** | Monotonic improvement over existing historical best scores |",
        f"| **Peaks Preserved** | **{preserved_scores}** | Existing historical peak was higher than newly extracted score |",
        f"| **Disputes Blocked** | **{disputed_scores}** | Numeric column kept NULL to guard leaderboard purity |",
        f"| **Unconfirmed Omitted** | **{unconfirmed_scores}** | Single-agent scores below confidence threshold omitted |",
        "",
        "## 2. Paper-by-Paper Curation Log",
        "",
        "| # | Model ID | Audit Status | Conf | Action | Novel Benchmarks | Peak Scores Inserted/Updated | Disputes Blocked |",
        "| :- | :--- | :--- | :- | :--- | :--- | :- | :- |",
    ]

    for idx, p in enumerate(plans, start=1):
        novel_b = ", ".join(c.dataset_name for c in p.schema_changes) or "-"
        peak_cnt = sum(1 for d in p.score_diffs if d.action in (CurationActionType.INSERT_SCORE, CurationActionType.UPDATE_PEAK_SCORE))
        disp_cnt = sum(1 for d in p.score_diffs if d.action == CurationActionType.SKIP_DISPUTED_SCORE)
        act = p.model_action.value if p.can_commit else f"BLOCKED ({p.rejection_reason})"

        lines.append(
            f"| {idx} | **{p.model_id}** | `{p.audit_status}` | {p.audit_confidence * 100:.0f}% | "
            f"`{act}` | {novel_b} | {peak_cnt} | {disp_cnt} |"
        )

    return "\n".join(lines)

