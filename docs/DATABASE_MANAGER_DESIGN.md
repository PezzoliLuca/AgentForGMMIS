# DATABASE MANAGER AGENT: Architectural Design & Thesis Specification

> **Module**: `DatabaseAgent` (Database Manager Agent)  
> **Package**: `agent_gmmis.database` (`src/agent_gmmis/database/database_agent.py`)  
> **Author**: Luca Pezzoli (`luca2.pezzoli@mail.polimi.it`)  
> **Supervisor**: Andrea Moglia (`andrea.moglia@polimi.it`) — DEIB, Politecnico di Milano  
> **Reference Publication**: Moglia et al., *"Generalist models in medical image segmentation: A survey and performance comparison with task-specific approaches"*, *Information Fusion*, 127 (2026) 103709.  
> **Last Updated**: `2026-10-26`  
> **Language**: All code and technical documentation in English.

---

## 1. Executive Motivation: Why a Database Manager Agent?

### 1.1. The Limitation of Static Literature Surveys
In standard academic literature surveys (including the foundation study by Moglia et al.), the synthesis of empirical findings into a structured database (`database.sqlite`) is carried out manually:
1. Human reviewers read manuscripts and extract metrics into spreadsheets or static tables.
2. Changes to benchmarks or models require manual SQL statements or brittle ad-hoc scripts.
3. Errors of transcription, ambiguous prompt settings, and human copy-paste errors (such as the empirical refuso identified in `LHU-Net`, documented in [`docs/THESIS_DESIGN_LOG.md`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/docs/THESIS_DESIGN_LOG.md#L24-L26)) corrupt historical accuracy without any formal audit trail.
4. Medical AI advances so rapidly that within months of survey publication, newly released 3D foundation models (e.g. MedSAM-2, SAM-Med3D-Turbo) and emerging challenge benchmarks (e.g. AbdomenAtlas, TotalSegmentator v2) render the static database obsolete.

### 1.2. The Thesis Contribution: Autonomous Knowledge Curation
The **Database Manager Agent (`DatabaseAgent`)** serves as the authoritative, transactional closing loop of the AgentForGMMIS multi-agent framework. It transforms the survey from an ephemeral snapshot into an **autonomous, self-evolving knowledge base**.

Rather than acting as a simple CRUD script, the `DatabaseAgent` is an intelligent curator that enforces:
* **Epistemic Gatekeeping**: Rejection of uncertified or rejected audits;
* **Entity Resolution & Deduplication**: Disambiguating new architectures vs preprint-to-journal lifecycle transitions;
* **Dataset Governance & The Epistemic Boundary**: Enforcing the strict distinction between public reproducible benchmarks and private in-house clinical cohorts;
* **Monotonic Peak Improvement**: Preserving historical baselines while conditionally updating best-in-literature frontiers;
* **Wide-Table Schema Evolution**: Dynamically and safely altering SQLite table schemas when legitimate novel 3D public benchmarks emerge;
* **ACID Transactions & Full Provenance Audit Trail**: Documenting every inserted scalar with prompt regimes, dual-agent consensus statistics, and change diffs.

---

## 2. Architectural Position in the Multi-Agent Pipeline

The `DatabaseAgent` sits directly downstream of the quality assurance critic (`AuditorAgent`) and human-in-the-loop delegation modules (`RepoInspector`, `OutreachAgent`).

```
                              ┌────────────────────────┐
                              │  Scientific Paper/PDF  │
                              └───────────┬────────────┘
                                          │
                        ┌─────────────────┴─────────────────┐
                        ▼ (Worker A)                        ▼ (Worker B)
              ┌───────────────────┐               ┌───────────────────┐
              │ ExtractionAgent A │               │ ExtractionAgent B │
              └─────────┬─────────┘               └─────────┬─────────┘
                        │ Extracted Scores A                │ Extracted Scores B
                        └─────────────────┬─────────────────┘
                                          ▼
                      ┌───────────────────────────────────────┐
                      │ AuditorAgent ("Il Vigile" / Arbiter)  │
                      │  - Dual-Agent Consensus Analysis      │
                      │  - Physical & Historical Sanity Checks│
                      └───────────────────┬───────────────────┘
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   │ (If Missing / Needs Repo / Outreach)        │ (If VERIFIED / FLAGGED_WARNING)
                   ▼                                             ▼
         ┌───────────────────┐                 ┌───────────────────────────────────┐
         │ Delegation Layer  │                 │   DatabaseAgent (Database Manager)│
         │ - RepoInspector   │                 │ 1. Gatekeeper & Policy Validation │
         │ - OutreachAgent   │                 │ 2. Entity Resolution & Deduplic.  │
         └─────────┬─────────┘                 │ 3. Private vs Public Dataset Gate │
                   │ (Re-Audited Results)      │ 4. Dynamic Wide-Table Schema Evol.│
                   └──────────────────────────►│ 5. Monotonic Score Curation       │
                                               │ 6. ACID Commit & Audit History    │
                                               └─────────────────┬─────────────────┘
                                                                 ▼
                                               ┌───────────────────────────────────┐
                                               │    GMMIS SQLite Knowledge Base    │
                                               │  - models                         │
                                               │  - datasets                       │
                                               │  - results_primary / results_best │
                                               │  - results_*_comments / sources   │
                                               │  - curation_history (Immutable)   │
                                               └───────────────────────────────────┘
```

---

## 3. Core Functional Responsibilities

### 3.1. Quality Gatekeeping & Ingestion Policy
The `DatabaseAgent` acts as a firewall guarding the relational integrity and scientific truth of `database.sqlite`:
* **Accepted Input Contracts**:
  * Single-Agent: `AuditReport` + `ExtractedPaper` (or `ModelMetadata` + `BenchmarkScore`s).
  * Dual-Agent: `ConsensusAuditReport` (containing `ConsensusMetadata`, `ConsensusScore`s, `ConsensusScreening`).
* **Ingestion Guardrails**:
  * Status `VERIFIED`: Immediate green light for curation.
  * Status `FLAGGED_WARNING`: Ingested if overall certified confidence score $\ge \text{MIN\_CONFIDENCE}$ (default: $0.70$), with warning anomalies recorded in metadata comments.
  * Status `REJECTED`, `NEEDS_OUTREACH`, `NEEDS_REPO_INSPECTION`: Ingestion is strictly blocked. Staged into a pending queue or rejected outright.

### 3.2. Entity Resolution & Deduplication (Model Level)
When a paper is cleared for ingestion, the agent determines whether it represents an unindexed architecture or an evolution of an existing model:
1. **Resolution Pipeline**:
   * **Tier 1 (Exact Identifier)**: Exact match on `ID` (e.g. `MedSAM`, `nnU-Net`).
   * **Tier 2 (Canonical Link / DOI / arXiv ID)**: Matches publication URL (e.g. `arxiv.org/abs/2304.12306` matches arXiv ID `2304.12306`).
   * **Tier 3 (Normalized Title Similarity)**: Levenshtein distance $\le 2$ or token-set similarity $\ge 0.90$ on title and major affiliations.
2. **Lifecycle Transitions (Preprint $\to$ Published)**:
   * Many models initially enter the DB as arXiv preprints.
   * When the peer-reviewed conference or journal version appears (e.g. CVPR, MICCAI, Nature Communications):
     * The agent updates `Last Publication Date`, `Last Publisher`, `Last Publication Link`, `Last Publication BibKey`.
     * Preserves the original `First Publication Date` to maintain chronological priority.
     * Updates code repository links if newly disclosed.

### 3.3. Dataset Governance & The Epistemic Boundary (Public vs. Private)
A critical insight of this thesis is the **Epistemic Boundary between public benchmarks and private clinical cohorts**:
* **The Problem**: In literature, papers often evaluate both on public challenges (e.g. BraTS, AMOS, BTCV) and on private in-house hospital cohorts (e.g. *"Internal Cohort of 45 Pancreatic Cancer scans from Hospital X"*).
* **The Rule**:
  * **Private In-House Cohorts MUST NEVER trigger schema evolution** (i.e. never create a new column in `results_primary` or `results_best`). Doing so would pollute the global benchmark schema with non-reproducible, unshareable datasets that other models cannot evaluate against.
  * Private dataset evaluations are documented strictly in `commentOnPaper` or model notes.
* **Novel Public 3D Radiological Benchmarks**:
  * Must satisfy inclusion criteria: 3D volumetric, radiological (CT, MRI, PET), public access URL or official challenge website, known anatomical targets.
  * When a genuine novel public benchmark is verified (e.g. `AbdomenAtlas`, `TotalSegmentator v2`, `AutoPET-Organ`):
    1. Register dataset metadata into the `datasets` table (13 attributes: Name, Full Name, Modality, Anatomy, Region, URL, Cit Keys, etc.).
    2. Execute safe, transactional **Wide-Table Schema Evolution** across all 5 evaluation tables.

### 3.4. Wide-Table Schema Evolution Pattern
The GMMIS SQLite schema organizes benchmark evaluations in wide pivot tables:
* `results_primary`: Primary fair-comparison scores (REAL).
* `results_best`: Peak performance scores (REAL).
* `results_best_source`: Citation/table provenance references (TEXT).
* `results_primary_comments`: Granular extraction notes and prompt settings (TEXT).
* `results_best_comments`: Granular extraction notes for best scores (TEXT).

When a novel public benchmark $D_{\text{new}}$ is admitted, the `DatabaseAgent` issues atomic DDL statements:
```sql
ALTER TABLE results_primary ADD COLUMN "D_new" REAL DEFAULT NULL;
ALTER TABLE results_best ADD COLUMN "D_new" REAL DEFAULT NULL;
ALTER TABLE results_best_source ADD COLUMN "D_new" TEXT DEFAULT NULL;
ALTER TABLE results_primary_comments ADD COLUMN "D_new" TEXT DEFAULT NULL;
ALTER TABLE results_best_comments ADD COLUMN "D_new" TEXT DEFAULT NULL;
```
All DDL operations are verified against existing table `PRAGMA table_info()` before execution to prevent duplicate column errors.

---

## 4. Score Ingestion & Epistemic Arbitration Policies

### 4.1. Consensus Status Handling Matrix
In dual-agent extraction, scores are categorized by the `AuditorAgent`. The `DatabaseAgent` applies the following formal policies:

| Consensus Status | Condition | Ingestion Policy | Benchmark Column (`results_best`) | Comment / Provenance Annotation |
| :--- | :--- | :--- | :--- | :--- |
| **`CONCORDANT`** | $\|s_A - s_B\| \le 0.01$ | **Standard Ingestion** | `resolved_score` (mean or verified peak) | `"[Concordant Dual-Agent: Gemini+Qwen, Δ=0.0000] Table 1, p. 5"` |
| **`SINGLE_AGENT_UNCONFIRMED`** | Extracted by 1 model only | **Restricted Gated Ingestion** ($\text{Conf} \ge 0.85$) | `extracted_score` if confidence $\ge 0.85$; otherwise skipped | `"[Unconfirmed Single-Agent: Model A, Conf: 0.90] Table 3, p. 8"` |
| **`DISPUTED`** | $\|s_A - s_B\| > 0.01$ | **Conservative Omission (Purity Guard)** | **`NULL` (Unchanged)** | `"[DISPUTE BLOCKED] Model A: 0.820 vs Model B: 0.865 (Δ=0.045 > 0.01)"` logged in comments & history |

#### Rationale for Single-Agent Gated Ingestion:
* Singletons often represent *coverage breadth* (e.g. Qwen unpacking 13 individual organs while Gemini only extracts the summary row).
* If the extracting model possesses high extraction confidence ($\ge 0.85$), the score is admitted with an explicit flag so readers know it was verified by one agent. If confidence $< 0.85$, it is omitted until further confirmation.

#### Rationale for Disputed Score Omission:
* Inserting an unverified, disputed number risks corrupting leaderboards and biasing downstream clinical decision support.
* Preserving the numeric column as `NULL` while fully documenting the dispute in `results_best_comments` maintains both data integrity and transparency.

---

### 4.2. Monotonic Peak Improvement Policy
> **Scientific Design Decision**: *Updates to benchmark records are strictly conditioned on monotonic improvement.*

When an existing model is re-evaluated or updated with new manuscript data:
1. **Peak Score Update Rule**:
   $$\text{Dice}_{\text{best}}^{\text{new}} = \max\left(\text{Dice}_{\text{best}}^{\text{existing}}, \text{Dice}_{\text{extracted}}\right)$$
   * If $\text{Dice}_{\text{extracted}} > \text{Dice}_{\text{best}}^{\text{existing}}$: The score is updated, and the provenance reference (`results_best_source`, `results_best_comments`) is refreshed to point to the new peak achievement.
   * If $\text{Dice}_{\text{extracted}} \le \text{Dice}_{\text{best}}^{\text{existing}}$: The existing best score is preserved. The lower score is documented in comments as an alternative or baseline setting without downgrading the state-of-the-art claim.
2. **Primary Baseline Stability**:
   * `results_primary` captures the standardized, canonical baseline (e.g. standard 1-point interactive prompt or default zero-shot).
   * Once established from the primary publication, it remains stable unless explicitly superseded by an authoritative erratum or official survey update.
3. **Curatorial Mathematics ($\Delta\%$ Calculation)**:
   Whenever both primary and best scores exist for a dataset, the agent computes and stores the relative gain:
   $$\Delta\% = \frac{\text{Dice}_{\text{best}} - \text{Dice}_{\text{primary}}}{\text{Dice}_{\text{primary}}} \times 100$$

> [!NOTE]
> **Thesis Note / Topic for Advisor (Prof. Andrea Moglia)**:
> Confirm whether `results_primary` should ever be updated if a subsequent paper by the same authors releases an improved standardized baseline, or if `results_primary` must strictly reflect the original survey baseline forever.

---

## 5. Transaction Safety, Rollback & Curation History

### 5.1. ACID Transaction Boundary
All database mutations orchestrated by the `DatabaseAgent` execute within an explicit SQLite transaction:
```python
with sqlite_transaction(db_path) as conn:
    # 1. Deduplicate / Insert Model
    # 2. Register Novel Datasets (if any)
    # 3. Alter Schema Columns (if novel datasets)
    # 4. Insert / Update Primary & Best Scores
    # 5. Insert / Update Comments & Sources
    # 6. Record Curation History Log
    # If any error occurs -> AUTOMATIC ROLLBACK
```

### 5.2. Immutable Audit Trail Table (`curation_history`)
To guarantee full reproducibility, every action taken by the `DatabaseAgent` is committed to a new dedicated table: `curation_history`.

```sql
CREATE TABLE IF NOT EXISTS curation_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    model_id TEXT NOT NULL,
    action_type TEXT NOT NULL,  -- 'INSERT_MODEL', 'UPDATE_METADATA', 'SCHEMA_EVOLUTION', 'UPDATE_SCORES'
    status TEXT NOT NULL,       -- 'COMMITTED', 'DRY_RUN', 'REVERTED'
    audit_status TEXT,          -- 'VERIFIED', 'FLAGGED_WARNING'
    audit_confidence REAL,
    diff_payload TEXT,          -- JSON serialized before/after delta
    notes TEXT
);
```

### 5.3. Dry-Run Execution Mode (`dry_run=True`)
The agent supports a non-destructive dry-run mode:
* Analyzes the audit report against current database state.
* Resolves entities and dataset aliases.
* Simulates schema evolution and score updates.
* Returns a structured `CurationPlan` containing:
  - Target Model: Action (`INSERT` vs `UPDATE`)
  - Schema Changes: Columns to add
  - Score Diffs: `{ dataset: { 'old': 0.82, 'new': 0.88, 'status': 'UPDATED_PEAK' } }`
  - Skipped / Disputed Scores: `{ dataset: 'Dispute Δ=0.04 > 0.01' }`
  - Zero mutations written to disk.

---

## 6. Implementation Architecture

### 6.1. File Structure
The database agent is organized under `src/agent_gmmis/database/`:
```
src/agent_gmmis/database/
├── __init__.py               # Exports DatabaseAgent, CurationPlan, CurationResult
├── schemas.py                # Pydantic v2 schemas for CurationPlan, ScoreUpdateAction, CurationHistoryEntry
└── database_agent.py         # DatabaseAgent core implementation
```
Supported by extensions in [`src/agent_gmmis/data/db_manager.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/data/db_manager.py) (transaction management, schema column inspection, safe DDL alteration).

### 6.2. Public API Contract of `DatabaseAgent`
```python
class DatabaseAgent:
    def __init__(self, db_manager: DatabaseManager, single_agent_min_conf: float = 0.85):
        ...

    def curate(
        self,
        audit_report: Union[AuditReport, ConsensusAuditReport],
        model_metadata: Optional[ModelMetadata] = None,
        dry_run: bool = False,
    ) -> CurationResult:
        """
        Executes atomic curation of audited extraction into GMMIS SQLite database.
        Returns detailed CurationResult detailing all changes and actions taken.
        """
        ...
```

---

## 7. Verification & Empirical Testing Matrix
To ensure 100% test coverage and scientific robustness:
1. **`test_curation_gatekeeping`**: Verifies that `REJECTED` and `NEEDS_OUTREACH` reports are strictly rejected.
2. **`test_model_entity_resolution`**: Tests exact ID matching, arXiv link matching, and preprint-to-journal metadata updates.
3. **`test_private_dataset_isolation`**: Verifies that private in-house cohorts are excluded from schema columns and stored solely in comments.
4. **`test_dynamic_schema_evolution`**: Tests transactional `ALTER TABLE` addition of a valid novel public 3D benchmark across all 5 tables.
5. **`test_consensus_score_policies`**:
   - `CONCORDANT` $\to$ committed cleanly.
   - `SINGLE_AGENT_UNCONFIRMED` ($\text{Conf} \ge 0.85$) $\to$ committed with provenance note.
   - `SINGLE_AGENT_UNCONFIRMED` ($\text{Conf} < 0.85$) $\to$ skipped.
   - `DISPUTED` $\to$ numeric column kept `NULL`, dispute logged to comments.
6. **`test_monotonic_peak_improvement`**:
   - New score $>$ Old score $\to$ updated.
   - New score $\le$ Old score $\to$ old score preserved.
7. **`test_dry_run_and_transaction_rollback`**:
   - `dry_run=True` produces complete plan with zero disk changes.
   - Artificial error triggers complete transaction rollback with pristine database state.

---

## 8. Empirical Validation: 40-Paper Benchmark (Job 5461) & Discovered Benchmarks

### 8.1. End-to-End Benchmark Execution
On 2026-10-26, the complete multi-agent pipeline (`ExtractionAgent` A & B, `AuditorAgent`, `RepoInspector`, `OutreachAgent`, and `DatabaseAgent`) was validated across **40 literature manuscripts** in a single autonomous execution (`Job 5461` / `Run 20261026_142312_mufasa_dual_local_40`) on the DEIB Mufasa cluster:
* **Hardware**: NVIDIA A100 GPU (`3g.20gb:1` MIG partition), running `Qwen2.5-7B-Instruct-AWQ` on port 8000 and `Meta-Llama-3.1-8B-Instruct-AWQ` on port 8001.
* **Throughput**: 38 out of 40 papers completed successfully (95.0% completion rate) in 62 minutes (~98s per paper).
* **Artifact Generation**: Full SQL curation transaction script (`database_curation.sql`, 289 KB, 4,146 statements) and structured executive report (`database_curation_report.md`).

### 8.2. Empirical Curation Statistics

| Metric | Result | Operational Meaning |
| :--- | :---: | :--- |
| **New Models Inserted** | **6** | Added `MedSAM / IMIS-Net`, `SAT-Pro`, `Disruptive Autoencoders`, `PCT-Net`, `Diff-UNet`, `MedSegDiff` to `models`. |
| **Existing Models Updated** | **28** | Enriched with newly disclosed parameters, official GitHub URLs, and publication metadata. |
| **Epistemic Gatekeeper Blocks** | **4** | Blocked `SAM3D` (0.47), `MA-SAM` (0.45), `Hermes` (0.52), and `GVSL` (0.52) due to confidence $<0.60$. |
| **Monotonic Peak Improvements** | **62** | Peak scores in `results_best` updated when new score strictly exceeded prior recorded best. |
| **Historical Peaks Preserved** | **242** | Alternative prompt regimes or ablation scores logged to comments without degrading state of the art. |
| **Consensus Disputes Blocked** | **38** | Set to `NULL` in benchmark numeric columns to preserve leaderboard purity; documented in comments. |
| **Unconfirmed Scores Omitted** | **432** | Single-agent extractions below confidence threshold ($0.85$) safely omitted. |
| **Novel Benchmarks Identified** | **133** | Public challenge datasets and splits triggering synchronized wide-table DDL alteration. |

### 8.3. Wide-Table Schema Evolution Mechanism
When `DatabaseAgent` certifies a novel model or a public benchmark dataset that was not indexed in Moglia et al.'s original survey, it dynamically issues atomic DDL operations synchronized across all 5 wide tables:
```sql
ALTER TABLE results_primary ADD COLUMN `<dataset_name>` REAL;
ALTER TABLE results_best ADD COLUMN `<dataset_name>` REAL;
ALTER TABLE results_best_source ADD COLUMN `<dataset_name>` TEXT;
ALTER TABLE results_primary_comments ADD COLUMN `<dataset_name>` TEXT;
ALTER TABLE results_best_comments ADD COLUMN `<dataset_name>` TEXT;
```
This guarantees that any newly discovered public benchmark is immediately accessible for:
1. Canonical survey baseline comparisons (`results_primary`);
2. State-of-the-art peak performance tracking (`results_best`);
3. Exact citation and prompt-setting provenance (`results_best_source`);
4. Qualitative annotations and dispute histories (`results_*_comments`).

### 8.4. Taxonomy of Discovered Benchmarks: What Moglia et al. Left Out
The original human survey by Moglia et al. focused on a fixed set of 33 canonical benchmark datasets. The autonomous multi-agent system identified **133 novel benchmark identifiers** in the literature papers that were not in those 33 columns.
These fall into three distinct scientific categories:

1. **Category 1: Genuine Omitted Public Challenge Benchmarks (High Academic Value)**:
   Real, publicly accessible segmentation benchmarks and challenges evaluated in literature papers but omitted from the survey's 33 canonical columns:
   - `IMed-361M`: High-impact multi-task benchmark evaluated in `IMIS-Net` / `MedSAM`.
   - `BraTS2020`: Dedicated challenge release with distinct validation splits evaluated in `Diff-UNet` (separate from BraTS 2021).
   - `ULS23`: Universal Lesion Segmentation 2023 challenge evaluated in `SegVol`.
   - `CAS2023` & `SEG.A.2023`: Coronary Artery Segmentation challenge benchmarks evaluated in `SAM-Med3D`.
   - `UKB Data`: UK Biobank brain MRI cohort evaluated in `BrainSegFounder`.
   - `MICCAI 2015 Head-Neck`: Established head-and-neck organ segmentation challenge evaluated in `PCT-Net`.
   - `ATLAS` & `REFUGE`: Stroke lesion (ATLAS v1/v2) and retinal glaucoma (REFUGE) benchmarks evaluated in `One-Prompt`.
   - `RAVIR`: Retinal Artery/Vein Segmentation evaluated in `NexToU`.
   - `Decathlon-Lungs`: Medical Segmentation Decathlon Task 06 (Lung) evaluated in `UNETR++`.
   - `JHH`: Johns Hopkins Hospital multi-organ clinical cohort evaluated in `CLIP-Driven Universal Model`.
   - `Fed-IXI`: Federated learning benchmark split on IXI evaluated in `FLAP-SAM`.

2. **Category 2: Organ/Class-Level Sub-Tasks Extracted as Dataset Names (Granularity Divergence)**:
   Papers reporting evaluations per anatomical organ rather than as an aggregate challenge score:
   - `MultiTalent`: Extracted individual anatomical structures as separate benchmark columns (`Aorta`, `Bladder`, `Bronchies`, `Duodenum`, `Esophagus`, `Heart`, `Kidney`, `Liver`, `Pancreas`, `Rectum`, `Spleen`, `Stomach`), representing sub-classes of BTCV / TotalSegmentator.
   - `3DSAM-adapter`: Extracted organ tumor subsets (`Colon Cancer`, `Kidney Tumor`, `Liver Tumor`, `Pancreas Tumor`) representing MSD / KiTS / LiTS sub-tasks.
   - `UniSeg`: Extracted individual MSD task targets (`Colon`, `HepaV`, `Kidney`, `Liver`, `Lung`, `Pancreas`, `Prostate`, `Spleen`).
   - `MoME`: Extracted MRI sequence splits (`DWI`, `FLAIR`, `T1`, `T1ce`, `T2`).

3. **Category 3: Modality-Level Zero-Shot Aggregations**:
   Papers reporting macro-level zero-shot segmentation splits across imaging modalities:
   - `SAM-Med2D`: Extracted macro-modality columns (`CT`, `MR`, `PET`, `Ultrasound`, `X-ray`, `Dermoscopy`, `Endoscopy`, `Fundus`, `Histopathology`, `Microscopy`).
   - `BioSAM-2`: Extracted domain subsets (`Abdominal CT scans`, `Abdominal MRI scans`, `Endoscopy images`, `Microscopy images`).
   - `TotalSegmentator MRI`: Extracted broad cohort splits (`CT Test Set`, `MRI test set`).

### 8.5. Advisor Alignment Agenda (Discussion with Prof. Andrea Moglia)
To prepare for the thesis consultation with Prof. Andrea Moglia:
1. **Adoption of Category 1 Challenges**: Confirm whether verified public challenge benchmarks (`IMed-361M`, `BraTS2020`, `ULS23`, `CAS2023`, `ATLAS`, `RAVIR`, `JHH`) should be permanently adopted as official columns in the GMMIS database schema.
2. **Granularity Harmonization for Category 2**: Determine whether organ-level breakdowns (e.g. `MultiTalent`, `UniSeg`) should:
   - Be dynamically mapped and averaged into their parent challenge (e.g. averaging organ scores into BTCV); or
   - Be maintained as dedicated organ sub-columns in an auxiliary `results_organ_breakdown` table.
3. **Storage Strategy for Category 3**: Determine whether modality zero-shot splits (e.g. `SAM-Med2D`) belong in wide-table benchmark columns or in qualitative model metadata notes (`commentOnPaper`).
4. **Primary vs Peak Monotonic Policy**: Confirm that `results_primary` serves as the permanent historical anchor to the original survey, while `results_best` autonomously tracks monotonic peak improvements.

