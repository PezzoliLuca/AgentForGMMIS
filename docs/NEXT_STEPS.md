# NEXT STEPS & HANDOFF GUIDE: AgentForGMMIS

> **Document Purpose**: Executive summary and actionable roadmap for resuming development in future sessions without needing to re-read prior conversational histories.  
> **Repository**: `PezzoliLuca/AgentForGMMIS`  
> **Last Updated**: `2026-10-26`  
> **Status**: Steps 1, 2, 2.5, **Phase 3.1 (AuditorAgent / 'Il Vigile')**, **Phase 3.1.5 (Dual-Agent Consensus & Epistemic Arbitration Engine Refinement)**, **Phase 3.2 Parts 1 & 2 (`RepoInspector` & `OutreachAgent`)**, **Phase 3.3 (`DatabaseAgent` / Curator Engine)**, and **40-Paper Comparative Benchmarking (Dual-Local Job 5461 vs Hybrid Job 5471)** Completed & Empirically Validated (93/93 tests passing). Current Priority: **Advisor Alignment Meeting with Prof. Andrea Moglia & Transition to Step 4 (`ScoutAgent`)**.

---

## 1. Where We Stand (Current Project State)


### Completed Milestones
1. **Step 1: Data Layer & Strict Schema Contracts** (`src/agent_gmmis/data/`)
   * Pydantic v2 schemas: `ModelMetadata`, `DatasetMetadata`, `BenchmarkScore`, `InclusionCriteriaCheck`.
   * SQLite Data Access Layer (`DatabaseManager`) validating all 98 survey models and 58 benchmark datasets in `data/database.sqlite`.
2. **Step 2: Information Extraction Engine** (`src/agent_gmmis/extraction/`)
   * Universal `LLMClient` supporting Google AI Studio Free Tier (`gemini-3.5-flash-lite`, 0.00 €), Ollama, and DEIB Mufasa HPC (`Qwen2.5-14B-Instruct-AWQ` via vLLM).
   * `PDFParser` with PyMuPDF: Multi-dimensional page scoring, main-body priority (pages $\le 25$), multi-page table continuation bonus, and evaluation figure recognition.
   * `PaperFetcher`: Local persistent PDF caching in `data/cache_papers/` and automatic arXiv version suffix sanitization.
   * `DatasetAliasResolver`: Reverse-indexing 30+ canonical benchmarks and 120+ synonym variants with word-boundary matching.
   * `prompts.py`: Standardized prompt rules prioritizing 1-point prompt interactive settings, SOTA comparative tables over internal ablations, and macro-average multi-organ rows.
3. **Step 2.5 & Step 3.2: 30-Paper Cross-Architecture Benchmark & Empirical Validation**
   * Parallel double-run across 30 foundational 3D medical foundation models comparing Google Gemini 3.5 Flash Lite vs Qwen2.5-14B-Instruct-AWQ on DEIB Mufasa HPC (SLURM Job 4913).
   * **14 models achieved identical 0.0000 Dice MAE** across both platforms.
   * **Empirically replicated human survey copy-paste error** in `LHU-Net` across both independent LLM engines (MAE 0.0293).
   * Analyzed Speed vs. Recall tradeoffs: Gemini is 10.1x faster (9.3s vs 94.2s), while Mufasa achieves +7.0% higher recall (73.6% vs 66.6%).
4. **Step 3.1 & Hardening: Auditor & Quality Assurance Critic ("Il Vigile")** (`src/agent_gmmis/auditing/`)
   * **`AuditorAgent` (`auditor.py`)**: Multi-level inspection engine enforcing inclusion screening, architectural parameter bounds (ViT-B/L/H), statistical distribution outlier detection via SQLite historical baselines, and missing data diagnosis.
   * **Epistemological Autonomy**: Completely decoupled deploy mode (`benchmark_mode=False`) from ground truth knowledge; auditing is driven entirely by mathematical bounds, architectural physics, table self-consistency, and Bayesian empirical priors.
   * **Multi-Tier Acronym Compatibility**: Resolved `Med-SA` false rejection via generalized punctuation normalization, stopword-filtered acronym synthesis, and manuscript self-nomination checks (`_is_model_name_compatible_with_paper`).
   * **Dynamic Database Statistics**: Extended `DatabaseManager.get_dataset_statistics()` to dynamically compute historical `min`, `max`, `mean`, and sample count per benchmark.
5. **Phase 3.2 Part 1: GitHub Repository Inspector (`RepoInspector`)** (`src/agent_gmmis/auditing/repo_inspector.py`)
   * Inspects official code repositories for papers lacking scalar tables (`BiomedParse`).
   * Multi-format parsing: Markdown tables (long and wide format), structured JSON artifacts, CSV evaluation logs.
   * Resolves aliases and standardizes scores with `DatasetAliasResolver`.
   * End-to-end `inspect_and_update` workflow: automatically recovers scores and re-audits paper from `NEEDS_REPO_INSPECTION` to `VERIFIED` / `FLAGGED_WARNING`.
   * **Empirically Validated on DEIB Mufasa Cluster (Job 4501)**: `BiomedParse` benchmarked with `Qwen2.5-14B-Instruct-AWQ` recovered from **0/14 (0.0% Recall)** to **12/14 (85.7% Recall)** with a **0.0199 Dice MAE** and 4 exact dataset matches.
   * **41/41 tests passing in <0.70s** with 100% test coverage across all modules.

---

## 2. Immediate Objective: Phase 3.1.5 (`Dual-Agent Consensus & Arbitration`), Phase 3.2 Part 2 (`OutreachAgent`) & Phase 3.3 (`DatabaseAgent`)

With the single-agent Critic (`AuditorAgent`) and `RepoInspector` operational, Step 3 advances to:
1. **Phase 3.1.5**: Extending extraction to **Dual-Agent Parallel Consensus** with distinct models and multi-model arbitration by the Auditor.
2. **Phase 3.2 Part 2**: **Delegation & Resolution Layer (`OutreachAgent`)** for author email drafting.
3. **Phase 3.3**: **`DatabaseAgent`** for atomic SQLite curation and dynamic schema evolution.

### Architecture of Step 3 (Dual-Actor / Arbiter & Delegation Flow)
```
                  ┌────────────────────────┐
                  │   Manuscript / PDF     │
                  └───────────┬────────────┘
                              │
               ┌──────────────┴──────────────┐
               ▼ (Parallel Worker A)         ▼ (Parallel Worker B)
     ┌───────────────────┐         ┌───────────────────┐
     │ ExtractionAgent A │         │ ExtractionAgent B │
     │  (e.g. Model A)   │         │  (e.g. Model B)   │
     └─────────┬─────────┘         └─────────┬─────────┘
               │ Extracted A                 │ Extracted B
               └──────────────┬──────────────┘
                              ▼
            ┌──────────────────────────────────────────────┐
            │     AuditorAgent ("Il Vigile" / Arbiter)     │
            │  - Inter-agent agreement & discrepancy match │
            │  - Categorization: CONCORDANT / DISPUTED /   │
            │    SINGLE_AGENT_UNCONFIRMED                  │
            │  - Numerical sanity checks ([0, 1], params)  │
            │  - Historical baseline outlier detection     │
            │  - Missing table diagnosis (figures vs text) │
            └──────────────┬───────────────────────────────┘
                           │
           ┌───────────────┴───────────────────────────────┐
           │ (If Missing / Needs Repo / Outreach)          │ (If Reconciled & Validated)
           ▼                                               ▼
 ┌─────────────────────────┐                     ┌─────────────────────────┐
 │     Delegation Layer    │                     │      DatabaseAgent      │
 │ 1. GitHub Repo Inspector│                     │  - Harmonize aliases    │
 │    (Parse JSON/CSV logs)│                     │  - Compute Delta %      │
 │ 2. OutreachAgent        │                     │  - Atomic SQLite commit │
 │    (Draft author email) │                     └─────────────────────────┘
 └─────────────────────────┘
```

### Action Items for Step 3 Implementation:

#### 1. Implement `AuditorAgent` (`src/agent_gmmis/auditing/auditor.py`) — **COMPLETED & VALIDATED**
* Sanity checks on physical ranges, parameter scaling, and inclusion criteria.
* Statistical baseline outlier detection against historical SQLite distributions.
* Missing data diagnosis triaging between text tables, visual charts, and GitHub repositories.
* Multi-tier acronym compatibility (`_is_model_name_compatible_with_paper`).

#### 2. Implement GitHub Repository Inspector (`src/agent_gmmis/auditing/repo_inspector.py`) — **COMPLETED & VALIDATED**
* Ingests public repositories when manuscript tables are missing (`BiomedParse`).
* Multi-format parser: Markdown tables (long & wide layouts), structured JSON results, CSV evaluation logs.
* Resolves synonyms to canonical benchmarks via `DatasetAliasResolver`.
* Seamless `inspect_and_update` integration with `AuditorAgent` recovering scores and re-certifying to `VERIFIED`.
* Verified with 9 dedicated unit tests (41/41 tests green across suite).

#### 3. Implement Dual-Agent Consensus Extraction & Epistemic Arbitration Engine — **COMPLETED & VALIDATED**
* **Dynamic Model-Agnostic Parametrization**:
  * Decoupled models from agent logic; supports arbitrary pairings (Gemini + Qwen, Llama + DeepSeek, etc.) via CLI flags: `--model-a`, `--provider-a`, `--model-b`, `--provider-b`.
  * Backward-compatible fallback to single-agent mode when Model B is omitted.
* **`DualExtractionCoordinator` (`src/agent_gmmis/extraction/dual_coordinator.py`)**:
  * Orchestrates two `ExtractionAgent` instances in parallel using `ThreadPoolExecutor(max_workers=2)`.
  * Guaranteed independent extraction preventing anchoring bias.
* **Consensus Schemas (`src/agent_gmmis/auditing/schemas.py`)**:
  * `ConsensusStatus` enum: `CONCORDANT`, `DISPUTED`, `SINGLE_AGENT_UNCONFIRMED`.
  * `ConsensusScore`, `ConsensusScreening`, `ConsensusMetadata`, and polymorphic `ConsensusAuditReport`.
* **Consensus Auditing in `AuditorAgent` (`audit_consensus`)**:
  * Semantic alignment of scores using `DatasetAliasResolver` and macro-average synonym matching (`OVERALL_SYNONYMS`).
  * Tolerance threshold $\epsilon = 0.01$ (1% DSC) for `CONCORDANT` agreement.
  * Discrepancies $> \epsilon$ flagged transparently as `DISPUTED` (preserving both candidate values).
  * **Disentangled Paired IAA from Coverage Granularity**: `concordance_rate` calculated strictly on paired benchmarks (`concordant_count / paired_count`), preventing single-agent coverage differences from artificially deflating agreement metrics.
  * **Bounded & Proportional Confidence Scoring**: Replaced linear deductions with bounded dispute ratio penalties (`min(0.25, dispute_ratio * 0.25)`), protecting certified confidence for models with 0.0000 MAE.
  * **Partition Split Disambiguation**: Enforced `data_budget_or_setting` (e.g. 10% vs 20% labeled) to prevent cross-partition false disputes in semi-supervised benchmarks.
* **CLI Runners & Script Integration**:
  * Updated [`scripts/run_batch_benchmark.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/scripts/run_batch_benchmark.py) and [`scripts/run_extraction.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/scripts/run_extraction.py) with dual-agent support and formatted Consensus Matrix rendering.
* **Test Verification**:
  * 11 dedicated unit tests in `tests/test_consensus_auditing.py` (60/60 total test suite passing in 0.51s).

#### 4. Implement `OutreachAgent` (`src/agent_gmmis/auditing/outreach.py`) — **COMPLETED & VALIDATED**
* **Methodological Foundation**:
  * Direct operationalization of Moglia et al. (*"In cases where the Dice score was only available in graphical form without explicit numerical values, we contacted the corresponding authors"*).
* **Metadata & Contact Extraction**:
  * Hierarchical extraction of corresponding author name, email, and affiliation from front matter (supporting both Nature/Springer footnote asterisks and conference/arXiv inline headers).
  * Graceful fallback when emails are absent, assigning confidence metrics.
* **Target & Figure Triage**:
  * Automatically isolates benchmark datasets lacking scalar values (`dice_score is None`) and captions featuring visual charts (bar plots, boxplots, radar charts).
* **Formal Scholarly Inquiry Drafting**:
  * Generates professional, polite academic correspondence in English citing Politecnico di Milano DEIB, introducing the open-access GMMIS benchmark initiative, referencing specific figures, and requesting scalar Dice scores.
* **Human-in-the-Loop Governance & Multi-Format Storage**:
  * Automatically persists drafts to `data/outreach_drafts/` in both machine-readable JSON (`OutreachDraft` schema) and human-readable Markdown (`.md`) ready for manual review before dispatch.
* **Auditor Integration**:
  * `handle_audit_report()` automatically triages `AuditStatus.NEEDS_OUTREACH` and `MissingDataDiagnosis.MISSING_TABLE_FIGURES_FOUND`.
* **Test Verification**:
  * 8 dedicated unit tests in `tests/test_outreach.py` (57/57 total test suite passing in 0.41s).

#### 5. Implement `DatabaseAgent` (`src/agent_gmmis/database/database_agent.py`) — **COMPLETED & EMPIRICALLY VALIDATED**
* **Entity Resolution & Deduplication**:
  * Check if model already exists (via exact ID, arXiv ID/URL, or fuzzy title similarity) before insertion.
  * Resolve dataset synonyms using `DatasetAliasResolver`.
  * Support model lifecycle evolution (arXiv preprint $\to$ peer-reviewed journal/conference), updating publication records while preserving original priority dates.
* **New Model Ingestion (Production Deploy)**:
  * For certified novel architectures, perform atomic insertion into the `models` table (populating all 19 standardized attributes: backbone, parameters, open-source status, licensing, affiliations).
  * Initialize model records in `results_primary` and `results_best`.
* **New Dataset Ingestion & Dynamic Schema Evolution**:
  * **Private In-House Datasets**: Explicitly distinguish private clinical cohorts (e.g. "Internal Hospital Cohort 50 scans") from public benchmarks. Private datasets are documented in model comments/notes but **never** added as global public benchmark columns.
  * **Novel Public Benchmarks**: For newly emerging public 3D radiological benchmarks (e.g. *AbdomenAtlas*, *TotalSegmentator v2*, *FLARE 2024*):
    * Validate inclusion (3D, radiological, public access URL, known anatomical targets).
    * Atomically insert the benchmark metadata into the `datasets` table (13 attributes: modality, anatomy, sizes, license).
    * Perform safe, transactional schema evolution (`ALTER TABLE results_primary ADD COLUMN ...`, `ALTER TABLE results_best ADD COLUMN ...`, and comment tables) to dynamically accommodate new benchmarks without downtime or data corruption.
* **Score Curation & Epistemic Arbitration Policies**:
  * **`CONCORDANT`**: Ingested directly into primary/best benchmark columns.
  * **`SINGLE_AGENT_UNCONFIRMED`**: Ingested under restricted confidence gating ($\text{Conf} \ge 0.85$) with explicit provenance comment; otherwise skipped.
  * **`DISPUTED`**: Benchmark column left blank/NULL to preserve leaderboard purity; dispute details logged to comments and audit trail.
  * **Monotonic Peak Improvement Policy**: `results_best` updated only if new score exceeds existing peak ($\text{Dice}_{\text{new}} > \text{Dice}_{\text{existing}}$); lower alternative evaluations documented in comments. *(Note: flagged for advisor alignment with Prof. Andrea Moglia)*.
* **Curatorial Mathematics & Provenance Tracking**:
  * Automatically compute performance improvements against primary baselines:
    $$\Delta\% = \frac{\text{Dice}_{\text{best}} - \text{Dice}_{\text{primary}}}{\text{Dice}_{\text{primary}}} \times 100$$
  * Commit atomic SQLite transactions updating:
    * `models`
    * `datasets` (for novel benchmarks)
    * `results_primary`
    * `results_best`
    * `results_primary_comments` (documenting provenance, e.g. prompt regime, composite averaging, repository extraction).
    * `results_best_source`
    * `curation_history` (immutable audit trail table tracking all mutations, diffs, and audit certificates).

* **Empirically Validated on 40-Paper Cluster Benchmark (`Job 5461`)**:
  * 38/40 papers completed successfully (95.0%).
  * 6 new models inserted, 28 updated, 4 blocked by gatekeeper, 62 peak improvements, 242 peaks preserved, 38 disputes blocked, 432 unconfirmed omitted, 133 novel benchmarks detected.
* **Test Suite Status**:
  * 11 unit and integration tests in `tests/test_database_agent.py` (93/93 total tests green in 0.88s).

---

## 2. Immediate Objective: Advisor Consultation (Prof. Andrea Moglia) & ScoutAgent (Step 4)

With `ExtractionAgent`, `AuditorAgent`, `RepoInspector`, `OutreachAgent`, and `DatabaseAgent` fully implemented and validated on the 40-paper benchmark:
1. **Advisor Alignment Meeting with Prof. Andrea Moglia**:
   * Review the **133 novel benchmarks** discovered by the agents across the 38 papers.
   * Review the 3-category taxonomy (Category 1: Genuine public challenges omitted from survey; Category 2: Organ/class breakdowns; Category 3: Modality zero-shot splits).
   * Agree on adoption policies: whether Category 1 benchmarks (`IMed-361M`, `BraTS2020`, `ULS23`, `CAS2023`, `ATLAS`, `RAVIR`) become permanent schema columns, and how Category 2 organ breakdowns should be aggregated.
   * Confirm the monotonic peak policy ($\text{Dice}_{\text{best}} = \max(\text{old}, \text{new})$) while keeping `results_primary` as the immutable survey baseline.
2. **Tri-Continental Sovereign Model Evaluation on Mufasa**:
   * Deploy and benchmark a European sovereign open-weight model (`mistralai/Mistral-7B-Instruct-v0.3` or `casperhansen/codestral-22b-v0.1-awq`) on Mufasa's 20 GB VRAM slice.
   * Benchmark across the 40-paper corpus to complete the tri-continental comparison: USA (Gemini 3.5 Flash) vs China (Qwen 2.5 14B) vs Europe (Mistral/Codestral).
3. **Phase 4 (`ScoutAgent`)**:
   * Continuous discovery of newly published medical segmentation preprints via arXiv and PubMed APIs.
   * Automatic paper fetching and execution of the full multi-agent pipeline.

---

## 3. Subsequent Roadmap (Steps 4 & 5 Preview)

* **Step 4: Autonomous ScoutAgent**
  * Automated periodic harvesting via arXiv API, PubMed API, Semantic Scholar.
  * Automated screening on titles & abstracts against inclusion criteria.
* **Step 5: ConsultantAgent & Decision Support**
  * Clinical reasoning CLI/UI.
  * Multi-objective Pareto optimization (Dice vs VRAM vs Parameters).
  * Automated **Mann-Whitney U tests** for statistical significance across architectures.

---

## 4. Quick-Start Commands for Next Session

To resume work in the next session, run:

```bash
# 1. Activate Virtual Environment
source .venv/bin/activate

# 2. Run Test Suite (verify baseline 60/60 tests pass)
.venv/bin/pytest -v

# 3. Quick 3-Paper Extraction Smoke Test
.venv/bin/python3 scripts/run_batch_benchmark.py --count 3

# 4. View Latest Benchmark Report
cat logs/latest_batch_benchmark.md
```

---

## 5. Architectural & Thesis References
* **Development History**: [`docs/DEVELOPMENT_LOG.md`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/docs/DEVELOPMENT_LOG.md) (All milestone logs and Step 2.5 batch benchmark table).
* **Thesis Defense Gems**: [`docs/THESIS_DESIGN_LOG.md`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/docs/THESIS_DESIGN_LOG.md) (12 specific scientific gems including LHU-Net human typo, TotalSegmentator manual averaging, and Preprint Drift).
* **Core Philosophy**: Zero Cost (0.00 € mandate, Google AI Studio free tier), English documentation, Python 3.13, Pydantic v2 contracts.
