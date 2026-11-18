# SCOUT AGENT DESIGN & METHODOLOGY (`src/agent_gmmis/scouting/`)

> **Milestone**: Step 4 — Autonomous Literature Discovery & Epistemic Screening (`ScoutAgent`)  
> **Package**: [`src/agent_gmmis/scouting/`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/scouting/)  
> **Standalone Scripts**: [`scripts/run_scout.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/scripts/run_scout.py), [`scripts/evaluate_scout.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/scripts/evaluate_scout.py)  
> **Gold-Standard Benchmark**: [`data/benchmarks/scout_abstract_benchmark.json`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/data/benchmarks/scout_abstract_benchmark.json)

---

## 1. Methodological Rationale: Why a Hybrid PRISMA-Agentic Architecture?

A core design question in automating a Living Systematic Review (Moglia et al., *Information Fusion* 2026) is whether literature discovery should rely on **deterministic boolean queries** or **autonomous LLM-driven exploration**.

In medical and bioengineering systematic reviews, relying exclusively on one or the other introduces severe failure modes:
1. **Why Pure Unstructured LLM Web Browsing Fails**:
   * An LLM browsing the web ad-hoc lacks **PRISMA reproducibility**: running the same agent twice yields different paper sets, violating medical systematic review standards.
   * Web search engines rank by popularity rather than recency or domain completeness, missing newly uploaded arXiv preprints.
2. **Why Static Deterministic Queries Alone Become Obsolete (*Vocabulary Drift*)**:
   * Scientific terminology in medical AI evolves rapidly: from *3D CNN / U-Net* (2016–2020) $\rightarrow$ *Vision Transformers / Swin* (2021–2022) $\rightarrow$ *SAM / Promptable Foundation Models* (2023–2024) $\rightarrow$ *SAM-2, 3D State-Space Mamba, Vision-Language Open-Vocabulary CT/MRI* (2024–2026).
   * Static queries hardcoded today cannot anticipate novel architectural paradigms or newly named competing models mentioned inside recent abstracts.

### The Solution: 3-Pillar Hybrid Discovery
The `ScoutAgent` combines **deterministic scientific reproducibility** with **agentic LLM exploration**:

| Discovery Pillar | Mechanism | Scientific Role |
| :--- | :--- | :--- |
| **1. Systematic PRISMA Backbone** | Deterministic boolean queries across `arXiv`, `OpenAlex`, `EuropePMC/PubMed`, and `Google Scholar` | Guarantees 100% reproducible baseline coverage across known 3D radiological modalities and canonical benchmarks. |
| **2. Forward Citation Snowballing** | Queries `OpenAlex` (`cites:{work_id}`) for papers citing foundational seeds (*MedSAM*, *SAM-Med3D*, *TotalSegmentator*, *VISTA3D*, *Moglia et al.*) | Captures any novel 3D segmentation model that compares itself against core baselines, regardless of the keywords used in its title. |
| **3. Agentic Query Planner & Follow-the-Lead Explorer** | 1. **Pre-Harvest (`ScoutQueryPlan`)**: LLM inspects recent models in `database.sqlite` and formulates dynamic search queries.<br>2. **Post-Triage (`discovered_external_leads`)**: While reading abstracts, the LLM extracts names of unknown competing 3D models/benchmarks and triggers targeted follow-up searches. | Adapts autonomously to emerging terminology and hunts down newly referenced models discovered inside abstracts. |

---

## 2. The 4-Stage Filtering Funnel

To respect the **0.00 € Zero-Cost Mandate** and avoid downloading hundreds of irrelevant PDFs or wasting GPU/API tokens on out-of-scope papers, `ScoutAgent` processes candidates through four progressive stages:

```
[Stage 1: Multi-Source Harvesting (arXiv + OpenAlex + EuropePMC + Google Scholar + Snowballing)]
       │
       ▼
[Stage 2: Cross-Source Deduplication, SQLite Lifecycle Check & Lexical Pre-Filter (Zero-LLM)]
       │  ├─ Match in database.sqlite (same source) ────────────► SKIPPED (ALREADY_IN_DB)
       │  ├─ Match in database.sqlite (arXiv -> Journal DOI) ───► PROMOTED (LIFECYCLE_UPGRADE)
       │  └─ Survey/Review title or 2D-only microscopy/retina ──► REJECTED (LEXICAL_REJECTED)
       │
       ▼
[Stage 3: Abstract-Level LLM Triage (3-State Epistemic Logic + Canonical Dataset Rescue)]
       │  ├─ Explicitly 2D-only / Non-Seg / Private Cohort ─────► REJECTED
       │  ├─ Foundation / Promptable / VLM / Universal 3D ──────► PROMOTED (TIER_1_GENERALIST)
       │  └─ Task-Specific 3D Architecture on Public Benchmarks ► PROMOTED (TIER_2_BENCHMARK_SOTA)
       │
       ▼
[Stage 4: Open-Access PDF Resolver & Priority Queue]
          └─ Resolves direct PDF (arXiv / PMC / Preprint Twin) + writes JSON & Markdown Reports
```

---

## 3. Solving the "Incomplete Abstract" Paradox: Three-State Epistemic Logic

In Stage 3, the `ScoutAgent` evaluates only the **Title, Venue, and Abstract** (without downloading the PDF).
* **The Challenge**: Authors frequently omit the word *"Dice"* or fail to list all evaluated datasets in a 200-word abstract, deferring those details to Table 1–3 in the PDF body. Applying the strict post-PDF [`AuditorAgent`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/auditing/auditor.py) rules (`has_dice_metric == True`) at the abstract stage would cause severe **False Negatives**.
* **The Solution ([`TriStateCriterion`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/scouting/schemas.py#L19-L28))**:
  Each inclusion dimension (`is_3d_volumetric`, `is_radiological`, `uses_public_benchmarks`) is evaluated in three states:
  1. **`YES`**: Explicitly confirmed in the title or abstract.
  2. **`IMPLICIT_UNKNOWN`**: Not explicitly detailed in the abstract, but compatible with a 3D radiological segmentation model. **Promoted to PDF extraction** so `ExtractionAgent` and `AuditorAgent` can inspect the actual tables.
  3. **`NO`**: Explicit violation stated in the abstract (e.g., restricted exclusively to 2D retinal fundus, 2D histopathology WSI, 2D colonoscopy frames, non-segmentation image synthesis, or exclusively private hospital cohorts). **Immediately rejected**.

### Canonical 3D Benchmark Rescue Rule
If an abstract explicitly mentions any canonical 3D volumetric radiological benchmark resolved by [`DatasetAliasResolver`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/alias_resolver.py) (e.g., `BraTS`, `AMOS`, `BTCV`, `TotalSegmentator`, `FLARE`, `KiTS`, `LiTS`, `MSD`), [`ScoutAgent.screen_candidate_stage3()`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/scouting/scout_agent.py#L225-L250) deterministically overrides any false-negative `is_3d_volumetric = NO` or `is_radiological = NO` flag from the LLM.

---

## 4. Preprint-to-Journal Lifecycle Tracking (`LIFECYCLE_UPGRADE`)

Out of the 98 models in [`data/database.sqlite`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/data/database.sqlite), **85 (86.7%)** were initially indexed as `arXiv` preprints. When those preprints are subsequently accepted at peer-reviewed venues (*IEEE TMI*, *Medical Image Analysis*, *MICCAI*, *Nature npj Digital Medicine*), a naive scout would either:
* Ignore them completely (leaving the database with outdated preprint metadata), or
* Insert them as duplicate models.

[`ScoutDeduplicator.check_against_database()`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/scouting/deduplicator.py#L168-L226) detects when a newly harvested paper matches an existing `arXiv`-only model in `database.sqlite` (by title similarity $\ge 0.85$ or shared arXiv ID) but carries a **new peer-reviewed journal/conference DOI and venue**, classifying it as **`LIFECYCLE_UPGRADE`** so [`DatabaseAgent`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/database/database_agent.py) can update `Last Publisher`, `Last Publication Date`, and `Last Publication Link`.

---

## 5. Standalone Testing & Calibration Suite

To validate and refine `ScoutAgent` independently of the downstream PDF extraction pipeline:

1. **Gold-Standard 40-Abstract Benchmark ([`scripts/evaluate_scout.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/scripts/evaluate_scout.py))**:
   * Evaluates [`data/benchmarks/scout_abstract_benchmark.json`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/data/benchmarks/scout_abstract_benchmark.json) containing:
     * **20 True Positives**: 12 `TIER_1_GENERALIST` models (*MedSAM2*, *SAM-Med3D*, *VISTA3D*, *SegAnyPET*, *SAT*, *OpenVocabCT*, *ProtoSAM-3D*, *SegVol*, *Prism*, *BiomedParse*, *FastSAM3D*, *MedLSAM*) and 8 `TIER_2_BENCHMARK_SOTA` models (*STU-Net*, *LHU-Net*, *U-Mamba*, *Swin UNETR*, *UNETR++*, *Diff-UNet*, *SegResNet*, *Slim UNETR*).
     * **20 Hard Negatives**: 5 2D-only SAM adaptations (*Polyp-SAM*, *PathoSAM-WSI*, *RetinaSAM*, *SkinSAM*, *CellSAM*), 5 Survey/Review papers, 5 3D Non-Segmentation papers (Diffusion synthesis, VLM report generation, registration, super-resolution), and 5 Private-Only Hospital Cohort papers.
   * Computes **Precision, Recall (Sensitivity), Specificity, F1-Score, Stage 2 Zero-LLM Rejection Count, and Tier 1 vs Tier 2 Classification Accuracy**.

### 5.1. Empirical Cross-Platform Comparison: Cloud Gemini vs. DEIB Mufasa HPC (`Job 5831`)

Both engines were evaluated across all 40 abstracts (`20 TP + 20 Hard TN`):

| Metric | Google Cloud (`gemini-3.1-flash-lite`) | DEIB Mufasa HPC (`Qwen2.5-14B-Instruct-AWQ`, Job 5831) |
| :--- | :---: | :---: |
| **Pure LLM Inclusion Recall (`TP / 20`)** | **100.00%** (`20/20`) | **100.00%** (`20/20`) |
| **Pure LLM Inclusion Specificity (`TN / 20`)** | **100.00%** (`20/20`) | **100.00%** (`20/20`) |
| **Pure LLM Inclusion Precision** | **100.00%** | **100.00%** |
| **Pure LLM Inclusion F1-Score** | **100.00%** | **100.00%** |
| **Stage 2 Zero-LLM Hard Negatives Caught** | `15/20` (75.0%) | `15/20` (75.0%) |
| **Full Funnel (S2 + S3) F1-Score** | **100.00%** | **100.00%** |
| **Tier 1 (`GENERALIST`) vs Tier 2 (`SOTA`) Acc.** | **95.00%** (`19/20`)* | **100.00%** (`20/20`) |
| **Total Runtime (40 Abstracts)** | `254.59s` (`6.36s/abstract`) | **`219.92s` (`5.50s/abstract`)** |
| **Report Artifact** | [`eval_40_gemini_gemini-3.1-flash-lite.json`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/logs/scout_runs/eval_40_gemini_gemini-3.1-flash-lite.json) | [`eval_40_mufasa_Qwen_Qwen2.5-14B-Instruct-AWQ.json`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/logs/scout_runs/eval_40_mufasa_Qwen_Qwen2.5-14B-Instruct-AWQ.json) |

*\*Note: On `TP_13_STUNet` ("STU-Net: Scalable and Transferable Medical Image Segmentation Models Empowered by Large-Scale Supervised Pre-training", 1.4B parameters evaluated across 14 public 3D benchmarks), Gemini classified it as `TIER_1_GENERALIST` due to its large-scale foundation pre-training, whereas `Qwen2.5-14B-Instruct-AWQ` matched the exact benchmark label `TIER_2_BENCHMARK_SOTA` (`20/20`).*

2. **Retrospective Time-Machine Backtest (`scripts/run_scout.py --backtest-2025` & April–May 2025 Stress Test)**:
   * Masks models published in 2025 from `database.sqlite` (`ignore_db_models_after_date="2025-01-01"`) and queries live APIs for `2025-04-01` $\rightarrow$ `2025-05-31` (where the survey contains 3 ground-truth models: `MedSAM2`, `TAGS`, `CDPDNet` hidden among 326 unique wild papers).
   * **4 Real-World Failure Modes Discovered & Resolved via Wild Backtesting**:
     1. *Elsevier/Springer DOI False arXiv Regex*: Publisher DOIs such as `10.1016/j.bspc.2025.104340` contain the substring `2025.10434`, which naive arXiv regexes misparse as a fake `YYMM.NNNNN` arXiv identifier (`https://arxiv.org/pdf/2025.10434.pdf`). Resolved in [`_extract_arxiv_id()`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/scouting/harvesters.py#L55-L78) by enforcing valid month bounds (`01`..`12`) and requiring an explicit `arxiv` prefix whenever a `/` or `10.` DOI prefix is present.
     2. *Google Scholar Year-Only Temporal Drift*: Google Scholar metadata snippets only report the publication year (`2025`), causing November 2025 preprints (`arXiv:2511.19046`, *MedSAM3*) to leak into an April–May 2025 window as `2025-01-01`. Resolved via [`_infer_date_from_arxiv_id()`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/scouting/harvesters.py#L81-L88), which reconstructs `20YY-MM-01` directly from the canonical `YYMM` arXiv prefix.
     3. *Off-the-Shelf Baseline Clinical Applications & Benchmarking Odysseys*: Wild literature includes clinical studies that merely apply an unmodified off-the-shelf `3D nnU-Net` or `SAM 2` to an organ cohort, as well as pure benchmarking studies (*"A Benchmarking Odyssey..."*). Resolved by adding comparative/benchmarking study patterns to Stage 2 lexical filters and enforcing an off-the-shelf baseline gatekeeper in [`ScoutAgent.screen_candidate_stage3()`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/scouting/scout_agent.py#L293-L313).
     4. *Implicit-Modality Title Pre-Ranking (`CDPDNet`)*: In its real arXiv abstract (`2505.18958`), `CDPDNet` never mentions the words `"3D"`, `"CT"`, `"MRI"`, or `"Dice"`. Adding the `<ModelAcronym>:` title prefix bonus and `"text guidance"` / `"hybrid vision encoder"` signals to [`ScoutDeduplicator.compute_stage2_pre_rank()`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/scouting/deduplicator.py#L332-L377) elevated `CDPDNet` into the top 25 candidates, where Stage 3's `IMPLICIT_UNKNOWN` logic promoted it as `TIER_1_GENERALIST`.
3. **Post-Survey Living Discovery (`scripts/run_scout.py --since 2025-06-01`)**:
   * Harvests and screens new candidates published after the Moglia et al. survey cutoff, generating `scout_report.md`, `scout_cot_trace.md`, and `scout_report.json` in `logs/scout_runs/`.

---

## 6. Structured Chain-of-Thought (CoT) Audit Logging & 3-Stage Defense-in-Depth Verification

### 6.1. Schema-Embedded 3-Step Chain-of-Thought (`ScoutThoughtProcess`)
To guarantee full explainability and scientific auditability across every screening decision, [`ScoutScreeningResult`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/scouting/schemas.py#L140-L157) embeds a mandatory pre-decision [`ScoutThoughtProcess`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/scouting/schemas.py#L105-L137) object evaluated *before* any boolean or tri-state labels are emitted:
1. **`task_and_novelty_analysis`**: Evaluates whether the paper introduces a genuinely novel medical image segmentation architecture vs. a survey, non-segmentation task (synthesis/registration/report-gen), or an off-the-shelf baseline clinical application (`nnU-Net`, `SAM 2`).
2. **`dimensionality_and_modality_analysis`**: Analyzes explicit or implicit 3D volumetric radiological cues (CT, MRI, PET, 3D Ultrasound) vs. 2D-only optical/microscopy domains.
3. **`benchmark_and_paradigm_analysis`**: Identifies public benchmarks and justifies `TIER_1_GENERALIST` vs. `TIER_2_BENCHMARK_SOTA`.

Every `ScoutAgent.save_report()` invocation automatically persists **`scout_cot_trace.md`** (e.g., [`logs/scout_runs/scout_20261104_001220/scout_cot_trace.md`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/logs/scout_runs/scout_20261104_001220/scout_cot_trace.md)), logging:
* **Stage 0 (`AgenticQueryPlanner` CoT)**: The full strategic `thought_process` and rationale for each generated search query.
* **Stage 3 (Per-Paper CoT Cards)**: Input abstract, 3-step `ScoutThoughtProcess`, `reasoning`, Tri-State matrix, deterministic rescue/guard overrides, and LLM latency for every promoted and rejected candidate.
* **Stage 2 (Lexical Rejection Table)**: Exact exclusion triggers for all deterministically filtered candidates.

### 6.2. 3-Stage PRISMA Defense-in-Depth (`ScoutAgent` $\rightarrow$ `ExtractionAgent` $\rightarrow$ `AuditorAgent`)
Because `ScoutAgent` operates exclusively on **Title + Abstract** using high-recall 3-state epistemic logic (`YES`, `IMPLICIT_UNKNOWN`, `NO`), papers with vague or incomplete abstracts ("stealth traps") are intentionally promoted when criteria are `IMPLICIT_UNKNOWN`. Downstream, the full-text pipeline acts as a strict second and third firewall:
* **Firewall 1 (`ScoutAgent` — Abstract Triage)**: Filters out explicit 2D-only, non-radiological, survey, synthesis, private-cohort, and off-the-shelf baseline papers, while promoting valid 3D models and `IMPLICIT_UNKNOWN` candidates to PDF download.
* **Firewall 2 (`ExtractionAgent` — Full-Text Reading & `ExtractionThoughtProcess` CoT)**: Parses the full PDF manuscript and experimental tables, resolving all `IMPLICIT_UNKNOWN` fields into strict full-text booleans (`InclusionCriteriaCheck`: `is_3d`, `is_radiological`, `is_public_dataset`, `has_dice_metric`).
* **Firewall 3 (`AuditorAgent` — "Il Vigile" Critic)**: Audits `ExtractionAgent`'s output (`auditor.py`) and immediately sets `AuditStatus.REJECTED` if any full-text inclusion criterion fails or if baseline model confusion is detected, blocking [`DatabaseAgent`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/database/database_agent.py) from ingesting the paper into `database.sqlite`.

This 3-stage filter chain is empirically verified by [`scripts/test_scout_extractor_filter_chain.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/scripts/test_scout_extractor_filter_chain.py) (full CoT trace saved at [`logs/scout_runs/defense_in_depth_chain_trace.md`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/logs/scout_runs/defense_in_depth_chain_trace.md)):

| Scenario | Stage 3 `ScoutAgent` (Abstract Only) | Full-Text `ExtractionAgent` (`InclusionCriteriaCheck`) | Critic `AuditorAgent` Verdict | Final Database Action |
| :--- | :---: | :---: | :---: | :---: |
| **1. Real 2025 Survey Model (`MedSAM2` — Real 9MB PDF)** | **PROMOTE** (`TIER_1_GENERALIST`) | `is_included=True` (`10` scores extracted) | **`needs_repo_inspection` (ACCEPTED)** | ✅ **INGEST INTO SQLITE** |
| **2. Stealth Trap A: Ambiguous 3D MRI Abstract $\rightarrow$ Private Cohort Only in Full Text** | **PROMOTE** (`Public=IMPLICIT_UNKNOWN`) | `is_included=False` (`is_public_dataset=False`) | **`rejected`** (`Private cohort only`) | 🛑 **BLOCKED FROM DB** |
| **3. Stealth Trap B: Vague Generalist Abstract (`CDPDNet`-style) $\rightarrow$ 2D Retina/Dermoscopy in Full Text** | **PROMOTE** (`3D=IMPLICIT_UNKNOWN`) | `is_included=False` (`is_3d=False, is_radiological=False`) | **`rejected`** (`2D / Non-radiological`) | 🛑 **BLOCKED FROM DB** |
| **4. Stealth Trap C: 3D Abdominal CT Abstract $\rightarrow$ Off-the-Shelf `nnU-Net` & HD95 Only (Zero Dice)** | **PROMOTE** (`TIER_2_BENCHMARK_SOTA`) | `is_included=False` (`has_dice_metric=False`) | **`rejected`** (`No DSC reported`) | 🛑 **BLOCKED FROM DB** |


