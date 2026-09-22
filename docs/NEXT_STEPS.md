# NEXT STEPS & HANDOFF GUIDE: AgentForGMMIS

> **Document Purpose**: Executive summary and actionable roadmap for resuming development in future sessions without needing to re-read prior conversational histories.  
> **Repository**: `PezzoliLuca/AgentForGMMIS`  
> **Last Updated**: `2026-09-22`  
> **Status**: Steps 1, 2, and 2.5 **Completed & Verified**. Ready for **Step 3 (Auditor & Curation Engine)**.

---

## 1. Where We Stand (Current Project State)

### Completed Milestones
1. **Step 1: Data Layer & Strict Schema Contracts** (`src/agent_gmmis/data/`)
   * Pydantic v2 schemas: `ModelMetadata`, `DatasetMetadata`, `BenchmarkScore`, `InclusionCriteriaCheck`.
   * SQLite Data Access Layer (`DatabaseManager`) validating all 98 survey models and 58 benchmark datasets in `data/database.sqlite`.
2. **Step 2: Information Extraction Engine** (`src/agent_gmmis/extraction/`)
   * Universal `LLMClient` supporting Google AI Studio Free Tier (`gemini-3.5-flash-lite`, 0.00 €), Ollama, and DEIB Mufasa HPC.
   * `PDFParser` with PyMuPDF: Multi-dimensional page scoring, main-body priority (pages $\le 25$), multi-page table continuation bonus, and evaluation figure recognition.
   * `PaperFetcher`: Local persistent PDF caching in `data/cache_papers/` and automatic arXiv version suffix sanitization.
   * `DatasetAliasResolver`: Reverse-indexing 30+ canonical benchmarks and 120+ synonym variants with word-boundary matching.
3. **Step 2.5: 15-Paper Batch Benchmark & Empirical Validation**
   * Benchmarked across 15 real generalist medical segmentation papers.
   * **100% Recall & 0.0000 Dice MAE** achieved across 8 target models (`MEA M-SAM`, `SFR SAM`, `MedSAM-2`, `BioSAM-2`, `FLAP-SAM`, `BrainSegFounder`, `TP-Mamba`, `TotalSegmentator MRI`).
   * **Empirically detected human survey transcription typo** in `LHU-Net` (Table 2 baseline copy-paste error), proving agentic extraction surpasses human precision.
   * Full 16/16 pytest suite passing in <0.4s. Clean git state pushed to `origin/main`.

---

## 2. Immediate Objective: Step 3 — Auditor & Curation Engine ("Il Vigile")

Step 3 introduces the **Actor-Critic pattern** and **Automated Curation**, transitioning the system from a passive table extractor into an autonomous, self-auditing intelligence platform.

### Architecture of Step 3
```
           ┌──────────────────────────────────────────────┐
           │         ExtractionAgent (Actor)              │
           └──────────────────────┬───────────────────────┘
                                  │ (Raw Extracted Data)
                                  ▼
           ┌──────────────────────────────────────────────┐
           │     AuditorAgent ("Il Vigile" / Critic)      │
           │  - Numerical sanity checks ([0, 1], params)  │
           │  - Historical baseline outlier detection     │
           │  - Missing table diagnosis (figures vs text) │
           └──────────────┬───────────────────────────────┘
                          │
          ┌───────────────┴───────────────────────────────┐
          │ (If Missing / Anomalous)                      │ (If Validated)
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

#### 1. Implement `AuditorAgent` (`src/agent_gmmis/auditing/auditor.py`)
* **Sanity Checks**:
  * Verify Dice scores are within $[0.0, 1.0]$.
  * Verify parameter counts match architecture scale (e.g. flag ViT-B claiming 10M params or CNN claiming 1000B).
  * Validate inclusion criteria: reject non-3D, non-radiological (dermoscopy/fundus), or private-dataset papers.
* **Outlier & Anomaly Detection**:
  * Cross-reference extracted scores against historical distribution in `database.sqlite` (e.g., flag 0.99 DSC on pancreas tumors where state-of-the-art is ~0.68).
* **Missing Data Diagnosis**:
  * When a paper has 0 scalar scores (e.g. `BiomedParse`, `Merlin`): classify whether data is in visual figures (bar/radar charts) or supplementary appendix sections.

#### 2. Implement `OutreachAgent` (`src/agent_gmmis/auditing/outreach.py`)
* Direct operationalization of Moglia et al. methodology (*"In cases where the Dice score was only available in graphical form without explicit numerical values, we contacted the corresponding authors"*).
* Automatically extracts corresponding author name, email, and institution from manuscript header.
* Drafts a formal, polite academic inquiry email in English requesting numerical values for the specific benchmark figures.
* Saves email draft for human-in-the-loop review before dispatch.

#### 3. Implement GitHub Repository Inspector (`src/agent_gmmis/auditing/repo_inspector.py`)
* For papers like `BiomedParse` where benchmark tables do not exist in the manuscript:
  * Clone or inspect the public GitHub repository URL extracted by `ExtractionAgent`.
  * Search for benchmark result files (`results.json`, `eval_metrics.csv`, `test_scores.json`).
  * Extract numerical Dice scores directly from code artifacts.

#### 4. Implement `DatabaseAgent` (`src/agent_gmmis/curation/curator.py`)
* Entity resolution via `DatasetAliasResolver`.
* Automatically compute performance improvements:
  $$\Delta\% = \frac{\text{Dice}_{\text{best}} - \text{Dice}_{\text{primary}}}{\text{Dice}_{\text{primary}}} \times 100$$
* Commit atomic SQLite transactions updating:
  * `models`
  * `results_primary`
  * `results_best`
  * `results_primary_comments` (documenting provenance, e.g. prompt regime, composite averaging, repository extraction).

#### 5. Integration Testing & Verification
* Test auditing pipeline on known cases:
  * `LHU-Net` $\rightarrow$ Audit flags survey discrepancy (87.83% vs 87.49%).
  * `BiomedParse` $\rightarrow$ Audit detects missing table $\rightarrow$ Triggers GitHub repo inspector.
  * Synthetic outlier (e.g. Dice = 1.05 or Dice = 0.99 on Pancreas) $\rightarrow$ Audit raises rejection/warning flag.

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

# 2. Run Test Suite (verify baseline 16/16 tests pass)
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
