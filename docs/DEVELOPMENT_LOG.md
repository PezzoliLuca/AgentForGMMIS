# DEVELOPMENT LOG: AgentForGMMIS

This document tracks all project milestones, code implementations, schema updates, and test validations across development steps.

---

## [2026-09-22] — Step 1: Data Layer & Strict Schema Contracts

### 1. Objective
Establish the foundational data contracts, Pydantic v2 domain schemas, and SQLite Data Access Layer (DAL) to ensure type safety, data integrity, and strict adherence to the GMMIS benchmark schema.

### 2. Architecture & Design Decisions
* **Packaging & Structure**: Standard modern `src`-layout packaging (`agent-gmmis`) using `pyproject.toml` with setuptools.
* **Typing Standard**: Pydantic v2 schemas used across all data models to strictly validate inputs, types, and boundaries.
* **Score Normalization**: Implemented a smart validator in `BenchmarkScore` that normalizes percentage notation (e.g., $88.75\% \rightarrow 0.8875$) and enforces the $[0.0, 1.0]$ interval.
* **Inclusion/Exclusion Logic**: Formalized Moglia et al.'s scientific screening criteria into `InclusionCriteriaCheck` (3D volume, public dataset, radiological CT/MRI/PET, DSC metric present).
* **Database Access**: Implemented `DatabaseManager` as a clean DAL wrapper around `data/database.sqlite` using `sqlite3.Row` factory.

### 3. Files Created & Modified
* **Repository Setup**:
  * [`.gitignore`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/.gitignore): Ignores Python caches (`__pycache__`), virtual environments (`.venv`), IDE files (`.DS_Store`), and SLURM execution logs (`logs/*.out`, `logs/*.err`).
  * [`pyproject.toml`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/pyproject.toml): Defines dependencies (`pydantic>=2.7.0`, `tabulate>=0.9.0`, `pytest>=8.0.0`).
* **Source Code (`src/agent_gmmis/`)**:
  * [`src/agent_gmmis/__init__.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/__init__.py): Package version `0.1.0`.
  * [`src/agent_gmmis/config.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/config.py): Path resolution for root, data directory, and default database.
  * [`src/agent_gmmis/data/__init__.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/data/__init__.py): Data package exports.
  * [`src/agent_gmmis/data/enums.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/data/enums.py): Enums for `ImagingModality`, `AnatomicalRegion`, `FusionLevel`, and `ScoreCategory`.
  * [`src/agent_gmmis/data/models.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/data/models.py): Strongly-typed schemas for `ModelMetadata`, `DatasetMetadata`, `BenchmarkScore`, and `InclusionCriteriaCheck`.
  * [`src/agent_gmmis/data/db_manager.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/data/db_manager.py): SQLite DAL methods (`get_all_models`, `get_all_datasets`, `get_scores_for_model`, `get_leaderboard_for_dataset`, `save_model`).
* **Testing (`tests/`)**:
  * [`tests/test_db_layer.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/tests/test_db_layer.py): 9 comprehensive unit and integration tests covering database validation, score normalization, and inclusion rules.

### 4. Verification & Test Results
* **Test Suite**: Executed `.venv/bin/pytest tests/ -v`
* **Result**: All 9 tests passed in 0.08s.
  * 98/98 models from `database.sqlite` loaded and validated with 0 errors.
  * 58/58 datasets loaded and validated with 0 errors.
  * Score range and percentage conversion validated.
  * Inclusion screening logic validated.
* **Live Query Verification**: Verified leaderboard queries on anchor datasets (e.g. `BraTS` top 3 models: `MEA M-SAM`, `3DINO-ViT`, `TransUNet`).

---

## [2026-09-22] — Step 2: Information Extraction Engine & Ground Truth Evaluation

### 1. Objective
Design and implement the autonomous manuscript extraction pipeline (`ExtractionAgent`) to ingest academic PDF papers, extract structured architectural metadata and benchmark evaluation tables using LLMs with Pydantic Structured Outputs, and quantitatively benchmark extraction fidelity against human-curated ground truth from Moglia et al.

### 2. Architecture & Design Decisions
* **High-Speed PDF Document Ingestion**: Implemented `PDFParser` powered by PyMuPDF (`fitz`), featuring heuristic table-dense page detection, abstract isolation, and structural sectioning.
* **Universal Model-Agnostic LLM Layer**: Designed `LLMClient` implementing the standard OpenAI-compatible API to seamlessly switch between:
  * **Google Gemini 2.0 Flash (Free Tier)** via Google AI Studio (zero cost, 1M context window).
  * **Local Ollama** (offline on Mac).
  * **DEIB Mufasa HPC Cluster** (self-hosted models).
  * **Mock / Offline Provider** (for reproducible, zero-network unit testing and CI).
* **Pydantic Structured Outputs**: Used strict response models (`ExtractedPaperData`, `RawExtractedScore`) to enforce JSON schema conformance, eliminate parsing hallucinations, and automatically normalize percentage scores ($85.5\% \rightarrow 0.855$).
* **Prompt Grounding**: Engineered extraction prompts adhering strictly to the survey's inclusion criteria (3D CT/MRI/PET, public datasets, DSC metrics) and directing the LLM to separate the *proposed* model from *baseline* competitors (e.g. nnU-Net, Swin UNETR).
* **Quantitative Ground Truth Evaluator**: Implemented `ExtractionEvaluator` to compare extracted values against verified records in `database.sqlite`, calculating:
  * Dataset retrieval Precision, Recall, and F1-score.
  * Dice Score Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE).
  * Parameter count percentage error and GitHub code URL match.

### 3. Files Created & Modified
* **Dependencies & Packaging**:
  * [`pyproject.toml`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/pyproject.toml): Added `pymupdf>=1.24.0`, `openai>=1.30.0`, `python-dotenv>=1.0.0`, and test warning filters.
* **Extraction Module (`src/agent_gmmis/extraction/`)**:
  * [`src/agent_gmmis/extraction/__init__.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/__init__.py): Module exports.
  * [`src/agent_gmmis/extraction/pdf_parser.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/pdf_parser.py): `PDFParser` and `ParsedManuscript` extracting text and table locations.
  * [`src/agent_gmmis/extraction/schemas.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/schemas.py): Pydantic schemas for `ExtractedPaperData` and `RawExtractedScore`.
  * [`src/agent_gmmis/extraction/prompts.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/prompts.py): System prompts and extraction guidelines.
  * [`src/agent_gmmis/extraction/extractor.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/extractor.py): `ExtractionAgent` orchestrating document parsing, prompting, and structured extraction.
  * [`src/agent_gmmis/extraction/evaluator.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/evaluator.py): `ExtractionEvaluator` and `ExtractionEvaluationReport`.
* **LLM Client Layer (`src/agent_gmmis/llm/`)**:
  * [`src/agent_gmmis/llm/__init__.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/llm/__init__.py): Module exports.
  * [`src/agent_gmmis/llm/client.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/llm/client.py): `LLMClient` with `generate_structured()` and `generate_text()`.
* **Testing & Sample Data**:
  * [`data/sample_papers/medsam_arxiv.pdf`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/data/sample_papers/medsam_arxiv.pdf): Downloaded MedSAM preprint (arXiv:2304.12306) as anchor sample.
  * [`tests/test_extraction.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/tests/test_extraction.py): 5 test suites covering PDF parsing, schema validation, evaluator metrics, and offline mock agent execution.

### 4. Verification & Test Results
* **Live Paper Benchmark on SAM-Med3D (`sam_med3d_arxiv.pdf`)**:
  * Real manuscript extraction executed using Google AI Studio free tier (`gemini-3.5-flash-lite`, 0.00 €).
  * **Screening**: Automatically verified 3D radiological volume eligibility and public dataset usage (`INCLUDED`).
  * **Metadata Extraction**: Extracted model name (`SAM-Med3D`), affiliations, 3D ViT visual backbone, and exact GitHub repository URL (`https://github.com/uni-medical/SAM-Med3D`).
  * **Table Extraction & Prompt Regimes**: Extracted 35 distinct segmentation scores from Tables 3 & 4 across multiple prompt settings (`1 point`, `3 points`, `5 points`, `10 points`), capturing both proposed model and baseline (`nnU-Net`, `UNETR`) numbers.
  * **Critical Methodological Finding (Primary vs. Best & Interactive Prompting)**:
    * Interactive foundation models (e.g., SAM-based architectures) report different Dice scores depending on prompt interaction count (e.g. `Ours (1 pt)` vs `Ours (10 pts)` in Table 4).
    * In Moglia et al., survey records standardized on the 1-point interactive prompt (`results_primary_comments`: *"Results reported in the 1-point prompt framework"*).
    * Enhanced `RawExtractedScore` schema with `prompt_type` and updated `ExtractionEvaluator` to benchmark directly against primary paper reports (`compare_source="primary"`).
  * **Evaluation against Ground Truth (`database.sqlite` - Primary Report)**:
    * TotalSegmentator (`1 point`): **0.8468** extracted vs. **0.8468** ground truth (diff: **0.0000** - EXACT MATCH).
    * KiTS (`1 point`): **0.7206** extracted vs. **0.7206** ground truth (diff: **0.0000** - EXACT MATCH).
    * BTCV (`1 point`): **0.7917** extracted vs. **0.7917** ground truth (diff: **0.0000** - EXACT MATCH).
    * AMOS-MR (`1 point`): **0.7541** extracted vs. **0.7541** ground truth (diff: **0.0000** - EXACT MATCH).
    * AMOS-CT (`1 point`): **0.7994** extracted (survey note: *"Average Dice score between AMOS2 CT (79.94%) and AMOS2 MRI (75.41%)"*).
    * Overall Dice Score Mean Absolute Error (MAE): **0.0091** (< 0.9%).
    * Code repository URL match: **True** (100% exact match).



