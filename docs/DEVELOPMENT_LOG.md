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
