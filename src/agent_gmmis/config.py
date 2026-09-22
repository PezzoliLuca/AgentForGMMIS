"""Configuration loader and path management for AgentForGMMIS."""

import os
from pathlib import Path

# Project root: directory containing pyproject.toml
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data directory and default SQLite database path
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_PATH = DATA_DIR / "database.sqlite"

# Docs directory
DOCS_DIR = PROJECT_ROOT / "docs"

# Log directory
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
