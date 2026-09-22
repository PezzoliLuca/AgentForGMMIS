"""SQLite Data Access Layer (DAL) for GMMIS benchmark data."""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from agent_gmmis.config import DEFAULT_DB_PATH
from agent_gmmis.data.enums import ScoreCategory
from agent_gmmis.data.models import BenchmarkScore, DatasetMetadata, ModelMetadata


class DatabaseManager:
    """Provides validated data access to the GMMIS SQLite knowledge base."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at: {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a connection with Row factory enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ---------------------------------------------------------
    # Models
    # ---------------------------------------------------------
    def get_all_models(self) -> List[ModelMetadata]:
        """Retrieves and validates all models from the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM models;")
            rows = cursor.fetchall()
            return [ModelMetadata(**dict(row)) for row in rows]

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Retrieves a single model by its ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM models WHERE ID = ?;", (model_id,))
            row = cursor.fetchone()
            if row:
                return ModelMetadata(**dict(row))
            return None

    def count_models(self) -> int:
        """Returns total number of models in database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT count(*) FROM models;")
            return cursor.fetchone()[0]

    # ---------------------------------------------------------
    # Datasets
    # ---------------------------------------------------------
    def get_all_datasets(self) -> List[DatasetMetadata]:
        """Retrieves and validates all datasets from the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM datasets;")
            rows = cursor.fetchall()
            return [DatasetMetadata(**dict(row)) for row in rows]

    def get_dataset(self, name: str) -> Optional[DatasetMetadata]:
        """Retrieves a single dataset by its canonical name."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM datasets WHERE Name = ?;", (name,))
            row = cursor.fetchone()
            if row:
                return DatasetMetadata(**dict(row))
            return None

    def count_datasets(self) -> int:
        """Returns total number of benchmark datasets."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT count(*) FROM datasets;")
            return cursor.fetchone()[0]

    # ---------------------------------------------------------
    # Benchmark Scores & Leaderboards
    # ---------------------------------------------------------
    def get_scores_for_model(self, model_id: str) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Retrieves all evaluated dataset scores for a specific model.
        Returns: { dataset_name: { 'primary': float, 'best': float } }
        """
        results: Dict[str, Dict[str, Optional[float]]] = {}
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Primary scores
            cursor.execute("SELECT * FROM results_primary WHERE ID = ?;", (model_id,))
            primary_row = cursor.fetchone()
            if primary_row:
                for col in primary_row.keys():
                    if col not in ("ID", "Related Paper", "Date"):
                        val = primary_row[col]
                        if val is not None:
                            results.setdefault(col, {})["primary"] = round(float(val), 4)

            # Best scores
            cursor.execute("SELECT * FROM results_best WHERE ID = ?;", (model_id,))
            best_row = cursor.fetchone()
            if best_row:
                for col in best_row.keys():
                    if col not in ("ID", "Related Paper", "Date"):
                        val = best_row[col]
                        if val is not None:
                            results.setdefault(col, {})["best"] = round(float(val), 4)

        return results

    def get_leaderboard_for_dataset(
        self,
        dataset_name: str,
        category: ScoreCategory = ScoreCategory.BEST,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Returns top-performing models for a specified dataset.
        Returns: [(model_id, dice_score), ...]
        """
        table = "results_best" if category == ScoreCategory.BEST else "results_primary"
        query = f'SELECT ID, "{dataset_name}" as score FROM {table} WHERE "{dataset_name}" IS NOT NULL ORDER BY "{dataset_name}" DESC LIMIT ?;'

        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(query, (top_k,))
                rows = cursor.fetchall()
                return [(row["ID"], round(float(row["score"]), 4)) for row in rows]
            except sqlite3.OperationalError:
                # Column does not exist in table
                return []

    # ---------------------------------------------------------
    # Insert / Upsert (For Downstream Database Agent)
    # ---------------------------------------------------------
    def save_model(self, model: ModelMetadata) -> None:
        """Inserts or replaces a model record."""
        query = """
        INSERT OR REPLACE INTO models (
            ID, Title, [Major Affiliations], [First Publication Date], [First Publisher],
            [First Publication Link], [First Publication BibKey], [Last Publication Date],
            [Last Publisher], [Last Publication Link], [Last Publication BibKey],
            Code, Framework, Architecture, [Visual Backbone], [Millions of Parameters],
            [Number of GFlops], Resources, [Resources Total V-RAM]
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        values = (
            model.id,
            model.title,
            model.major_affiliations,
            model.first_publication_date,
            model.first_publisher,
            model.first_publication_link,
            model.first_publication_bibkey,
            model.last_publication_date,
            model.last_publisher,
            model.last_publication_link,
            model.last_publication_bibkey,
            model.code,
            model.framework,
            model.architecture,
            model.visual_backbone,
            model.millions_of_parameters,
            model.number_of_gflops,
            model.resources,
            model.resources_total_vram,
        )
        with self._get_connection() as conn:
            conn.execute(query, values)
            conn.commit()
