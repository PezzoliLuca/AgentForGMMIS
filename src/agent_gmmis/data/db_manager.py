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
        """Retrieves a single model by its ID (exact or case-insensitive/partial fallback)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM models WHERE ID = ?;", (model_id,))
            row = cursor.fetchone()
            if row:
                return ModelMetadata(**dict(row))

            # Case-insensitive / partial fallback (e.g. 'MedSAM-2' matches 'Medical SAM 2 (MedSAM-2)')
            cursor.execute(
                "SELECT * FROM models WHERE LOWER(ID) = LOWER(?) OR LOWER(ID) LIKE LOWER(?) OR LOWER(Title) LIKE LOWER(?);",
                (model_id, f"%{model_id}%", f"%{model_id}%"),
            )
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
        model = self.get_model(model_id)
        target_id = model.id if model else model_id

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Primary scores
            cursor.execute("SELECT * FROM results_primary WHERE ID = ?;", (target_id,))
            primary_row = cursor.fetchone()
            if primary_row:
                for col in primary_row.keys():
                    if col not in ("ID", "Related Paper", "Date"):
                        val = primary_row[col]
                        if val is not None:
                            results.setdefault(col, {})["primary"] = round(float(val), 4)

            # Best scores
            cursor.execute("SELECT * FROM results_best WHERE ID = ?;", (target_id,))
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

    def get_dataset_statistics(
        self,
        dataset_name: str,
        category: ScoreCategory = ScoreCategory.BEST,
    ) -> Optional[Dict[str, float]]:
        """
        Calculates aggregate statistics for a given dataset column across historical models.
        Returns values normalized to [0.0, 1.0]:
        { 'min': float, 'max': float, 'mean': float, 'count': int }
        or None if dataset column doesn't exist or has 0 entries.
        """
        table = "results_best" if category == ScoreCategory.BEST else "results_primary"
        query = f'SELECT MIN("{dataset_name}"), MAX("{dataset_name}"), AVG("{dataset_name}"), COUNT("{dataset_name}") FROM {table} WHERE "{dataset_name}" IS NOT NULL;'

        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(query)
                row = cursor.fetchone()
                if not row or row[3] == 0 or row[0] is None:
                    return None
                min_v = float(row[0]) / 100.0 if float(row[0]) > 1.0 else float(row[0])
                max_v = float(row[1]) / 100.0 if float(row[1]) > 1.0 else float(row[1])
                mean_v = float(row[2]) / 100.0 if float(row[2]) > 1.0 else float(row[2])
                return {
                    "min": round(min_v, 4),
                    "max": round(max_v, 4),
                    "mean": round(mean_v, 4),
                    "count": int(row[3]),
                }
            except sqlite3.OperationalError:
                return None

    # ---------------------------------------------------------
    # Schema Introspection & Dynamic Evolution (For DatabaseAgent)
    # ---------------------------------------------------------
    def get_table_columns(self, table_name: str) -> List[str]:
        """Returns the list of column names for a specified table."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'PRAGMA table_info("{table_name}");')
            return [row["name"] for row in cursor.fetchall()]

    def has_column(self, table_name: str, column_name: str) -> bool:
        """Checks if a column exists in the specified table."""
        return column_name in self.get_table_columns(table_name)

    def add_column_to_table(
        self,
        table_name: str,
        column_name: str,
        col_type: str = "REAL",
        default: Any = None,
    ) -> str:
        """
        Dynamically adds a new column to a table.
        Returns the exact SQL statement executed.
        """
        default_clause = f" DEFAULT {default}" if default is not None else ""
        query = f'ALTER TABLE "{table_name}" ADD COLUMN "{column_name}" {col_type}{default_clause};'
        with self._get_connection() as conn:
            conn.execute(query)
            conn.commit()
        return query

    def init_curation_history_table(self) -> None:
        """Initializes the immutable curation_history table if it does not exist."""
        query = """
        CREATE TABLE IF NOT EXISTS curation_history (
            history_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model_id TEXT NOT NULL,
            action_type TEXT NOT NULL,
            status TEXT NOT NULL,
            audit_status TEXT,
            audit_confidence REAL,
            diff_payload TEXT,
            notes TEXT
        );
        """
        with self._get_connection() as conn:
            conn.execute(query)
            conn.commit()

    def record_curation_history(
        self,
        model_id: str,
        action_type: str,
        status: str,
        audit_status: Optional[str] = None,
        audit_confidence: Optional[float] = None,
        diff_payload: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Inserts a record into the curation_history table."""
        self.init_curation_history_table()
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        query = """
        INSERT INTO curation_history (
            timestamp, model_id, action_type, status, audit_status,
            audit_confidence, diff_payload, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self._get_connection() as conn:
            conn.execute(
                query,
                (now, model_id, action_type, status, audit_status, audit_confidence, diff_payload, notes),
            )
            conn.commit()

    def get_score_raw(
        self,
        model_id: str,
        dataset_name: str,
        table_name: str = "results_best",
    ) -> Optional[float]:
        """
        Returns the exact raw score from results_best or results_primary.
        Returns None if column does not exist or score is NULL.
        """
        if not self.has_column(table_name, dataset_name):
            return None
        query = f'SELECT "{dataset_name}" FROM "{table_name}" WHERE ID = ?;'
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (model_id,))
            row = cursor.fetchone()
            if row and row[0] is not None:
                return float(row[0])
            return None

    def execute_sql_script(self, statements: List[str]) -> None:
        """Executes a list of SQL statements inside an atomic transaction."""
        script_full = "\n".join(stmt.strip() for stmt in statements if stmt.strip())
        with self._get_connection() as conn:
            try:
                conn.executescript(f"BEGIN TRANSACTION;\n{script_full}\nCOMMIT;")
            except Exception as e:
                conn.rollback()
                raise RuntimeError(f"Database transaction rolled back due to error: {e}") from e

    # ---------------------------------------------------------
    # Insert / Upsert (For DatabaseAgent)
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

    def save_dataset(self, dataset: DatasetMetadata) -> None:
        """Inserts or replaces a dataset metadata record."""
        query = """
        INSERT OR REPLACE INTO datasets (
            Name, [Full Name], Description, [Sub-dataset], [Imaging Modality],
            Objects, [Main Anatomical Structure], Region, [Links URL], [Links Text],
            [Cit Keys], [N Tot], [N with Labels]
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        values = (
            dataset.name,
            dataset.full_name,
            dataset.description,
            dataset.sub_dataset,
            dataset.imaging_modality,
            dataset.objects,
            dataset.main_anatomical_structure,
            dataset.region,
            dataset.links_url,
            dataset.links_text,
            dataset.cit_keys,
            dataset.n_tot,
            dataset.n_with_labels,
        )
        with self._get_connection() as conn:
            conn.execute(query, values)
            conn.commit()

