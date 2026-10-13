"""GitHub Repository Inspector for the Delegation Layer of AgentForGMMIS.

Inspects public code repositories (e.g. GitHub) to recover quantitative benchmark
scores when scientific manuscripts omit scalar tables (e.g. BiomedParse, radar plots).
Parses README markdown tables, evaluation JSON files, and benchmark CSV artifacts.
"""

import csv
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from agent_gmmis.auditing.schemas import RepoInspectionResult
from agent_gmmis.config import PROJECT_ROOT
from agent_gmmis.extraction.alias_resolver import DatasetAliasResolver
from agent_gmmis.extraction.schemas import ExtractedPaperData, RawExtractedScore

logger = logging.getLogger(__name__)


class RepoInspector:
    """Inspects code repositories to extract evaluation metrics directly from artifacts."""

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        alias_resolver: Optional[DatasetAliasResolver] = None,
        timeout_seconds: int = 45,
    ) -> None:
        """Initialize the RepoInspector.

        Args:
            cache_dir: Directory to cache cloned repositories. Defaults to data/cache_repos/.
            alias_resolver: Resolver to map dataset synonyms to canonical benchmarks.
            timeout_seconds: Maximum execution time for git operations.
        """
        if cache_dir is None:
            self.cache_dir = PROJECT_ROOT / "data" / "cache_repos"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.alias_resolver = alias_resolver or DatasetAliasResolver()
        self.timeout_seconds = timeout_seconds

    @classmethod
    def parse_github_url(cls, url: str) -> Optional[Tuple[str, str]]:
        """Extracts (owner, repo) from a GitHub URL or string.

        Supports formats like:
        - https://github.com/owner/repo
        - https://github.com/owner/repo.git
        - https://github.com/owner/repo/tree/main/subfolder
        - https://owner.github.io/repo/
        - Shortlinks like https://aka.ms/biomedparse-project
        - git@github.com:owner/repo.git

        Returns:
            Tuple of (owner, repo) or None if invalid.
        """
        if not url:
            return None
        clean_url = url.strip()

        # Support https://owner.github.io/repo
        io_match = re.search(r"https?://(?P<owner>[\w.-]+)\.github\.io/(?P<repo>[\w.-]+)", clean_url, re.IGNORECASE)
        if io_match:
            return io_match.group("owner"), io_match.group("repo").rstrip("/")

        match = re.search(r"github\.com[/:](?P<owner>[\w.-]+)/(?P<repo>[\w.-]+)", clean_url, re.IGNORECASE)
        if match:
            owner = match.group("owner")
            repo = match.group("repo")
            if repo.endswith(".git"):
                repo = repo[:-4]
            return owner, repo

        # Follow redirects for shortlinks (e.g. aka.ms/biomedparse-project)
        if any(d in clean_url.lower() for d in ("aka.ms", "bit.ly", "tinyurl.com", "t.co", "cutt.ly")):
            try:
                import ssl, urllib.request
                ctx = ssl._create_unverified_context()
                req = urllib.request.Request(clean_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=5, context=ctx) as resp:
                    redirected = resp.url
                    if redirected and redirected != clean_url:
                        return cls.parse_github_url(redirected)
            except Exception:
                pass

        return None

    def _ensure_repository(self, repo_url_or_path: str) -> Tuple[Optional[Path], Optional[str]]:
        """Ensures repository files are available locally.

        If a local directory path is provided and exists, uses it directly.
        If a GitHub URL is provided, clones into cache_dir using shallow clone.

        Returns:
            Tuple of (local_repo_path, error_message).
        """
        local_path = Path(repo_url_or_path)
        if local_path.is_dir():
            return local_path, None

        parsed = self.parse_github_url(repo_url_or_path)
        if not parsed:
            return None, f"Invalid or unsupported repository URL: {repo_url_or_path}"

        owner, repo = parsed
        target_dir = self.cache_dir / f"{owner}_{repo}"

        if target_dir.is_dir():
            if any(p.name != ".git" for p in target_dir.iterdir()):
                return target_dir, None
            # Incomplete or broken clone (only .git exists)
            import shutil
            shutil.rmtree(target_dir, ignore_errors=True)

        clone_url = f"https://github.com/{owner}/{repo}.git"
        logger.info(f"Cloning repository shallowly into {target_dir}...")
        try:
            cmd = ["git", "clone", "--depth", "1", clone_url, str(target_dir)]
            env = os.environ.copy()
            env["GIT_LFS_SKIP_SMUDGE"] = "1"
            res = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            if res.returncode != 0:
                import shutil
                shutil.rmtree(target_dir, ignore_errors=True)
                err_msg = res.stderr.strip() or res.stdout.strip()
                return None, f"Git clone failed (exit code {res.returncode}): {err_msg}"
            return target_dir, None
        except subprocess.TimeoutExpired:
            import shutil
            shutil.rmtree(target_dir, ignore_errors=True)
            return None, f"Git clone timed out after {self.timeout_seconds}s"
        except Exception as e:
            import shutil
            shutil.rmtree(target_dir, ignore_errors=True)
            return None, f"Unexpected error cloning repository: {str(e)}"

    def _find_candidate_files(self, repo_dir: Path) -> List[Path]:
        """Finds potential benchmark documentation and artifact files within the repository."""
        candidates: List[Path] = []
        ignored_names = {
            "package.json",
            "package-lock.json",
            "tsconfig.json",
            "pyproject.toml",
            "setup.py",
            "requirements.txt",
            ".gitignore",
        }

        # 1. Readme files (highest priority)
        for ext in ("md", "markdown", "rst", "txt"):
            for name in (f"README.{ext}", f"readme.{ext}", f"Readme.{ext}"):
                p = repo_dir / name
                if p.is_file() and p not in candidates:
                    candidates.append(p)

        # 2. Markdown documentation files & result files
        for root, dirs, files in os.walk(repo_dir):
            dirs[:] = [
                d for d in dirs
                if not d.startswith(".")
                and d != "node_modules"
                and d != "__pycache__"
                and d != "assets"
                and d != "checkpoints"
                and d != "weights"
            ]
            for f in files:
                f_low = f.lower()
                if f_low in ignored_names:
                    continue
                p = Path(root) / f
                if p in candidates:
                    continue
                if f_low.endswith((".md", ".markdown")):
                    if any(term in f_low for term in ("result", "eval", "metric", "benchmark", "score", "table")):
                        candidates.append(p)
                elif f_low.endswith((".json", ".csv", ".tsv")):
                    if any(term in f_low for term in ("result", "eval", "metric", "benchmark", "score", "test")):
                        candidates.append(p)

        return candidates

    def _parse_score_value(self, val_str: Any) -> Optional[float]:
        """Parses a float or percentage string into a normalized [0.0, 1.0] Dice score."""
        if val_str is None:
            return None
        if isinstance(val_str, (int, float)):
            v = float(val_str)
        else:
            s = str(val_str).strip()
            # Remove percentage sign or +/- standard deviation (e.g. "88.5 ± 2.1" or "0.885±0.02")
            s = re.split(r"[±+/\-]", s)[0].strip()
            s = s.rstrip("%").strip()
            # Extract first floating-point number
            m = re.search(r"(\d+(?:\.\d+)?)", s)
            if not m:
                return None
            try:
                v = float(m.group(1))
            except ValueError:
                return None

        # Normalize percentages
        if 1.0 < v <= 100.0:
            v = v / 100.0
        if 0.0 <= v <= 1.0:
            return round(v, 4)
        return None

    def _parse_markdown_tables(
        self,
        content: str,
        rel_path: str,
        model_name: Optional[str] = None,
    ) -> List[RawExtractedScore]:
        """Extracts benchmark Dice scores from Markdown tables."""
        scores: List[RawExtractedScore] = []
        lines = [line.strip() for line in content.splitlines()]

        table_lines: List[str] = []
        for line in lines:
            if line.startswith("|") and line.endswith("|"):
                table_lines.append(line)
            else:
                if len(table_lines) >= 3:
                    scores.extend(self._process_single_markdown_table(table_lines, rel_path, model_name))
                table_lines = []

        if len(table_lines) >= 3:
            scores.extend(self._process_single_markdown_table(table_lines, rel_path, model_name))

        return scores

    def _process_single_markdown_table(
        self,
        table_lines: List[str],
        rel_path: str,
        model_name: Optional[str],
    ) -> List[RawExtractedScore]:
        """Parses a single detected markdown table."""
        scores: List[RawExtractedScore] = []
        # Header is line 0, separator is line 1, data starts from line 2
        raw_header = [col.strip() for col in table_lines[0].strip("|").split("|")]
        header_lower = [h.lower() for h in raw_header]

        # Determine table layout:
        # Layout A: Long format (columns: Dataset, Organ, [Model], Dice/DSC, ...)
        # Layout B: Wide format (columns: Dataset, Model1, Model2, ProposedModel, ...)
        dice_col_idx: Optional[int] = None
        dataset_col_idx: Optional[int] = None
        organ_col_idx: Optional[int] = None
        model_col_idx: Optional[int] = None

        for idx, h in enumerate(header_lower):
            if any(term in h for term in ("dataset", "benchmark")):
                if dataset_col_idx is None:
                    dataset_col_idx = idx
            elif any(term in h for term in ("organ", "anatomy", "target", "task")):
                if organ_col_idx is None:
                    organ_col_idx = idx
            elif any(term in h for term in ("dice", "dsc", "mean dice", "dice(%)", "dsc(%)")):
                if dice_col_idx is None:
                    dice_col_idx = idx
            elif any(term in h for term in ("model", "method", "approach")):
                if model_col_idx is None:
                    model_col_idx = idx
            elif "data" in h and dataset_col_idx is None:
                dataset_col_idx = idx

        # Check if table is wide format: columns are model names
        target_model_col_idx: Optional[int] = None
        if dice_col_idx is None and dataset_col_idx is not None:
            # Check if one of the column headers matches proposed model or "ours"
            m_target = (model_name or "").lower().strip()
            for idx, h in enumerate(header_lower):
                if idx == dataset_col_idx:
                    continue
                if (m_target and m_target in h) or any(term in h for term in ("ours", "proposed", "our method")):
                    target_model_col_idx = idx
                    break

        for row_str in table_lines[2:]:
            cells = [cell.strip() for cell in row_str.strip("|").split("|")]
            if len(cells) < len(raw_header):
                continue

            # Process Layout A (Long Format with explicit Dice column)
            if dice_col_idx is not None and dataset_col_idx is not None:
                # If there is a model column, verify it matches our model or is generic
                if model_col_idx is not None and model_name:
                    row_model = cells[model_col_idx].lower()
                    m_target = model_name.lower().strip()
                    if m_target not in row_model and not any(k in row_model for k in ("ours", "proposed")):
                        continue

                raw_ds = cells[dataset_col_idx]
                canonical_ds = self.alias_resolver.resolve(raw_ds) or raw_ds
                val = self._parse_score_value(cells[dice_col_idx])
                if val is not None and canonical_ds:
                    organ = cells[organ_col_idx] if organ_col_idx is not None else "Overall"
                    scores.append(
                        RawExtractedScore(
                            dataset_name=canonical_ds,
                            organ_or_task=organ,
                            dice_score=val,
                            table_or_section_ref=f"repo:{rel_path} (table)",
                            evaluated_model_name=model_name,
                        )
                    )

            # Process Layout B (Wide Format with model columns)
            elif target_model_col_idx is not None and dataset_col_idx is not None:
                raw_ds = cells[dataset_col_idx]
                canonical_ds = self.alias_resolver.resolve(raw_ds) or raw_ds
                val = self._parse_score_value(cells[target_model_col_idx])
                if val is not None and canonical_ds:
                    scores.append(
                        RawExtractedScore(
                            dataset_name=canonical_ds,
                            organ_or_task="Overall",
                            dice_score=val,
                            table_or_section_ref=f"repo:{rel_path} (table)",
                            evaluated_model_name=model_name,
                        )
                    )

        return scores

    def _parse_json_file(
        self,
        json_path: Path,
        rel_path: str,
        model_name: Optional[str] = None,
    ) -> List[RawExtractedScore]:
        """Parses structured benchmark results from JSON files."""
        scores: List[RawExtractedScore] = []
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return scores

        def extract_from_dict(d: Dict[str, Any], prefix_model: Optional[str] = None) -> None:
            for k, v in d.items():
                # Case 1: Dict key is a dataset name mapping directly to float/score
                canonical = self.alias_resolver.resolve(k)
                if canonical and isinstance(v, (int, float, str)):
                    parsed_score = self._parse_score_value(v)
                    if parsed_score is not None:
                        scores.append(
                            RawExtractedScore(
                                dataset_name=canonical,
                                organ_or_task="Overall",
                                dice_score=parsed_score,
                                table_or_section_ref=f"repo:{rel_path}",
                                evaluated_model_name=prefix_model or model_name,
                            )
                        )
                # Case 2: Dict key is a dataset name mapping to sub-dictionary (e.g. {"BTCV": {"dice": 0.885}})
                elif canonical and isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        if any(term in sub_k.lower() for term in ("dice", "dsc", "mean")):
                            parsed_score = self._parse_score_value(sub_v)
                            if parsed_score is not None:
                                scores.append(
                                    RawExtractedScore(
                                        dataset_name=canonical,
                                        organ_or_task="Overall",
                                        dice_score=parsed_score,
                                        table_or_section_ref=f"repo:{rel_path}",
                                        evaluated_model_name=prefix_model or model_name,
                                    )
                                )
                # Case 3: Dict key matches model name or generic results container
                elif isinstance(v, dict):
                    m_key = k.lower()
                    if model_name and model_name.lower() in m_key:
                        extract_from_dict(v, prefix_model=model_name)
                    elif any(c in m_key for c in ("results", "metrics", "benchmarks", "eval")):
                        extract_from_dict(v, prefix_model=prefix_model)

        if isinstance(data, dict):
            extract_from_dict(data)
        elif isinstance(data, list):
            # List of records: [{"dataset": "BTCV", "dice": 0.885}, ...]
            for item in data:
                if not isinstance(item, dict):
                    continue
                ds_val = None
                dice_val = None
                organ_val = "Overall"
                for ik, iv in item.items():
                    ik_low = ik.lower()
                    if any(term in ik_low for term in ("dataset", "benchmark", "name")):
                        ds_val = str(iv)
                    elif any(term in ik_low for term in ("dice", "dsc", "score")):
                        dice_val = iv
                    elif any(term in ik_low for term in ("organ", "anatomy", "task")):
                        organ_val = str(iv)
                if ds_val and dice_val is not None:
                    canonical = self.alias_resolver.resolve(ds_val) or ds_val
                    parsed_score = self._parse_score_value(dice_val)
                    if parsed_score is not None:
                        scores.append(
                            RawExtractedScore(
                                dataset_name=canonical,
                                organ_or_task=organ_val,
                                dice_score=parsed_score,
                                table_or_section_ref=f"repo:{rel_path}",
                                evaluated_model_name=model_name,
                            )
                        )

        return scores

    def _parse_csv_file(
        self,
        csv_path: Path,
        rel_path: str,
        model_name: Optional[str] = None,
    ) -> List[RawExtractedScore]:
        """Parses benchmark results from CSV or TSV files."""
        scores: List[RawExtractedScore] = []
        try:
            with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
                # Sniff delimiter
                sample = f.read(2048)
                f.seek(0)
                delimiter = "\t" if "\t" in sample and "," not in sample else ","
                reader = csv.DictReader(f, delimiter=delimiter)
                if not reader.fieldnames:
                    return scores

                fields_low = {fn: fn.lower().strip() for fn in reader.fieldnames if fn}
                ds_field = next(
                    (fn for fn, fl in fields_low.items() if any(t in fl for t in ("dataset", "benchmark", "task"))),
                    None,
                )
                dice_field = next(
                    (fn for fn, fl in fields_low.items() if any(t in fl for t in ("dice", "dsc", "mean dice"))),
                    None,
                )
                organ_field = next(
                    (fn for fn, fl in fields_low.items() if any(t in fl for t in ("organ", "anatomy", "task"))),
                    None,
                )
                model_field = next(
                    (fn for fn, fl in fields_low.items() if any(t in fl for t in ("model", "method"))),
                    None,
                )

                if ds_field and dice_field:
                    for row in reader:
                        if model_field and model_name:
                            rm = str(row.get(model_field, "")).lower()
                            if model_name.lower() not in rm and not any(k in rm for k in ("ours", "proposed")):
                                continue
                        raw_ds = row.get(ds_field, "")
                        canonical = self.alias_resolver.resolve(raw_ds) or raw_ds
                        parsed_score = self._parse_score_value(row.get(dice_field))
                        if parsed_score is not None and canonical:
                            organ = row.get(organ_field, "Overall") if organ_field else "Overall"
                            scores.append(
                                RawExtractedScore(
                                    dataset_name=canonical,
                                    organ_or_task=organ,
                                    dice_score=parsed_score,
                                    table_or_section_ref=f"repo:{rel_path}",
                                    evaluated_model_name=model_name,
                                )
                            )
                elif dice_field and not ds_field:
                    # Dataset name may reside in the directory structure (e.g. results/amos22/CT/liver_metrics.csv)
                    path_parts = Path(rel_path).parts
                    path_ds = None
                    for part in reversed(path_parts[:-1]):
                        res = self.alias_resolver.resolve(part)
                        if res:
                            path_ds = res
                            break
                    if path_ds:
                        dices = []
                        for row in reader:
                            val = self._parse_score_value(row.get(dice_field))
                            if val is not None:
                                dices.append(val)
                        if dices:
                            mean_dice = round(sum(dices) / len(dices), 4)
                            organ_raw = csv_path.stem.replace("_metrics", "").replace("+", " ").strip()
                            organ = organ_raw.title() if organ_raw else "Overall"
                            scores.append(
                                RawExtractedScore(
                                    dataset_name=path_ds,
                                    organ_or_task=organ,
                                    dice_score=mean_dice,
                                    table_or_section_ref=f"repo:{rel_path} (mean of {len(dices)} cases)",
                                    evaluated_model_name=model_name,
                                )
                            )
        except Exception:
            pass

        return scores

    def inspect_repository(
        self,
        repo_url_or_path: str,
        model_name: Optional[str] = None,
    ) -> RepoInspectionResult:
        """Main entry point: inspects repository files and extracts benchmark metrics.

        Args:
            repo_url_or_path: GitHub URL (e.g. 'https://github.com/microsoft/BiomedParse')
                              or local folder path.
            model_name: Name of proposed model to filter benchmark columns/rows.

        Returns:
            RepoInspectionResult with extracted scores, inspected files, and diagnostic message.
        """
        repo_dir, err = self._ensure_repository(repo_url_or_path)
        if err or not repo_dir:
            return RepoInspectionResult(
                repo_url=repo_url_or_path,
                success=False,
                message=err or "Unknown error accessing repository.",
            )

        candidate_files = self._find_candidate_files(repo_dir)
        inspected_rel_paths: List[str] = []
        all_scores: List[RawExtractedScore] = []

        for file_path in candidate_files:
            rel_path = str(file_path.relative_to(repo_dir))
            inspected_rel_paths.append(rel_path)
            f_low = file_path.name.lower()

            if f_low.endswith((".md", ".markdown", ".rst", ".txt")):
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    md_scores = self._parse_markdown_tables(content, rel_path, model_name=model_name)
                    all_scores.extend(md_scores)
                except Exception as e:
                    logger.debug(f"Error parsing markdown file {file_path}: {e}")

            elif f_low.endswith(".json"):
                json_scores = self._parse_json_file(file_path, rel_path, model_name=model_name)
                all_scores.extend(json_scores)

            elif f_low.endswith((".csv", ".tsv")):
                csv_scores = self._parse_csv_file(file_path, rel_path, model_name=model_name)
                all_scores.extend(csv_scores)

        # Deduplicate scores: keep unique (dataset_name, organ_or_task)
        unique_scores: List[RawExtractedScore] = []
        seen_keys: Set[Tuple[str, str]] = set()
        for sc in all_scores:
            key = (sc.dataset_name.lower(), sc.organ_or_task.lower())
            if key not in seen_keys:
                seen_keys.add(key)
                unique_scores.append(sc)

        success = len(unique_scores) > 0
        summary_msg = (
            f"Successfully extracted {len(unique_scores)} benchmark scores from {len(inspected_rel_paths)} repository files."
            if success
            else f"Inspected {len(inspected_rel_paths)} files in repository, but found no quantitative benchmark tables."
        )

        return RepoInspectionResult(
            repo_url=repo_url_or_path,
            success=success,
            extracted_scores=unique_scores,
            inspected_files=inspected_rel_paths,
            message=summary_msg,
            cache_path=str(repo_dir),
        )

    def inspect_and_update(
        self,
        extracted: ExtractedPaperData,
        auditor: Optional[Any] = None,
        manuscript: Optional[Any] = None,
    ) -> Tuple[ExtractedPaperData, Optional[Any]]:
        """Inspects paper's code repository if available, merges new scores, and optionally re-audits.

        Args:
            extracted: The ExtractedPaperData containing code_repository_url.
            auditor: Optional AuditorAgent instance to re-audit paper upon score recovery or failure.
            manuscript: Optional ParsedManuscript to evaluate visual figures if repo lacks tables.

        Returns:
            Tuple of (updated_extracted_data, optional_audit_report).
        """
        if not extracted.code_repository_url:
            return extracted, None

        result = self.inspect_repository(
            repo_url_or_path=extracted.code_repository_url,
            model_name=extracted.proposed_model_id,
        )

        audit_report = None
        if result.success and result.extracted_scores:
            # Purge any LLM template hallucinations (scores repeated 4+ times across tasks)
            proposed_scores = [
                s.dice_score for s in extracted.extracted_scores
                if not s.is_baseline_comparison and s.dice_score is not None
            ]
            if len(proposed_scores) >= 4:
                from collections import Counter
                counts = Counter(proposed_scores)
                hallucinated_values = {val for val, count in counts.items() if count >= 4}
                if hallucinated_values:
                    extracted.extracted_scores = [
                        s for s in extracted.extracted_scores
                        if s.is_baseline_comparison or s.dice_score not in hallucinated_values
                    ]

            # Merge genuine repo scores avoiding exact duplicates
            existing_keys = {
                (s.dataset_name.lower(), s.organ_or_task.lower()) for s in extracted.extracted_scores if s.dice_score is not None
            }
            new_scores_added = 0
            for score in result.extracted_scores:
                k = (score.dataset_name.lower(), score.organ_or_task.lower())
                if k not in existing_keys:
                    extracted.extracted_scores.append(score)
                    existing_keys.add(k)
                    new_scores_added += 1

            if new_scores_added > 0 and auditor is not None:
                audit_report = auditor.audit(extracted, manuscript=manuscript)
        elif auditor is not None:
            # Repository was inspected, but contained no quantitative benchmark tables.
            # Re-audit with repo_inspected=True so Auditor can transition to NEEDS_OUTREACH.
            audit_report = auditor.audit(extracted, manuscript=manuscript, repo_inspected=True)

        return extracted, audit_report
