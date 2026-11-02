"""HPC Cluster Controller and SSH / SLURM Remote Orchestrator for Mufasa.

Handles remote cluster health checks, code synchronization, SLURM batch job submission,
live log tailing, job cancellation, and bidirectional benchmark log syncing.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple


class ClusterManager:
    """Manages SSH and SLURM interactions with the DEIB Mufasa A100 GPU cluster."""

    ENV_PREFIX = (
        'export PATH="/opt/shared/sw/amd/gcc-13.3.0/slurm-25.05.4/bin:$HOME/.local/bin:$PATH"; '
    )

    def __init__(
        self,
        host: Optional[str] = None,
        user: Optional[str] = None,
        remote_dir: str = "~/AgentForGMMIS",
    ):
        self.host = host or os.getenv("MUFASA_HOST", "10.79.23.96")
        self.user = user or os.getenv("MUFASA_USER", "lpezzoli")
        self.remote_dir = remote_dir
        self.target = f"{self.user}@{self.host}"
        self.proj_root = Path(__file__).resolve().parent.parent.parent.parent

    def _run_ssh(self, cmd: str, timeout: int = 10) -> Tuple[int, str, str]:
        """Executes a remote command via SSH with standard SLURM environment."""
        full_ssh = [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout={min(timeout, 5)}",
            "-o", "StrictHostKeyChecking=no",
            self.target,
            f"{self.ENV_PREFIX}{cmd}",
        ]
        try:
            res = subprocess.run(
                full_ssh,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return res.returncode, res.stdout, res.stderr
        except subprocess.TimeoutExpired:
            return 124, "", f"SSH command timed out after {timeout} seconds."
        except Exception as exc:
            return 1, "", str(exc)

    def check_mufasa_status(self) -> Dict[str, Any]:
        """
        Pings Mufasa over SSH, retrieves active SLURM jobs, and discovers cached HF models.
        """
        # Composite remote diagnostic command with graceful fallbacks
        check_cmd = (
            "echo '===SECTION:PING===' && echo 'PONG' && "
            "echo '===SECTION:SQUEUE===' && (squeue -u $USER -o '%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R' 2>/dev/null || true) && "
            "echo '===SECTION:HF_MODELS===' && (ls -1 ~/.cache/huggingface/hub/ 2>/dev/null || true)"
        )
        code, stdout, stderr = self._run_ssh(check_cmd, timeout=8)

        if code != 0 or "PONG" not in stdout:
            return {
                "online": False,
                "host": self.host,
                "user": self.user,
                "error": stderr.strip() or stdout.strip() or "Connection timed out or unreachable.",
                "cached_models": [],
                "active_jobs": [],
            }

        parts = stdout.split("===SECTION:")
        squeue_raw = ""
        hf_raw = ""

        for p in parts:
            if p.startswith("SQUEUE==="):
                squeue_raw = p.replace("SQUEUE===", "", 1).strip()
            elif p.startswith("HF_MODELS==="):
                hf_raw = p.replace("HF_MODELS===", "", 1).strip()

        # Parse active jobs
        active_jobs: List[Dict[str, Any]] = []
        if squeue_raw:
            lines = squeue_raw.splitlines()
            if len(lines) > 1:
                for line in lines[1:]:
                    parts = line.split()
                    if len(parts) >= 6:
                        active_jobs.append({
                            "job_id": parts[0],
                            "partition": parts[1],
                            "name": parts[2],
                            "user": parts[3],
                            "state": parts[4],
                            "time": parts[5],
                            "nodes": parts[6] if len(parts) > 6 else "1",
                            "nodelist": parts[7] if len(parts) > 7 else "",
                        })

        # Parse cached Hugging Face models
        cached_models: List[str] = []
        if hf_raw:
            for line in hf_raw.splitlines():
                line = line.strip()
                if line.startswith("models--"):
                    clean_name = line.replace("models--", "").replace("--", "/")
                    cached_models.append(clean_name)

        return {
            "online": True,
            "host": self.host,
            "user": self.user,
            "cached_models": sorted(cached_models),
            "active_jobs": active_jobs,
            "error": None,
        }

    def sync_code_to_mufasa(self) -> Tuple[bool, str]:
        """Pushes current local codebase to Mufasa using rsync."""
        rsync_cmd = [
            "rsync",
            "-avz",
            "--exclude=.venv",
            "--exclude=logs",
            "--exclude=data/cache_papers",
            "--exclude=__pycache__",
            "--exclude=.git",
            str(self.proj_root / "src"),
            str(self.proj_root / "scripts"),
            str(self.proj_root / "pyproject.toml"),
            f"{self.target}:{self.remote_dir}/",
        ]
        try:
            res = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=30)
            return res.returncode == 0, res.stdout if res.returncode == 0 else res.stderr
        except Exception as exc:
            return False, str(exc)

    def launch_mufasa_job(
        self,
        launch_cfg: Dict[str, Any],
        sync_first: bool = True,
    ) -> Tuple[bool, str]:
        """
        Submits a unified benchmark job to SLURM on Mufasa.
        """
        if sync_first:
            ok, sync_out = self.sync_code_to_mufasa()
            if not ok:
                return False, f"Code sync failed before job launch:\n{sync_out}"

        model_a = launch_cfg.get("model_a", "qwen2.5-7b-awq")
        model_b = launch_cfg.get("model_b", "")
        provider_a = launch_cfg.get("provider_a", "mufasa")
        provider_b = launch_cfg.get("provider_b", "")
        count = launch_cfg.get("count", 40)
        max_rounds = launch_cfg.get("max_rounds", 3)
        compare_source = launch_cfg.get("compare_source", "primary")
        tag = launch_cfg.get("tag", "")

        export_vars = [
            f"MODEL_A={model_a}",
            f"COUNT={count}",
            f"MAX_ROUNDS={max_rounds}",
            f"COMPARE_SOURCE={compare_source}",
        ]
        if model_b:
            export_vars.append(f"MODEL_B={model_b}")
        if provider_a:
            export_vars.append(f"PROVIDER_A={provider_a}")
        if provider_b:
            export_vars.append(f"PROVIDER_B={provider_b}")
        if tag:
            export_vars.append(f"TAG={tag}")

        export_arg = ",".join(["ALL"] + export_vars)

        sbatch_cmd = (
            f"cd {self.remote_dir} && "
            f"sbatch --export={export_arg} scripts/mufasa/run_unified_benchmark_mufasa.sh"
        )

        code, stdout, stderr = self._run_ssh(sbatch_cmd, timeout=15)
        if code == 0 and "Submitted batch job" in stdout:
            job_id_match = re.search(r"Submitted batch job (\d+)", stdout)
            job_id = job_id_match.group(1) if job_id_match else stdout.strip()
            return True, job_id
        return False, stderr.strip() or stdout.strip()

    def get_mufasa_job_logs(self, job_id: str, tail_lines: int = 100) -> str:
        """Tails the stdout/stderr and vLLM logs of a running or completed SLURM job."""
        fetch_cmd = (
            f"cd {self.remote_dir} && "
            f"echo '=== SLURM STDOUT/STDERR (Job {job_id}) ===' && "
            f"(cat logs/*_{job_id}.out logs/*_{job_id}.err 2>/dev/null | tail -n {tail_lines} || echo 'No SLURM log yet.') && "
            f"echo '\n=== vLLM SERVER A LOG ===' && "
            f"(cat logs/vllm_server_a_{job_id}.log 2>/dev/null | tail -n 40 || echo 'No Server A log.') && "
            f"echo '\n=== vLLM SERVER B LOG ===' && "
            f"(cat logs/vllm_server_b_{job_id}.log 2>/dev/null | tail -n 40 || echo 'No Server B log.')"
        )
        code, stdout, stderr = self._run_ssh(fetch_cmd, timeout=10)
        return stdout if code == 0 else f"Error fetching logs:\n{stderr}"

    def cancel_mufasa_job(self, job_id: str) -> Tuple[bool, str]:
        """Cancels a running SLURM job via scancel."""
        cancel_cmd = f"scancel {job_id}"
        code, stdout, stderr = self._run_ssh(cancel_cmd, timeout=10)
        return code == 0, stdout if code == 0 else stderr

    def sync_logs_from_mufasa(self) -> Tuple[bool, str]:
        """Synchronizes benchmark run artifacts from Mufasa to local logs/mufasa_runs/."""
        sync_script = self.proj_root / "scripts" / "sync_mufasa_logs.sh"
        if sync_script.exists():
            try:
                res = subprocess.run([str(sync_script)], capture_output=True, text=True, timeout=60)
                return res.returncode == 0, res.stdout if res.returncode == 0 else res.stderr
            except Exception as exc:
                return False, str(exc)

        # Fallback direct rsync
        dest = self.proj_root / "logs" / "mufasa_runs"
        dest.mkdir(parents=True, exist_ok=True)
        rsync_cmd = [
            "rsync",
            "-avz",
            "--progress",
            f"{self.target}:{self.remote_dir}/logs/runs/",
            f"{dest}/",
        ]
        try:
            res = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=60)
            return res.returncode == 0, res.stdout if res.returncode == 0 else res.stderr
        except Exception as exc:
            return False, str(exc)

    def download_model_on_mufasa(self, hf_repo_id: str) -> Tuple[bool, str]:
        """Triggers model pre-caching on Mufasa in background via download script."""
        dl_cmd = f"cd {self.remote_dir} && nohup bash scripts/mufasa/download_open_models.sh '{hf_repo_id}' > logs/download_{sanitize_filename(hf_repo_id)}.log 2>&1 &"
        code, stdout, stderr = self._run_ssh(dl_cmd, timeout=10)
        return code == 0, stdout if code == 0 else stderr
