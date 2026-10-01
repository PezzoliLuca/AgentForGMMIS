# AgentForGMMIS
Master's Thesis Project - Pezzoli Luca - luca2.pezzoli@mail.polimi.it

> **Documentation & Thesis Notes**:
> * **[PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)**: Foundational thesis background, paper methodology (Moglia et al., *Information Fusion* 2026), database schema, and system specifications.
> * **[docs/THESIS_DESIGN_LOG.md](docs/THESIS_DESIGN_LOG.md)**: Architectural decisions, multi-agent hierarchy rationale ("Vigile" / Outreach), zero-cost budget strategy, and notes for the final thesis dissertation.
> * **[docs/DEVELOPMENT_LOG.md](docs/DEVELOPMENT_LOG.md)**: Step-by-step descriptive development log tracking all milestones, schema updates, code changes, and test outcomes.

---

## 1. Mufasa Environment Setup (One-time)

### SSH Access
Requires an active DEIB Polimi VPN connection (`gp-deib-saml.vpn.polimi.it`):
```bash
ssh lpezzoli@10.79.23.96
```

### Python Virtual Environment
The environment resides inside the repository at `~/AgentForGMMIS/venv`:
* **Python:** 3.12 (without system `ensurepip`)
* **PyTorch:** 2.6.0+cu124 (CUDA 12.4 supported by host 580.95 drivers / CUDA 13.0)
* **Manual activation:**
  ```bash
  source ~/AgentForGMMIS/venv/bin/activate
  ```

---

## 2. Common Commands Cheat Sheet

### Code Synchronization (Git)

```bash
# On Mac (after editing code)
git add .
git commit -m "..."
git push origin main

# On Mufasa (pulling latest changes before running)
cd ~/AgentForGMMIS
git pull
```

### SLURM Job Management

```bash
# Submit batch job to the queue
sbatch run_job.sh

# Monitor job status (PD = Pending, R = Running)
squeue -u lpezzoli

# Follow output logs in real time
tail -f logs/gmmis_job_<JOBID>.out

# Inspect runtime errors
cat logs/gmmis_job_<JOBID>.err

# Cancel a running or pending job
scancel <JOBID>
```

### Interactive GPU Session (Debug & Prototyping)

To open an interactive shell directly on a compute node with an allocated GPU:
```bash
srun --qos=gpu --gres=gpu:3g.20gb:1 --cpus-per-task=4 --mem=32G --pty /bin/bash
```

## 3. Mufasa Open-Source LLM Execution (vLLM)

Mufasa provides dedicated GPU nodes (`--gres=gpu:3g.20gb:1` MIG slices or full A100s). For autonomous table extraction and JSON verification, the system supports self-hosted open-source models served via high-throughput **vLLM** with OpenAI API compatibility.

### Recommended Models for Mufasa (20 GB VRAM Allocation)

| Model | Format | VRAM Footprint | Max Context | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **`Qwen/Qwen2.5-14B-Instruct-AWQ`** *(Recommended)* | 4-bit AWQ | ~9 GB | 32k - 64k tokens | State-of-the-art structured JSON & table extraction, leaves 11 GB for KV cache |
| **`Qwen/Qwen2.5-7B-Instruct`** | float16 / 4-bit | ~5 - 14 GB | 32k - 128k tokens | Ultra-fast inference, low resource consumption |
| **`meta-llama/Llama-3.1-8B-Instruct`** | float16 / 4-bit | ~8 - 16 GB | 128k tokens | General reasoning and robust prompt alignment |

---

### Step-by-Step Workflow on Mufasa

#### 1. Download Model Weights
Run on Mufasa (login node or interactive session):
```bash
ssh lpezzoli@10.79.23.96
cd ~/AgentForGMMIS
git pull
bash scripts/mufasa/download_model.sh Qwen/Qwen2.5-14B-Instruct-AWQ
```

#### 2. Run Autonomous All-in-One Benchmark via SLURM
Submits an autonomous job that allocates the GPU, launches vLLM in the background, waits for health check, runs the 15-paper benchmark, and cleans up:
```bash
sbatch scripts/mufasa/run_benchmark_mufasa.sh Qwen/Qwen2.5-14B-Instruct-AWQ 15
```
Monitor progress:
```bash
squeue -u lpezzoli
tail -f logs/mufasa_bench_<JOBID>.out
```

#### 3. Alternative: Running Interactively or via Mac Tunnel
If you run `sbatch scripts/mufasa/start_vllm.sh` on Mufasa, you can port-forward to your Mac:
```bash
# On your local Mac:
ssh -L 8000:localhost:8000 lpezzoli@10.79.23.96

# Then in local Mac terminal:
.venv/bin/python3 scripts/run_batch_benchmark.py --provider mufasa --model Qwen/Qwen2.5-14B-Instruct-AWQ --count 15
```

---

## 4. Project Structure

```text
AgentForGMMIS/
├── data/
│   ├── database.sqlite       # Single source of truth (105 models, 64 datasets)
│   └── cache_papers/         # 15 pre-cached arXiv PDF manuscripts
├── scripts/
│   ├── run_batch_benchmark.py # Autonomous batch benchmark runner (all models)
│   ├── run_extraction.py     # Single paper extraction CLI
│   └── mufasa/               # Mufasa SLURM HPC execution scripts
│       ├── download_model.sh
│       ├── start_vllm.sh
│       └── run_benchmark_mufasa.sh
├── src/agent_gmmis/          # Production Python package
│   ├── data/                 # SQLite DAL & Pydantic models
│   ├── extraction/           # PDFParser, Extractor, Prompts, Evaluator
│   ├── auditing/             # AuditorAgent ("Il Vigile") & RepoInspector
│   └── llm/                  # Universal model client (Gemini, Mufasa, Ollama, Mock)
├── tests/                    # 40 comprehensive unit & domain tests (pytest)
└── logs/                     # Benchmark execution logs & Markdown reports
```