# AgentForGMMIS
Master's Thesis Project - Pezzoli Luca - luca2.pezzoli@mail.polimi.it

> **Documentation & Thesis Notes**:
> * **[PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)**: Foundational thesis background, paper methodology (Moglia et al., *Information Fusion* 2026), database schema, and system specifications.
> * **[docs/THESIS_DESIGN_LOG.md](docs/THESIS_DESIGN_LOG.md)**: Architectural decisions, multi-agent hierarchy rationale ("Vigile" / Outreach), zero-cost budget strategy, and notes for the final thesis dissertation.
> * **[docs/DEVELOPMENT_LOG.md](docs/DEVELOPMENT_LOG.md)**: Step-by-step descriptive development log tracking all milestones, schema updates, code changes, and test outcomes.

---

## 1. Mufasa Environment Setup (40 GB A100 Profile)

### SSH Access
Requires an active DEIB Polimi VPN connection (`gp-deib-saml.vpn.polimi.it`):
```bash
ssh amoglia@10.79.23.96
cd ~/lpezzoli/AgentForGMMIS
```
*(Isolated inside `~/lpezzoli/` on the advisor's account to prevent conflicts with other researchers, with symlinked Hugging Face model cache `~/lpezzoli/.cache/huggingface -> /home/lpezzoli/.cache/huggingface` and virtualenv `~/lpezzoli/AgentForGMMIS/venv -> /home/lpezzoli/AgentForGMMIS/venv` to preserve NFS quota).*

### Python Virtual Environment
The environment is accessible at `~/lpezzoli/AgentForGMMIS/venv`:
* **Python:** 3.12 (without system `ensurepip`)
* **PyTorch / vLLM:** PyTorch 2.6.0+cu124 & vLLM 0.30.0 (NVIDIA A100-SXM4-40GB / A100-PCIE-40GB)
* **Manual activation:**
  ```bash
  source ~/lpezzoli/AgentForGMMIS/venv/bin/activate
  export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
  ```

---

## 2. Common Commands Cheat Sheet

### Code & Log Synchronization (Mac <-> Mufasa)

```bash
# Launch interactive Streamlit Dashboard (includes 1-click Code Push, Job Launch & Log Sync)
./scripts/run_dashboard.sh

# Sync completed benchmark runs from Mufasa to local logs/mufasa_runs/
./scripts/sync_mufasa_logs.sh

# Monitor live SLURM job and auto-sync upon completion
./scripts/monitor_and_sync_mufasa.sh
```

### SLURM Job Management (40 GB A100 — `gpuheavy-40`)

```bash
# Submit unified benchmark job (Single, Hybrid, or Dual-Local Simultaneous)
sbatch --export=ALL,MODEL_A=qwen2.5-14b-awq,MODEL_B=deepseek-r1-14b-awq,COUNT=40 scripts/mufasa/run_unified_benchmark_mufasa.sh

# Monitor job status (all our jobs are prefixed with lpezzoli_)
squeue -u amoglia

# Follow output logs in real time
tail -f logs/mufasa_unified_<JOBID>.out

# Cancel a running or pending job
scancel <JOBID>
```

### Interactive 40 GB GPU Session (Debug & Prototyping)

To open an interactive shell directly on a compute node with an allocated 40 GB A100 GPU:
```bash
srun --job-name=lpezzoli_debug --qos=gpuheavy-40 --gres=gpu:40gb:1 --cpus-per-task=8 --mem=64G --pty /bin/bash
```

## 3. Mufasa Open-Source LLM Execution (vLLM on 40 GB A100)

Mufasa provides dedicated 40 GB A100 GPUs via `--qos=gpuheavy-40 --gres=gpu:40gb:1` (alongside 20 GB MIG slices on `--qos=gpu --gres=gpu:3g.20gb:1`). With 40 GB VRAM, the system supports both **Single/Hybrid 32B AWQ models** and **Simultaneous Dual-Local execution of two 14B–22B models (`combined_vram <= 26.0 GB`)** on a single GPU.

### Supported Models in `ModelRegistry` (40 GB VRAM Allocation)

| Model | Format | VRAM Footprint | Max Context | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **`Qwen/Qwen2.5-32B-Instruct-AWQ`** | 4-bit AWQ | ~19.5 GB | 32k tokens | Flagship 32B dense extraction on 40 GB A100 |
| **`casperhansen/deepseek-r1-distill-qwen-32b-awq`** | 4-bit AWQ | ~19.5 GB | 32k tokens | Flagship 32B reasoning (`<think>`) model on 40 GB A100 |
| **`TechxGenus/Codestral-22B-v0.1-AWQ`** | 4-bit AWQ | ~12.5 GB | 24k - 32k tokens | European structured data specialist (22B); co-locates with 14B models on 40 GB |
| **`Qwen/Qwen2.5-14B-Instruct-AWQ`** *(Recommended)* | 4-bit AWQ | ~10.5 GB | 24k - 32k tokens | State-of-the-art structured JSON & table extraction; co-locates in Dual-Local on 40 GB |
| **`casperhansen/deepseek-r1-distill-qwen-14b-awq`** | 4-bit AWQ | ~9.5 GB | 24k - 32k tokens | Reasoning model with `<think>` traces; co-locates with Qwen-14B/Codestral-22B on 40 GB |
| **`solidrust/Mistral-7B-Instruct-v0.3-AWQ`** | 4-bit AWQ | ~4.2 GB | 32k tokens | Ultra-fast European sovereign model |
| **`Qwen/Qwen2.5-7B-Instruct-AWQ`** | 4-bit AWQ | ~5.7 GB | 32k tokens | High-speed local consensus worker |
| **`hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`** | 4-bit AWQ | ~5.5 GB | 32k tokens | US open-weights baseline |

---

### Step-by-Step Workflow on Mufasa

#### 1. Download Model Weights
Run on Mufasa (saves into shared cache `~/lpezzoli/.cache/huggingface/hub/`):
```bash
ssh amoglia@10.79.23.96
cd ~/lpezzoli/AgentForGMMIS
bash scripts/mufasa/download_open_models.sh qwen32b
```

#### 2. Run Autonomous Unified Benchmark via SLURM
Submits an autonomous job that selects the cleanest 40 GB A100 GPU, launches vLLM server(s), runs the benchmark, and cleans up:
```bash
sbatch --export=ALL,MODEL_A=qwen2.5-14b-awq,MODEL_B=deepseek-r1-14b-awq,COUNT=40 scripts/mufasa/run_unified_benchmark_mufasa.sh
```
Monitor progress:
```bash
squeue -u amoglia
tail -f logs/mufasa_unified_<JOBID>.out
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