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

---

## 3. Default Batch Job Script (`run_job.sh`)

Submission script pre-configured for Mufasa policies:

```bash
#!/bin/bash
#SBATCH --job-name=gmmis_job
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Create logs directory
mkdir -p logs

# Activate virtual environment
source venv/bin/activate

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "=========================================="

# Inspect allocated GPU diagnostics
nvidia-smi

# Execute Python script with unbuffered output (-u)
$SLURM_SUBMIT_DIR/venv/bin/python3 -u main.py

echo "Finished: $(date)"
```

---

## 4. Project Structure

```text
AgentForGMMIS/
├── logs/              # SLURM output (.out) and error(.err) logs

```