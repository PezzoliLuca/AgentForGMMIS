#!/bin/bash
# ==============================================================================
# MUFASA HPC — Autonomous Dual-Agent Consensus Benchmark Job
# Agent A: Sovereign On-Premise Model (e.g. Qwen 2.5 14B on A100 GPU)
# Agent B: Frontier Commercial API (Google Gemini 3.5 Flash Lite)
#
# Allocates GPU -> Starts vLLM -> Runs 20-Paper Parallel Benchmark -> Shuts Down
#
# Usage on Mufasa:
#   sbatch scripts/mufasa/run_dual_benchmark_mufasa.sh [MODEL_NAME] [PAPER_COUNT]
# Example:
#   sbatch scripts/mufasa/run_dual_benchmark_mufasa.sh Qwen/Qwen2.5-14B-Instruct-AWQ 20
# ==============================================================================

#SBATCH --job-name=gmmis_dual_bench
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mufasa_dual_bench_%j.out
#SBATCH --error=logs/mufasa_dual_bench_%j.err

mkdir -p logs

# Activate Python environment on Mufasa
if [ -f "$HOME/AgentForGMMIS/venv/bin/activate" ]; then
    source "$HOME/AgentForGMMIS/venv/bin/activate"
elif [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
fi

# Load GEMINI_API_KEY from .env if not already set in environment
if [ -z "$GEMINI_API_KEY" ] && [ -f ".env" ]; then
    export $(grep -v '^#' .env | grep -E '^GEMINI_API_KEY=' | xargs)
fi

if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️ Warning: GEMINI_API_KEY is not set. Ensure it is defined in .env or exported in your shell."
fi

# Expose user-space Python C headers to GCC for Triton/CUDA extensions
export CPATH="$HOME/.local/usr/include:$HOME/.local/usr/include/python3.12:$HOME/.local/usr/include/x86_64-linux-gnu/python3.12:$CPATH"
export C_INCLUDE_PATH="$HOME/.local/usr/include:$HOME/.local/usr/include/python3.12:$HOME/.local/usr/include/x86_64-linux-gnu/python3.12:$C_INCLUDE_PATH"

MODEL_NAME=${1:-"Qwen/Qwen2.5-14B-Instruct-AWQ"}
GEMINI_MODEL=${GEMINI_MODEL:-"gemini-3.1-flash-lite"}
PORT=8000
COUNT=${2:-20}

echo "=================================================================="
echo "🤝 AGENT FOR GMMIS — DUAL-AGENT CONSENSUS BENCHMARK ON MUFASA"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "Agent A:      Mufasa GPU vLLM -> $MODEL_NAME"
echo "Agent B:      Google AI Studio -> $GEMINI_MODEL"
echo "Paper Count:  $COUNT"
echo "=================================================================="

nvidia-smi

# 1. Start vLLM background server for Agent A
export VLLM_USE_V1=0
echo "Starting vLLM server on port $PORT for Agent A..."
vllm serve "$MODEL_NAME" \
    --port "$PORT" \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --trust-remote-code &
VLLM_PID=$!

# Trap signals to ensure background vLLM is killed on exit/error
trap "kill -9 $VLLM_PID 2>/dev/null" EXIT INT TERM

# 2. Wait for vLLM to initialize (health check loop)
echo "Waiting for vLLM server to be ready..."
MAX_WAIT=180
ELAPSED=0
until curl -sf http://localhost:$PORT/health > /dev/null || [ $ELAPSED -ge $MAX_WAIT ]; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "❌ vLLM server process died during startup."
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "   ... waiting for vLLM ($ELAPSED / $MAX_WAIT seconds)"
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "❌ Error: vLLM server failed to start within $MAX_WAIT seconds."
    exit 1
fi

echo "✅ vLLM Server is LIVE and healthy! Starting Dual-Agent Benchmark..."

# 3. Run Agent For GMMIS Batch Benchmark in Dual-Agent Consensus Mode
python3 -u scripts/run_batch_benchmark.py \
    --provider-a mufasa \
    --model-a "$MODEL_NAME" \
    --provider-b gemini \
    --model-b "$GEMINI_MODEL" \
    --count "$COUNT" \
    "${@:3}"

echo "=================================================================="
echo "✅ Dual-Agent Benchmark Completed on $(date)"
echo "=================================================================="
