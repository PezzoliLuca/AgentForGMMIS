#!/bin/bash
# ==============================================================================
# MUFASA HPC — 100% On-Premise Dual-Agent Local Consensus Benchmark Job
# Agent A: Qwen 2.5 14B Instruct AWQ (vLLM on port 8000)
# Agent B: Meta Llama 3.1 8B Instruct AWQ (vLLM on port 8001)
#
# Runs entirely on local Mufasa A100 GPU (zero external API calls, no rate limits).
# Both models run concurrently with dedicated ports.
#
# Usage:
#   sbatch scripts/mufasa/run_dual_local_benchmark_mufasa.sh [COUNT] [MAX_ROUNDS]
# Example:
#   sbatch scripts/mufasa/run_dual_local_benchmark_mufasa.sh 20 3
# ==============================================================================

#SBATCH --job-name=gmmis_local_dual
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mufasa_dual_local_%j.out
#SBATCH --error=logs/mufasa_dual_local_%j.err

mkdir -p logs

# Activate Python environment on Mufasa
if [ -f "$HOME/AgentForGMMIS/venv/bin/activate" ]; then
    source "$HOME/AgentForGMMIS/venv/bin/activate"
elif [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
fi

# Expose user-space Python C headers to GCC for Triton/CUDA extensions
export CPATH="$HOME/.local/usr/include:$HOME/.local/usr/include/python3.12:$HOME/.local/usr/include/x86_64-linux-gnu/python3.12:$CPATH"
export C_INCLUDE_PATH="$HOME/.local/usr/include:$HOME/.local/usr/include/python3.12:$HOME/.local/usr/include/x86_64-linux-gnu/python3.12:$C_INCLUDE_PATH"

MODEL_A=${MODEL_A:-"Qwen/Qwen2.5-14B-Instruct-AWQ"}
MODEL_B=${MODEL_B:-"hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"}
PORT_A=8000
PORT_B=8001
COUNT=${1:-20}
MAX_ROUNDS=${2:-3}

echo "=================================================================="
echo "🤝 AGENT FOR GMMIS — 100% LOCAL DUAL-AGENT BENCHMARK ON MUFASA"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "Agent A:      Mufasa GPU (port $PORT_A) -> $MODEL_A"
echo "Agent B:      Mufasa GPU (port $PORT_B) -> $MODEL_B"
echo "Paper Count:  $COUNT"
echo "Max Rounds:   $MAX_ROUNDS"
echo "=================================================================="

nvidia-smi

export VLLM_USE_V1=0

# 1. Start vLLM Server A (Qwen 14B AWQ)
echo "Starting vLLM Server A on port $PORT_A for $MODEL_A..."
vllm serve "$MODEL_A" \
    --port "$PORT_A" \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.54 \
    --enforce-eager \
    --trust-remote-code > "logs/vllm_server_a_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID_A=$!

# Trap signals to ensure both servers are cleanly killed on exit/error
trap "kill -9 $VLLM_PID_A $VLLM_PID_B 2>/dev/null" EXIT INT TERM

# Wait for Server A
echo "Waiting for vLLM Server A to be ready on port $PORT_A..."
MAX_WAIT=180
ELAPSED=0
until curl -sf "http://localhost:$PORT_A/health" > /dev/null || [ $ELAPSED -ge $MAX_WAIT ]; do
    if ! kill -0 $VLLM_PID_A 2>/dev/null; then
        echo "❌ Server A process died during startup. Check logs/vllm_server_a_${SLURM_JOB_ID}.log"
        cat "logs/vllm_server_a_${SLURM_JOB_ID}.log" | tail -n 30
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "   ... waiting for Server A ($ELAPSED / $MAX_WAIT seconds)"
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "❌ Error: Server A failed to start within $MAX_WAIT seconds."
    exit 1
fi
echo "✅ Server A is LIVE on port $PORT_A!"

# 2. Start vLLM Server B (Llama 3.1 8B AWQ)
echo "Starting vLLM Server B on port $PORT_B for $MODEL_B..."
vllm serve "$MODEL_B" \
    --port "$PORT_B" \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.34 \
    --enforce-eager \
    --trust-remote-code > "logs/vllm_server_b_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID_B=$!

# Wait for Server B
echo "Waiting for vLLM Server B to be ready on port $PORT_B..."
ELAPSED=0
until curl -sf "http://localhost:$PORT_B/health" > /dev/null || [ $ELAPSED -ge $MAX_WAIT ]; do
    if ! kill -0 $VLLM_PID_B 2>/dev/null; then
        echo "❌ Server B process died during startup. Check logs/vllm_server_b_${SLURM_JOB_ID}.log"
        cat "logs/vllm_server_b_${SLURM_JOB_ID}.log" | tail -n 30
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "   ... waiting for Server B ($ELAPSED / $MAX_WAIT seconds)"
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "❌ Error: Server B failed to start within $MAX_WAIT seconds."
    exit 1
fi
echo "✅ Server B is LIVE on port $PORT_B!"

echo "🚀 Both local vLLM servers are live and healthy! Starting Dual-Agent Benchmark..."

# 3. Run Agent For GMMIS Batch Benchmark in 100% Local Dual-Agent Mode
python3 -u scripts/run_batch_benchmark.py \
    --provider-a mufasa \
    --model-a "$MODEL_A" \
    --base-url-a "http://localhost:$PORT_A/v1" \
    --provider-b mufasa \
    --model-b "$MODEL_B" \
    --base-url-b "http://localhost:$PORT_B/v1" \
    --count "$COUNT" \
    --max-rounds "$MAX_ROUNDS" \
    --tag "mufasa_dual_local_${COUNT}" \
    "${@:3}"

echo "=================================================================="
echo "✅ Local Dual-Agent Benchmark Completed on $(date)"
echo "=================================================================="
