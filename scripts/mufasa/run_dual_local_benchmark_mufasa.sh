#!/bin/bash
# ==============================================================================
# MUFASA HPC — 100% On-Premise Dual-Agent Local Consensus Benchmark Job
# Agent A: Qwen 2.5 7B Instruct AWQ (vLLM on port 8000)
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

MODEL_A_DEFAULT="Qwen/Qwen2.5-7B-Instruct-AWQ"
MODEL_B_DEFAULT="solidrust/Mistral-7B-Instruct-v0.3-AWQ"

if [[ "$1" =~ ^[0-9]+$ ]]; then
    COUNT=${1:-40}
    MAX_ROUNDS=${2:-3}
    MODEL_A=${3:-${MODEL_A:-"$MODEL_A_DEFAULT"}}
    MODEL_B=${4:-${MODEL_B:-"$MODEL_B_DEFAULT"}}
    EXTRA_ARGS=("${@:5}")
elif [ -n "$1" ]; then
    MODEL_A=$1
    MODEL_B=${2:-${MODEL_B:-"$MODEL_B_DEFAULT"}}
    COUNT=${3:-40}
    MAX_ROUNDS=${4:-3}
    EXTRA_ARGS=("${@:5}")
else
    COUNT=40
    MAX_ROUNDS=3
    MODEL_A=${MODEL_A:-"$MODEL_A_DEFAULT"}
    MODEL_B=${MODEL_B:-"$MODEL_B_DEFAULT"}
    EXTRA_ARGS=()
fi

PORT_A=8000
PORT_B=8001

# Chat templates for Mistral/Codestral if required by transformers/vLLM
TEMPLATE_ARG_A=""
if [[ "$MODEL_A" =~ [Mm]istral || "$MODEL_A" =~ [Cc]odestral ]]; then
    if [ -f "scripts/mufasa/mistral_chat_template.jinja" ]; then
        TEMPLATE_ARG_A="--chat-template scripts/mufasa/mistral_chat_template.jinja"
    fi
fi

TEMPLATE_ARG_B=""
if [[ "$MODEL_B" =~ [Mm]istral || "$MODEL_B" =~ [Cc]odestral ]]; then
    if [ -f "scripts/mufasa/mistral_chat_template.jinja" ]; then
        TEMPLATE_ARG_B="--chat-template scripts/mufasa/mistral_chat_template.jinja"
    fi
fi

CLEAN_A=$(basename "$MODEL_A" | sed -E 's/(Instruct|v0\.[0-9]|-AWQ|-INT4)//g; s/[^a-zA-Z0-9]//g')
CLEAN_B=$(basename "$MODEL_B" | sed -E 's/(Instruct|v0\.[0-9]|-AWQ|-INT4)//g; s/[^a-zA-Z0-9]//g')
TAG="mufasa_dual_local_${CLEAN_A}_${CLEAN_B}_${COUNT}"

echo "=================================================================="
echo "🤝 AGENT FOR GMMIS — 100% LOCAL DUAL-AGENT BENCHMARK ON MUFASA"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "Agent A:      Mufasa GPU (port $PORT_A) -> $MODEL_A"
echo "Agent B:      Mufasa GPU (port $PORT_B) -> $MODEL_B"
echo "Paper Count:  $COUNT"
echo "Max Rounds:   $MAX_ROUNDS"
echo "Run Tag:      $TAG"
echo "=================================================================="

nvidia-smi

export VLLM_USE_V1=0

cleanup() {
    echo "🛑 Shutting down vLLM servers..."
    [ -n "$VLLM_PID_A" ] && kill -9 "$VLLM_PID_A" 2>/dev/null || true
    [ -n "$VLLM_PID_B" ] && kill -9 "$VLLM_PID_B" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 1. Start vLLM Server A (port 8000)
echo "Starting vLLM Server A on port $PORT_A for $MODEL_A..."
vllm serve "$MODEL_A" \
    --port "$PORT_A" \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.43 \
    --enforce-eager \
    $TEMPLATE_ARG_A \
    --trust-remote-code > "logs/vllm_server_a_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID_A=$!

# Wait for Server A
echo "Waiting for vLLM Server A to be ready on port $PORT_A..."
MAX_WAIT=240
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

# 2. Start vLLM Server B (port 8001)
echo "Starting vLLM Server B on port $PORT_B for $MODEL_B..."
vllm serve "$MODEL_B" \
    --port "$PORT_B" \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.43 \
    --enforce-eager \
    $TEMPLATE_ARG_B \
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
    --tag "$TAG" \
    "${EXTRA_ARGS[@]}"

echo "=================================================================="
echo "✅ Local Dual-Agent Benchmark Completed on $(date)"
echo "=================================================================="
