#!/bin/bash
# ==============================================================================
# MUFASA HPC — ScoutAgent 40-Abstract Gold Standard Evaluation Job
# Allocates GPU -> Starts vLLM -> Runs evaluate_scout.py -> Shuts Down
# ==============================================================================

#SBATCH --job-name=lpezzoli_scout
#SBATCH --qos=gpuheavy-40
#SBATCH --gres=gpu:40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/mufasa_scout_%j.out
#SBATCH --error=logs/mufasa_scout_%j.err

mkdir -p logs

if [ -f "$HOME/lpezzoli/AgentForGMMIS/venv/bin/activate" ]; then
    source "$HOME/lpezzoli/AgentForGMMIS/venv/bin/activate"
elif [ -f "$HOME/AgentForGMMIS/venv/bin/activate" ]; then
    source "$HOME/AgentForGMMIS/venv/bin/activate"
fi

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
if [ -d "$HOME/lpezzoli/.cache" ]; then
    export HF_HOME="$HOME/lpezzoli/.cache/huggingface"
    export VLLM_CACHE_DIR="$HOME/lpezzoli/.cache/vllm"
    export TRITON_CACHE_DIR="$HOME/lpezzoli/.cache/triton"
fi

LOCAL_INC_BASE="$HOME/lpezzoli/.local/usr/include"
[ ! -d "$LOCAL_INC_BASE" ] && LOCAL_INC_BASE="$HOME/.local/usr/include"
export CPATH="$LOCAL_INC_BASE:$LOCAL_INC_BASE/python3.12:$LOCAL_INC_BASE/x86_64-linux-gnu/python3.12:$CPATH"
export C_INCLUDE_PATH="$LOCAL_INC_BASE:$LOCAL_INC_BASE/python3.12:$LOCAL_INC_BASE/x86_64-linux-gnu/python3.12:$C_INCLUDE_PATH"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if command -v nvidia-smi &>/dev/null; then
    if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] || [[ "${CUDA_VISIBLE_DEVICES}" == *","* ]]; then
        BEST_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t',' -k2 -nr | head -n 1 | awk -F',' '{print $1}' | tr -d ' ')
        export CUDA_VISIBLE_DEVICES="${BEST_GPU:-0}"
    fi
fi

MODEL_NAME=${1:-"Qwen/Qwen2.5-14B-Instruct-AWQ"}
PORT=8000

if [[ "$MODEL_NAME" =~ [Cc]odestral || "$MODEL_NAME" =~ 2[0-9][Bb] ]]; then
    DEFAULT_MAX_LEN=24576
else
    DEFAULT_MAX_LEN=32768
fi
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$DEFAULT_MAX_LEN}

echo "=================================================================="
echo "🔎 AGENT FOR GMMIS — MUFASA SCOUT AGENT 40-ABSTRACT BENCHMARK"
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $(hostname)"
echo "Model:       $MODEL_NAME (max-len: $MAX_MODEL_LEN)"
echo "=================================================================="

export VLLM_USE_V1=0
TEMPLATE_ARGS=()
if [[ ( "$MODEL_NAME" =~ [Cc]odestral || "$MODEL_NAME" =~ [Mm]istral ) && -f "scripts/mufasa/mistral_chat_template.jinja" ]]; then
    TEMPLATE_ARGS+=(--chat-template "scripts/mufasa/mistral_chat_template.jinja")
fi

echo "Starting vLLM server on port $PORT..."
vllm serve "$MODEL_NAME" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --trust-remote-code \
    "${TEMPLATE_ARGS[@]}" &
VLLM_PID=$!

trap "kill -9 $VLLM_PID 2>/dev/null" EXIT INT TERM

MAX_WAIT=300
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

echo "✅ vLLM Server is LIVE! Running ScoutAgent 40-Abstract Benchmark..."

python3 -u scripts/evaluate_scout.py \
    --provider mufasa \
    --model "$MODEL_NAME" \
    --base-url "http://localhost:$PORT/v1"

echo "=================================================================="
echo "✅ ScoutAgent Benchmark Completed on $(date)"
echo "=================================================================="

