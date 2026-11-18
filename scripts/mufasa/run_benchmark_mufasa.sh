#!/bin/bash
# ==============================================================================
# MUFASA HPC — All-in-One Autonomous Batch Benchmark Job
# Allocates GPU -> Starts vLLM -> Runs Batch Benchmark -> Shuts Down
# ==============================================================================

#SBATCH --job-name=lpezzoli_bench
#SBATCH --qos=gpuheavy-40
#SBATCH --gres=gpu:40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mufasa_bench_%j.out
#SBATCH --error=logs/mufasa_bench_%j.err

mkdir -p logs

# Activate Python environment on Mufasa
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

# Expose user-space Python C headers to GCC for Triton/CUDA extensions
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
COUNT=${2:-15}

# Determine max model context length based on model size / architecture to prevent KV cache OOM
if [[ "$MODEL_NAME" =~ [Cc]odestral || "$MODEL_NAME" =~ 2[0-9][Bb] ]]; then
    DEFAULT_MAX_LEN=24576
else
    DEFAULT_MAX_LEN=32768
fi
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$DEFAULT_MAX_LEN}

echo "=================================================================="
echo "🚀 AGENT FOR GMMIS — MUFASA HPC BENCHMARK"
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $(hostname)"
echo "Model:       $MODEL_NAME (max-len: $MAX_MODEL_LEN)"
echo "Paper Count: $COUNT"
echo "=================================================================="

nvidia-smi

# 1. Start vLLM background server
export VLLM_USE_V1=0
TEMPLATE_ARGS=()
if [[ "$MODEL_NAME" =~ [Cc]odestral && -f "scripts/mufasa/mistral_chat_template.jinja" ]]; then
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

# Trap signals to ensure background vLLM is killed on exit/error
trap "kill -9 $VLLM_PID 2>/dev/null" EXIT INT TERM

# 2. Wait for vLLM to initialize (health check loop)
echo "Waiting for vLLM server to be ready..."
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

echo "✅ vLLM Server is LIVE and healthy! Starting Benchmark..."

# 3. Run Agent For GMMIS Batch Benchmark targeting Mufasa local endpoint
python3 -u scripts/run_batch_benchmark.py \
    --provider mufasa \
    --model "$MODEL_NAME" \
    --count "$COUNT" \
    "${@:3}"

echo "=================================================================="
echo "✅ Benchmark Completed on $(date)"
echo "=================================================================="
