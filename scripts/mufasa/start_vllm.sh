#!/bin/bash
# ==============================================================================
# MUFASA HPC — vLLM OpenAI Server Launch Script
# Starts a local high-throughput OpenAI-compatible server on the allocated GPU.
#
# Hardware profile on Mufasa:
#   QOS: gpu
#   GRES: gpu:3g.20gb:1 (A100 MIG slice, 20 GB VRAM)
#   Memory: 32 GB RAM
# ==============================================================================

#SBATCH --job-name=lpezzoli_vllm
#SBATCH --qos=gpuheavy-40
#SBATCH --gres=gpu:40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/vllm_server_%j.out
#SBATCH --error=logs/vllm_server_%j.err

mkdir -p logs

# Activate Mufasa virtual environment
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

# Default Model: Qwen2.5-14B-Instruct-AWQ (fits in 9 GB VRAM, leaving 11 GB for 32k KV Cache)
MODEL_NAME=${1:-"Qwen/Qwen2.5-14B-Instruct-AWQ"}
PORT=${2:-8000}
MAX_LEN=${3:-32768}

echo "=========================================="
echo "Starting vLLM on node: $(hostname)"
echo "Model: $MODEL_NAME"
echo "Port:  $PORT"
echo "Context Limit: $MAX_LEN"
echo "=========================================="

export VLLM_USE_V1=0
vllm serve "$MODEL_NAME" \
    --port "$PORT" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --trust-remote-code
