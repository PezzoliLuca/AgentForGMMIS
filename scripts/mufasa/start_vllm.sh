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

#SBATCH --job-name=vllm_server
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/vllm_server_%j.out
#SBATCH --error=logs/vllm_server_%j.err

mkdir -p logs

# Activate Mufasa virtual environment
source ~/AgentForGMMIS/venv/bin/activate

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
