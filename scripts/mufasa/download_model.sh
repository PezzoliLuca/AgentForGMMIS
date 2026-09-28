#!/bin/bash
# ==============================================================================
# MUFASA HPC — Model Weights Pre-Download Script
# Pre-caches weights from Hugging Face into ~/.cache/huggingface/
#
# Usage on Mufasa (can be run on login node or via interactive session):
#   bash scripts/mufasa/download_model.sh [MODEL_NAME]
# ==============================================================================

set -e

source ~/AgentForGMMIS/venv/bin/activate

# Install requirements if not already present
pip install -q huggingface_hub vllm

MODEL_NAME=${1:-"Qwen/Qwen2.5-14B-Instruct-AWQ"}

echo "=========================================="
echo "📥 Downloading Model Weights to Mufasa Cache"
echo "Model: $MODEL_NAME"
echo "=========================================="

python3 -c "
import sys
from huggingface_hub import snapshot_download
model = sys.argv[1]
print(f'Downloading {model} using memory-safe streaming...')
snapshot_download(repo_id=model, max_workers=2)
" "$MODEL_NAME"

echo "=========================================="
echo "✅ Download complete! Model is ready for vLLM."
echo "=========================================="
