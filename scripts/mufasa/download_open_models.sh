#!/bin/bash
# ==============================================================================
# MUFASA HPC — Open-Weights Multi-Model Pre-Download Script
# Pre-caches weights from Hugging Face into ~/.cache/huggingface/hub/
#
# Supported Models (optimized for Mufasa A100 20GB MIG / 32k context):
#   - qwen     : Qwen/Qwen2.5-14B-Instruct-AWQ (~9 GB VRAM)
#   - llama    : casperhansen/llama-3.1-8b-instruct-awq (~5.5 GB VRAM)
#   - deepseek : casperhansen/deepseek-r1-distill-qwen-14b-awq (~9 GB VRAM)
#   - glm      : THUDM/glm-4-9b-chat (~18 GB VRAM)
#   - kimi     : moonshotai/Moonlight-16B-A3B-Instruct (Moonshot AI open MoE, ~15 GB VRAM)
#   - all      : Downloads all 5 families sequentially
#
# Usage:
#   bash scripts/mufasa/download_open_models.sh [qwen|llama|deepseek|glm|kimi|all|<hf_repo_id>]
# ==============================================================================

set -e

# Activate Mufasa virtual environment if present
if [ -f "$HOME/AgentForGMMIS/venv/bin/activate" ]; then
    source "$HOME/AgentForGMMIS/venv/bin/activate"
elif [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
fi

# Ensure huggingface_hub is installed
python3 -c "import huggingface_hub" 2>/dev/null || pip install -q huggingface_hub

# Disable Rust-based fast-transfer library (hf-xet) which allocates large memory buffers
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

TARGET=${1:-"qwen"}

download_repo() {
    local repo_id="$1"
    local desc="$2"
    echo ""
    echo "=================================================================="
    echo "📥 Downloading $desc"
    echo "Hugging Face Repo: $repo_id"
    echo "Destination:      ~/.cache/huggingface/hub/"
    echo "=================================================================="
    python3 -c "
import sys
from huggingface_hub import snapshot_download

model = sys.argv[1]
print(f'Starting streaming download for: {model}')
snapshot_download(repo_id=model, max_workers=1)
print(f'✅ Successfully cached: {model}')
" "$repo_id"
}

case "$TARGET" in
    qwen)
        download_repo "Qwen/Qwen2.5-14B-Instruct-AWQ" "Qwen 2.5 14B AWQ (Alibaba)"
        ;;
    llama)
        download_repo "casperhansen/llama-3.1-8b-instruct-awq" "Llama 3.1 8B Instruct AWQ (Meta)"
        ;;
    deepseek)
        download_repo "casperhansen/deepseek-r1-distill-qwen-14b-awq" "DeepSeek R1 Distill Qwen 14B AWQ (DeepSeek AI)"
        ;;
    glm)
        download_repo "THUDM/glm-4-9b-chat" "GLM-4 9B Chat (Zhipu AI / Tsinghua)"
        ;;
    kimi|moonlight)
        download_repo "moonshotai/Moonlight-16B-A3B-Instruct" "Moonlight 16B A3B Instruct (Moonshot AI / Kimi Family)"
        ;;
    all)
        echo "🚀 Downloading ALL 5 Model Families for the Thesis Comparative Benchmark..."
        download_repo "Qwen/Qwen2.5-14B-Instruct-AWQ" "Qwen 2.5 14B AWQ (Alibaba)"
        download_repo "casperhansen/llama-3.1-8b-instruct-awq" "Llama 3.1 8B Instruct AWQ (Meta)"
        download_repo "casperhansen/deepseek-r1-distill-qwen-14b-awq" "DeepSeek R1 Distill Qwen 14B AWQ (DeepSeek AI)"
        download_repo "THUDM/glm-4-9b-chat" "GLM-4 9B Chat (Zhipu AI / Tsinghua)"
        download_repo "moonshotai/Moonlight-16B-A3B-Instruct" "Moonlight 16B A3B Instruct (Moonshot AI / Kimi Family)"
        ;;
    *)
        # Custom Hugging Face repo ID passed directly
        download_repo "$TARGET" "Custom Model ($TARGET)"
        ;;
esac

echo ""
echo "=================================================================="
echo "🎉 Model weight acquisition completed successfully on $(date)"
echo "=================================================================="
