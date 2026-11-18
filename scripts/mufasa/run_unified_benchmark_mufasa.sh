#!/bin/bash
# ==============================================================================
# MUFASA HPC — Unified Multi-LLM Benchmark Orchestration Script
#
# Automatically resolves model parameters, VRAM allocation, and execution topology:
#   1. Single Agent (Local vLLM or Cloud)
#   2. Hybrid Dual-Agent (Local vLLM + Gemini Cloud)
#   3. Simultaneous Dual-Agent (2 concurrent local vLLM instances, VRAM <= 12.5GB)
#   4. 2-Stage Sequential Dual-Agent (vLLM A -> vLLM B, VRAM > 12.5GB on 20GB MIG)
#
# Usage:
#   sbatch scripts/mufasa/run_unified_benchmark_mufasa.sh [COUNT] [MAX_ROUNDS] [MODEL_A] [MODEL_B] [EXTRA_ARGS...]
# Or via environment variables:
#   sbatch --export=ALL,MODEL_A=deepseek-r1-14b-awq,MODEL_B=qwen2.5-14b-awq,COUNT=40 scripts/mufasa/run_unified_benchmark_mufasa.sh
# ==============================================================================

#SBATCH --job-name=lpezzoli_gmmis
#SBATCH --qos=gpuheavy-40
#SBATCH --gres=gpu:40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mufasa_unified_%j.out
#SBATCH --error=logs/mufasa_unified_%j.err

mkdir -p logs

# Ensure slurm binaries are in PATH on Mufasa compute nodes
export PATH="/opt/shared/sw/amd/gcc-13.3.0/slurm-25.05.4/bin:$PATH"

# 1. Activate Python virtual environment on Mufasa (isolated under ~/lpezzoli)
if [ -f "$HOME/lpezzoli/AgentForGMMIS/venv/bin/activate" ]; then
    source "$HOME/lpezzoli/AgentForGMMIS/venv/bin/activate"
elif [ -f "$HOME/AgentForGMMIS/venv/bin/activate" ]; then
    source "$HOME/AgentForGMMIS/venv/bin/activate"
elif [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
fi

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

# Isolate HuggingFace, vLLM, and Triton caches under ~/lpezzoli/.cache
if [ -d "$HOME/lpezzoli/.cache" ]; then
    export HF_HOME="$HOME/lpezzoli/.cache/huggingface"
    export VLLM_CACHE_DIR="$HOME/lpezzoli/.cache/vllm"
    export TRITON_CACHE_DIR="$HOME/lpezzoli/.cache/triton"
fi

# Load GEMINI_API_KEY from .env if present
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Expose user-space Python C headers for Triton / CUDA vLLM extensions
LOCAL_INC_BASE="$HOME/lpezzoli/.local/usr/include"
[ ! -d "$LOCAL_INC_BASE" ] && LOCAL_INC_BASE="$HOME/.local/usr/include"
export CPATH="$LOCAL_INC_BASE:$LOCAL_INC_BASE/python3.12:$LOCAL_INC_BASE/x86_64-linux-gnu/python3.12:$CPATH"
export C_INCLUDE_PATH="$LOCAL_INC_BASE:$LOCAL_INC_BASE/python3.12:$LOCAL_INC_BASE/x86_64-linux-gnu/python3.12:$C_INCLUDE_PATH"
export VLLM_USE_V1=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Auto-select the cleanest 40GB A100 GPU if SLURM exposes multiple cards
if command -v nvidia-smi &>/dev/null; then
    if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] || [[ "${CUDA_VISIBLE_DEVICES}" == *","* ]]; then
        BEST_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t',' -k2 -nr | head -n 1 | awk -F',' '{print $1}' | tr -d ' ')
        export CUDA_VISIBLE_DEVICES="${BEST_GPU:-0}"
    fi
fi

# 2. Argument Parsing (support both CLI positional arguments and environment variables)
if [[ "$1" =~ ^[0-9]+$ ]]; then
    COUNT=${1:-40}
    MAX_ROUNDS=${2:-3}
    MODEL_A=${3:-${MODEL_A:-"qwen2.5-7b-awq"}}
    MODEL_B=${4:-${MODEL_B:-""}}
    EXTRA_ARGS=("${@:5}")
elif [ -n "$1" ] && [[ "$1" != --* ]]; then
    MODEL_A=$1
    MODEL_B=${2:-${MODEL_B:-""}}
    COUNT=${3:-${COUNT:-40}}
    MAX_ROUNDS=${4:-${MAX_ROUNDS:-3}}
    EXTRA_ARGS=("${@:5}")
else
    COUNT=${COUNT:-40}
    MAX_ROUNDS=${MAX_ROUNDS:-3}
    MODEL_A=${MODEL_A:-"qwen2.5-7b-awq"}
    MODEL_B=${MODEL_B:-""}
    EXTRA_ARGS=("$@")
fi

MODEL_A=$(echo "$MODEL_A" | sed -e 's/^["'\'' ]*//' -e 's/["'\'' ]*$//')
MODEL_B=$(echo "$MODEL_B" | sed -e 's/^["'\'' ]*//' -e 's/["'\'' ]*$//')
PROVIDER_A=$(echo "${PROVIDER_A:-}" | sed -e 's/^["'\'' ]*//' -e 's/["'\'' ]*$//')
PROVIDER_B=$(echo "${PROVIDER_B:-}" | sed -e 's/^["'\'' ]*//' -e 's/["'\'' ]*$//')
COMPARE_SOURCE=${COMPARE_SOURCE:-"primary"}

# 3. Dynamic Plan Resolution via ModelRegistry
PLAN_ARGS=(--plan-bash --model-a "$MODEL_A")
[ -n "$PROVIDER_A" ] && PLAN_ARGS+=(--provider-a "$PROVIDER_A")
[ -n "$MODEL_B" ] && PLAN_ARGS+=(--model-b "$MODEL_B")
[ -n "$PROVIDER_B" ] && PLAN_ARGS+=(--provider-b "$PROVIDER_B")

echo "📐 Querying Model Registry Plan: python3 -m agent_gmmis.llm.registry ${PLAN_ARGS[*]}"
eval "$(python3 -m agent_gmmis.llm.registry "${PLAN_ARGS[@]}")"
export MUFASA_MAX_MODEL_LEN="${MAX_LEN_A:-24576}"

# Check auto-installation of bitsandbytes if requested by any selected model
if [ "$NEEDS_BITSANDBYTES" = "true" ] || [[ "$VLLM_ARGS_A $VLLM_ARGS_B" == *"bitsandbytes"* ]]; then
    echo "📦 Checking bitsandbytes installation for dynamic 4-bit quantization on Mufasa..."
    python3 -c "import bitsandbytes" 2>/dev/null || pip install -q "bitsandbytes>=0.45.0"
fi

if [ -z "$TAG" ]; then
    if [ -n "$MODEL_B_KEY" ]; then
        TAG="mufasa_${TOPOLOGY}_${MODEL_A_KEY}_${MODEL_B_KEY}_${COUNT}"
    else
        TAG="mufasa_${TOPOLOGY}_${MODEL_A_KEY}_${COUNT}"
    fi
fi

echo "=================================================================="
echo "🔬 AGENT FOR GMMIS — UNIFIED MUFASA BENCHMARK"
echo "Job ID:        ${SLURM_JOB_ID:-manual}"
echo "Node:          $(hostname)"
echo "Topology:      $TOPOLOGY"
echo "Agent A:       $PROVIDER_A ($MODEL_A_REPO)"
if [ -n "$MODEL_B_REPO" ]; then
    echo "Agent B:       $PROVIDER_B ($MODEL_B_REPO)"
fi
echo "Paper Count:   $COUNT"
echo "Max Rounds:    $MAX_ROUNDS"
echo "Ground Truth:  $COMPARE_SOURCE"
echo "Run Tag:       $TAG"
echo "=================================================================="

nvidia-smi

# Process cleanup handler (kills parent vLLM API server + VLLM::EngineCore children)
cleanup() {
    echo "🛑 Shutting down active vLLM servers..."
    [ -n "$VLLM_PID_A" ] && pkill -9 -P "$VLLM_PID_A" 2>/dev/null || true
    [ -n "$VLLM_PID_A" ] && kill -9 "$VLLM_PID_A" 2>/dev/null || true
    [ -n "$VLLM_PID_B" ] && pkill -9 -P "$VLLM_PID_B" 2>/dev/null || true
    [ -n "$VLLM_PID_B" ] && kill -9 "$VLLM_PID_B" 2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore|vllm serve" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for_server() {
    local port=$1
    local pid=$2
    local name=$3
    local max_wait=900
    local elapsed=0
    echo "Waiting for $name to be ready on port $port (PID: $pid)..."
    until curl -sf "http://localhost:$port/health" > /dev/null || [ $elapsed -ge $max_wait ]; do
        if ! kill -0 $pid 2>/dev/null; then
            echo "❌ $name (PID $pid) died during startup. Check vLLM logs."
            return 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        echo "   ... waiting for $name ($elapsed / $max_wait s)"
    done
    if [ $elapsed -ge $max_wait ]; then
        echo "❌ $name failed to start within $max_wait seconds."
        return 1
    fi
    echo "✅ $name is LIVE on port $port!"
    return 0
}

# ==============================================================================
# 4. EXECUTION TOPOLOGY BRANCHING
# ==============================================================================

if [ "$TOPOLOGY" = "single" ]; then
    if [ "$PROVIDER_A" = "mufasa" ]; then
        LOG_A="logs/vllm_server_a_${SLURM_JOB_ID:-manual}.log"
        echo "🚀 Starting vLLM Server A on port $PORT_A: vllm serve $VLLM_ARGS_A"
        vllm serve $VLLM_ARGS_A > "$LOG_A" 2>&1 &
        VLLM_PID_A=$!
        wait_for_server "$PORT_A" "$VLLM_PID_A" "Server A" || exit 1
        BASE_URL_A="http://localhost:$PORT_A/v1"
    else
        BASE_URL_A=""
    fi

    export MUFASA_MAX_MODEL_LEN="$MAX_LEN_A"
    python3 -u scripts/run_batch_benchmark.py \
        --provider-a "$PROVIDER_A" \
        --model-a "$MODEL_A_REPO" \
        ${BASE_URL_A:+--base-url-a "$BASE_URL_A"} \
        --compare-source "$COMPARE_SOURCE" \
        --count "$COUNT" \
        --tag "$TAG" \
        "${EXTRA_ARGS[@]}"

elif [ "$TOPOLOGY" = "hybrid" ]; then
    if [ "$PROVIDER_A" = "mufasa" ]; then
        LOG_A="logs/vllm_server_a_${SLURM_JOB_ID:-manual}.log"
        echo "🚀 Starting vLLM Server A on port $PORT_A: vllm serve $VLLM_ARGS_A"
        vllm serve $VLLM_ARGS_A > "$LOG_A" 2>&1 &
        VLLM_PID_A=$!
        wait_for_server "$PORT_A" "$VLLM_PID_A" "Server A" || exit 1
        BASE_URL_A="http://localhost:$PORT_A/v1"
        export MUFASA_MAX_MODEL_LEN="$MAX_LEN_A"
    else
        BASE_URL_A=""
    fi

    if [ "$PROVIDER_B" = "mufasa" ]; then
        LOG_B="logs/vllm_server_b_${SLURM_JOB_ID:-manual}.log"
        echo "🚀 Starting vLLM Server B on port $PORT_B: vllm serve $VLLM_ARGS_B"
        vllm serve $VLLM_ARGS_B > "$LOG_B" 2>&1 &
        VLLM_PID_B=$!
        wait_for_server "$PORT_B" "$VLLM_PID_B" "Server B" || exit 1
        BASE_URL_B="http://localhost:$PORT_B/v1"
        export MUFASA_MAX_MODEL_LEN="$MAX_LEN_B"
    else
        BASE_URL_B=""
    fi

    python3 -u scripts/run_batch_benchmark.py \
        --provider-a "$PROVIDER_A" \
        --model-a "$MODEL_A_REPO" \
        ${BASE_URL_A:+--base-url-a "$BASE_URL_A"} \
        --provider-b "$PROVIDER_B" \
        --model-b "$MODEL_B_REPO" \
        ${BASE_URL_B:+--base-url-b "$BASE_URL_B"} \
        --compare-source "$COMPARE_SOURCE" \
        --count "$COUNT" \
        --max-rounds "$MAX_ROUNDS" \
        --tag "$TAG" \
        "${EXTRA_ARGS[@]}"

elif [ "$TOPOLOGY" = "dual_simultaneous" ]; then
    LOG_A="logs/vllm_server_a_${SLURM_JOB_ID:-manual}.log"
    LOG_B="logs/vllm_server_b_${SLURM_JOB_ID:-manual}.log"

    echo "🚀 Starting concurrent Server A on port $PORT_A..."
    vllm serve $VLLM_ARGS_A > "$LOG_A" 2>&1 &
    VLLM_PID_A=$!
    wait_for_server "$PORT_A" "$VLLM_PID_A" "Server A" || exit 1

    echo "🚀 Starting concurrent Server B on port $PORT_B..."
    vllm serve $VLLM_ARGS_B > "$LOG_B" 2>&1 &
    VLLM_PID_B=$!
    wait_for_server "$PORT_B" "$VLLM_PID_B" "Server B" || exit 1

    export MUFASA_MAX_MODEL_LEN=$(( MAX_LEN_A < MAX_LEN_B ? MAX_LEN_A : MAX_LEN_B ))
    echo "🤝 Both local servers live (MUFASA_MAX_MODEL_LEN=$MUFASA_MAX_MODEL_LEN)! Running simultaneous consensus benchmark..."
    python3 -u scripts/run_batch_benchmark.py \
        --provider-a mufasa \
        --model-a "$MODEL_A_REPO" \
        --base-url-a "http://localhost:$PORT_A/v1" \
        --provider-b mufasa \
        --model-b "$MODEL_B_REPO" \
        --base-url-b "http://localhost:$PORT_B/v1" \
        --compare-source "$COMPARE_SOURCE" \
        --count "$COUNT" \
        --max-rounds "$MAX_ROUNDS" \
        --tag "$TAG" \
        "${EXTRA_ARGS[@]}"

elif [ "$TOPOLOGY" = "dual_sequential" ]; then
    STAGE_CACHE="logs/stage_cache/${TAG}_${SLURM_JOB_ID:-manual}"
    mkdir -p "$STAGE_CACHE"

    echo "=================================================================="
    echo "⚡ 2-STAGE SEQUENTIAL DUAL-LOCAL BENCHMARK (20GB MIG SLICE)"
    echo "Stage Cache: $STAGE_CACHE"
    echo "=================================================================="

    # --- STAGE 1: Launch Server A with 90% GPU, run Agent A extractions ---
    LOG_A="logs/vllm_server_a_${SLURM_JOB_ID:-manual}.log"
    echo "🚀 [Stage 1/2] Starting Server A on port $PORT_A with 90% VRAM slice..."
    vllm serve $VLLM_ARGS_A > "$LOG_A" 2>&1 &
    VLLM_PID_A=$!
    wait_for_server "$PORT_A" "$VLLM_PID_A" "Server A" || exit 1

    echo "📄 Running Stage 1 Extractions (Agent A, MUFASA_MAX_MODEL_LEN=$MAX_LEN_A)..."
    export MUFASA_MAX_MODEL_LEN="$MAX_LEN_A"
    python3 -u scripts/run_batch_benchmark.py \
        --provider-a mufasa \
        --model-a "$MODEL_A_REPO" \
        --base-url-a "http://localhost:$PORT_A/v1" \
        --provider-b mufasa \
        --model-b "$MODEL_B_REPO" \
        --count "$COUNT" \
        --tag "$TAG" \
        --staged-mode stage1-extract-a \
        --stage-cache-dir "$STAGE_CACHE" \
        "${EXTRA_ARGS[@]}"

    echo "🛑 Shutting down Server A and EngineCore workers to free GPU VRAM for Stage 2..."
    pkill -9 -P "$VLLM_PID_A" 2>/dev/null || true
    kill -9 "$VLLM_PID_A" 2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore|vllm serve" 2>/dev/null || true
    unset VLLM_PID_A
    sleep 10
    nvidia-smi

    # --- STAGE 2: Launch Server B with 90% GPU, run Agent B + Deliberation ---
    LOG_B="logs/vllm_server_b_${SLURM_JOB_ID:-manual}.log"
    echo "🚀 [Stage 2/2] Starting Server B on port $PORT_B with 90% VRAM slice..."
    vllm serve $VLLM_ARGS_B > "$LOG_B" 2>&1 &
    VLLM_PID_B=$!
    wait_for_server "$PORT_B" "$VLLM_PID_B" "Server B" || exit 1

    echo "🤝 Running Stage 2 Extractions, Reconciliations & Auditing (Agent B, MUFASA_MAX_MODEL_LEN=$MAX_LEN_B)..."
    export MUFASA_MAX_MODEL_LEN="$MAX_LEN_B"
    python3 -u scripts/run_batch_benchmark.py \
        --provider-a mufasa \
        --model-a "$MODEL_A_REPO" \
        --provider-b mufasa \
        --model-b "$MODEL_B_REPO" \
        --base-url-b "http://localhost:$PORT_B/v1" \
        --compare-source "$COMPARE_SOURCE" \
        --count "$COUNT" \
        --max-rounds "$MAX_ROUNDS" \
        --tag "$TAG" \
        --staged-mode stage2-extract-b-and-reconcile \
        --stage-cache-dir "$STAGE_CACHE" \
        "${EXTRA_ARGS[@]}"

    echo "🧹 Cleaning up intermediate stage cache..."
    rm -rf "$STAGE_CACHE"
fi

echo "=================================================================="
echo "🎉 Unified Benchmark Completed on $(date)"
echo "=================================================================="
