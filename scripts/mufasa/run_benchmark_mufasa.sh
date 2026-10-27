#!/bin/bash
# ==============================================================================
# MUFASA HPC — All-in-One Autonomous Batch Benchmark Job
# Allocates GPU -> Starts vLLM -> Runs Batch Benchmark -> Shuts Down
# ==============================================================================

#SBATCH --job-name=gmmis_bench_mufasa
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mufasa_bench_%j.out
#SBATCH --error=logs/mufasa_bench_%j.err

mkdir -p logs

# Activate Python environment on Mufasa
source ~/AgentForGMMIS/venv/bin/activate

# Expose user-space Python C headers to GCC for Triton/CUDA extensions
export CPATH="$HOME/.local/usr/include:$HOME/.local/usr/include/python3.12:$HOME/.local/usr/include/x86_64-linux-gnu/python3.12:$CPATH"
export C_INCLUDE_PATH="$HOME/.local/usr/include:$HOME/.local/usr/include/python3.12:$HOME/.local/usr/include/x86_64-linux-gnu/python3.12:$C_INCLUDE_PATH"

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
echo "Starting vLLM server on port $PORT..."
vllm serve "$MODEL_NAME" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --trust-remote-code &
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
