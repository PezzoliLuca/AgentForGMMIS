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

MODEL_NAME=${1:-"Qwen/Qwen2.5-14B-Instruct-AWQ"}
PORT=8000
COUNT=${2:-15}

echo "=================================================================="
echo "🚀 AGENT FOR GMMIS — MUFASA HPC BENCHMARK"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Model:     $MODEL_NAME"
echo "Paper Count: $COUNT"
echo "=================================================================="

nvidia-smi

# 1. Start vLLM background server
echo "Starting vLLM server on port $PORT..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --port "$PORT" \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code &
VLLM_PID=$!

# Trap signals to ensure background vLLM is killed on exit/error
trap "kill -9 $VLLM_PID 2>/dev/null" EXIT INT TERM

# 2. Wait for vLLM to initialize (health check loop)
echo "Waiting for vLLM server to be ready..."
MAX_WAIT=300
ELAPSED=0
until curl -sf http://localhost:$PORT/health > /dev/null || [ $ELAPSED -ge $MAX_WAIT ]; do
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
    --count "$COUNT"

echo "=================================================================="
echo "✅ Benchmark Completed on $(date)"
echo "=================================================================="
