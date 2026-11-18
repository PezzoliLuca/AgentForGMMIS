#!/bin/bash
# ==============================================================================
# MUFASA HPC — Sequential Benchmark Queue Feeder (respects QOS MaxSubmitPU=1)
#
# Submits the remaining Blocco 1–3 benchmarks one by one as soon as the previous
# `lpezzoli_gmmis` SLURM job finishes, skipping any run already completed.
# ==============================================================================

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR" || exit 1

mkdir -p logs
export PATH="/opt/shared/sw/amd/gcc-13.3.0/slurm-25.05.4/bin:$PATH"

USER_NAME="${USER:-amoglia}"
POLL_INTERVAL=60
PID_FILE="logs/queue_feeder.pid"
echo "$$" > "$PID_FILE"

BENCHMARKS=(
  "b1_2_codestral22b_gemini_40gb_v2|codestral-22b-awq|gemini-3.1-flash-lite-preview|gemini|04:00:00"
  "b3_3_qwen32b_deepseek32b_40gb_v2|qwen2.5-32b-awq|deepseek-r1-32b-awq|mufasa|06:00:00"
  "b2_2_codestral22b_qwen14b_40gb_v2|codestral-22b-awq|qwen2.5-14b-awq|mufasa|04:00:00"
  "b2_1_qwen14b_deepseek14b_40gb_v2|qwen2.5-14b-awq|deepseek-r1-14b-awq|mufasa|04:00:00"
  "b2_3_codestral22b_deepseek14b_40gb_v2|codestral-22b-awq|deepseek-r1-14b-awq|mufasa|04:00:00"
  "b1_5_qwen7b_llama8b_40gb_v2|qwen2.5-7b-awq|llama-3.1-8b-awq|mufasa|04:00:00"
  "b1_6_qwen7b_mistral7b_40gb_v2|qwen2.5-7b-awq|mistral-7b-awq|mufasa|04:00:00"
)

log_msg() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

wait_for_free_slot() {
  local logged_wait=0
  while true; do
    local active
    if ! active="$(squeue -u "$USER_NAME" -h -n lpezzoli_gmmis 2>/dev/null)"; then
      log_msg "[WARN] squeue check failed; retrying in ${POLL_INTERVAL}s..."
      sleep "$POLL_INTERVAL"
      continue
    fi
    if [ -z "$active" ]; then
      return 0
    fi
    if [ "$logged_wait" -eq 0 ]; then
      local summary
      summary="$(echo "$active" | awk '{print $1 ":" $5}' | tr '\n' ' ')"
      log_msg "[WAIT] Waiting for active/pending SLURM job(s) to finish before submitting next benchmark: $summary"
      logged_wait=1
    fi
    sleep "$POLL_INTERVAL"
  done
}

log_msg "=== Queue Feeder started (PID=$$, user=$USER_NAME, queue=${#BENCHMARKS[@]} benchmarks) ==="

for entry in "${BENCHMARKS[@]}"; do
  IFS='|' read -r TAG MA MB PB TIME_LIMIT <<< "$entry"

  if compgen -G "logs/runs/*_${TAG}/benchmark_results.json" > /dev/null; then
    log_msg "[SKIP] $TAG already completed (benchmark_results.json found)."
    continue
  fi

  wait_for_free_slot

  log_msg "[SUBMIT] $TAG (MODEL_A=$MA, MODEL_B=$MB, PROVIDER_B=$PB, TIME=$TIME_LIMIT)"
  while true; do
    JOB_ID="$(sbatch --parsable --cpus-per-task=1 --time="$TIME_LIMIT" \
      --export=ALL,MODEL_A="$MA",MODEL_B="$MB",PROVIDER_B="$PB",COUNT=40,MAX_ROUNDS=3,TAG="$TAG" \
      scripts/mufasa/run_unified_benchmark_mufasa.sh 2>&1)"
    RC=$?
    if [ "$RC" -eq 0 ] && [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
      log_msg "[SUBMITTED] $TAG -> SLURM Job ID $JOB_ID"
      sleep 5
      break
    fi
    log_msg "[RETRY] sbatch returned rc=$RC ($JOB_ID); waiting ${POLL_INTERVAL}s before retry..."
    sleep "$POLL_INTERVAL"
  done
done

wait_for_free_slot
log_msg "=== Queue Feeder finished all ${#BENCHMARKS[@]} queued benchmarks! ==="
