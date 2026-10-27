#!/usr/bin/env bash
# Monitors a running Mufasa benchmark job, streams execution logs,
# and automatically syncs all run artifacts to logs/mufasa_runs/ upon completion.
set -euo pipefail

MUFASA_HOST="${MUFASA_HOST:-10.79.23.96}"
MUFASA_USER="${MUFASA_USER:-lpezzoli}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "🔎 Checking active SLURM benchmark jobs on Mufasa (${MUFASA_USER}@${MUFASA_HOST})..."
ACTIVE_JOB=$(ssh "${MUFASA_USER}@${MUFASA_HOST}" "source /etc/profile.d/modules.sh 2>/dev/null; module load amd/slurm 2>/dev/null; squeue -u ${MUFASA_USER} -h -o '%i %j %T' | grep -E 'gmmis|bench' | head -n 1" || true)

if [ -z "${ACTIVE_JOB}" ]; then
    echo "ℹ️ No running benchmark jobs found on Mufasa. Syncing latest completed runs directly..."
    "${ROOT_DIR}/scripts/sync_mufasa_logs.sh"
    exit 0
fi

JOB_ID=$(echo "${ACTIVE_JOB}" | awk '{print $1}')
JOB_NAME=$(echo "${ACTIVE_JOB}" | awk '{print $2}')
echo "🎯 Found active job ${JOB_ID} (${JOB_NAME})."
echo "📋 Streaming live logs for SLURM Job ${JOB_ID}..."
echo "--------------------------------------------------------------------------------"

# Follow SLURM output logs until the job finishes
ssh "${MUFASA_USER}@${MUFASA_HOST}" "
    source /etc/profile.d/modules.sh 2>/dev/null
    module load amd/slurm 2>/dev/null
    
    # Wait for the SLURM log file to appear
    SLURM_LOG=\"\"
    for i in \$(seq 1 15); do
        SLURM_LOG=\$(ls -t ~/AgentForGMMIS/logs/*_${JOB_ID}.out 2>/dev/null | head -n 1 || true)
        [ -n \"\${SLURM_LOG}\" ] && break
        sleep 1
    done

    if [ -n \"\${SLURM_LOG}\" ]; then
        tail -n +1 -f \"\${SLURM_LOG}\" --pid=\$\$ &
        TAIL_PID=\$!
        while squeue -j ${JOB_ID} -h | grep -q \"${JOB_ID}\"; do
            sleep 3
        done
        # Give a moment for remaining buffer flush
        sleep 2
        kill \${TAIL_PID} 2>/dev/null || true
    else
        echo 'Waiting for SLURM job to start...'
        while squeue -j ${JOB_ID} -h | grep -q \"${JOB_ID}\"; do
            sleep 3
        done
    fi
" || true

echo "--------------------------------------------------------------------------------"
echo "🏁 Job ${JOB_ID} has finished! Automatically syncing logs to local Mac..."
"${ROOT_DIR}/scripts/sync_mufasa_logs.sh"

LATEST_RUN_DIR=$(ls -td "${ROOT_DIR}/logs/mufasa_runs/"* 2>/dev/null | head -n 1 || true)
if [ -n "${LATEST_RUN_DIR}" ]; then
    RUN_NAME=$(basename "${LATEST_RUN_DIR}")
    echo "📂 Synced locally to: ${ROOT_DIR}/logs/mufasa_runs/${RUN_NAME}/"
    if [ -f "${LATEST_RUN_DIR}/benchmark_report.md" ]; then
        echo "📄 Detailed Report: ${LATEST_RUN_DIR}/benchmark_report.md"
    fi
fi
