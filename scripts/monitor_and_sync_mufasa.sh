#!/usr/bin/env bash
# Monitors a running Mufasa benchmark job, streams execution logs,
# and automatically syncs all run artifacts to logs/mufasa_runs/ upon completion.
set -euo pipefail

MUFASA_HOST="${MUFASA_HOST:-10.79.23.96}"
MUFASA_USER="${MUFASA_USER:-lpezzoli}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "🔎 Checking active SLURM benchmark jobs on Mufasa (${MUFASA_USER}@${MUFASA_HOST})..."
ACTIVE_JOB=$(ssh "${MUFASA_USER}@${MUFASA_HOST}" "source /etc/profile; module load amd/slurm 2>/dev/null; squeue -u ${MUFASA_USER} -h -o '%i %j %T' | grep -E 'gmmis|bench' | head -n 1" || true)

if [ -z "${ACTIVE_JOB}" ]; then
    echo "ℹ️ No running benchmark jobs found on Mufasa. Syncing latest completed runs directly..."
    "${ROOT_DIR}/scripts/sync_mufasa_logs.sh"
    exit 0
fi

JOB_ID=$(echo "${ACTIVE_JOB}" | awk '{print $1}')
JOB_NAME=$(echo "${ACTIVE_JOB}" | awk '{print $2}')
echo "🎯 Found active job ${JOB_ID} (${JOB_NAME}). Finding latest run directory..."

LATEST_RUN_DIR=$(ssh "${MUFASA_USER}@${MUFASA_HOST}" "ls -td ~/AgentForGMMIS/logs/runs/* 2>/dev/null | head -n 1" || true)
if [ -z "${LATEST_RUN_DIR}" ]; then
    echo "⚠️ No run directory found yet. Waiting for job initialization..."
    sleep 5
    LATEST_RUN_DIR=$(ssh "${MUFASA_USER}@${MUFASA_HOST}" "ls -td ~/AgentForGMMIS/logs/runs/* 2>/dev/null | head -n 1" || true)
fi

RUN_NAME=$(basename "${LATEST_RUN_DIR}")
echo "📋 Streaming live logs for run: ${RUN_NAME}..."
echo "--------------------------------------------------------------------------------"

# Follow logs until the SLURM job finishes
ssh "${MUFASA_USER}@${MUFASA_HOST}" "
    source /etc/profile
    module load amd/slurm 2>/dev/null
    EXEC_LOG=\"${LATEST_RUN_DIR}/execution.log\"
    while squeue -j ${JOB_ID} -h | grep -q \"${JOB_ID}\"; do
        if [ -f \"\${EXEC_LOG}\" ]; then
            tail -n +1 -f \"\${EXEC_LOG}\" --pid=\$\$ &
            TAIL_PID=\$!
            while squeue -j ${JOB_ID} -h | grep -q \"${JOB_ID}\"; do
                sleep 5
            done
            kill \${TAIL_PID} 2>/dev/null || true
            break
        fi
        sleep 2
    done
" || true

echo "--------------------------------------------------------------------------------"
echo "🏁 Job ${JOB_ID} has finished! Automatically syncing logs to local Mac..."
"${ROOT_DIR}/scripts/sync_mufasa_logs.sh"

echo "📂 Synced locally to: ${ROOT_DIR}/logs/mufasa_runs/${RUN_NAME}/"
if [ -f "${ROOT_DIR}/logs/mufasa_runs/${RUN_NAME}/benchmark_report.md" ]; then
    echo "📄 Detailed Report: ${ROOT_DIR}/logs/mufasa_runs/${RUN_NAME}/benchmark_report.md"
fi
