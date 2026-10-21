#!/usr/bin/env bash
# Syncs benchmark runs and logs from Mufasa cluster to local logs/mufasa_runs/
set -euo pipefail

MUFASA_HOST="${MUFASA_HOST:-10.79.23.96}"
MUFASA_USER="${MUFASA_USER:-lpezzoli}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_DEST="${ROOT_DIR}/logs/mufasa_runs"

mkdir -p "$LOCAL_DEST"

echo "🔄 Syncing logs from ${MUFASA_USER}@${MUFASA_HOST}:~/AgentForGMMIS/logs/runs/ to ${LOCAL_DEST}..."
rsync -avz --progress --exclude='*.pt' --exclude='*.pth' --exclude='*.bin' "${MUFASA_USER}@${MUFASA_HOST}:~/AgentForGMMIS/logs/runs/" "$LOCAL_DEST/"

# Also sync the latest summary pointer files
rsync -avz "${MUFASA_USER}@${MUFASA_HOST}:~/AgentForGMMIS/logs/latest_batch_benchmark.*" "${ROOT_DIR}/logs/" 2>/dev/null || true

echo "✅ Sync complete! Runs available locally at ${LOCAL_DEST}"
