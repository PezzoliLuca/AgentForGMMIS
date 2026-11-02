#!/bin/bash
# ==============================================================================
# Launch Streamlit Control & Comparison Dashboard for Agent for GMMIS
# ==============================================================================
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
    source "$ROOT_DIR/.venv/bin/activate"
elif [ -f "$HOME/AgentForGMMIS/venv/bin/activate" ]; then
    source "$HOME/AgentForGMMIS/venv/bin/activate"
fi

echo "🚀 Starting Agent for GMMIS Streamlit Dashboard..."
streamlit run src/agent_gmmis/ui/dashboard.py "$@"

