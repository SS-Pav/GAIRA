#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${(%):-%N}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"
PORT="${PORT:-8501}"

cd "$REPO_DIR"

if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
fi

export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Launching GAIRAM v1 EV Graph Explorer"
echo "Repo: $REPO_DIR"
echo "Port: $PORT"
echo "Open http://localhost:$PORT once Streamlit is ready."

exec streamlit run "$REPO_DIR/streamlit_apps/ev_graph_explorer.py" --server.port "$PORT"
