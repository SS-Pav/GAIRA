#!/usr/bin/env bash
# GAIRA Scientific Reasoning Demo v2 — portable launcher.
#
# No hardcoded usernames or drive names. Resolves the demo dir from this
# script's own location, finds a usable Python/Streamlit, and launches.
#
# Optional environment:
#   GAIRA_DATA_ROOT         path to your GAIRA_DATA (with raw/ and processed/)
#   GAIRA_LEGACY_DEMO_DATA  path to the legacy grounding/calibration CSVs
#   GAIRA_DEMO_PORT         streamlit port (default 8501)
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DEMO_DIR/.." && pwd)"
PORT="${GAIRA_DEMO_PORT:-8501}"

# Pick a Python: repo .venv → active venv → python3.
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
  PY="$REPO_ROOT/.venv/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
  PY="$VIRTUAL_ENV/bin/python"
else
  PY="$(command -v python3 || command -v python)"
fi
echo "[run_demo] python: $PY"

# Verify streamlit is importable before launching.
if ! "$PY" -c "import streamlit" 2>/dev/null; then
  echo "[run_demo] ERROR: streamlit not available for $PY" >&2
  echo "           pip install streamlit pandas numpy plotly scipy scikit-learn (umap-learn optional)" >&2
  exit 1
fi

# Show what the demo will resolve before launching (fast, no server).
"$PY" "$DEMO_DIR/selfcheck.py" || true

echo "[run_demo] launching on http://localhost:$PORT"
cd "$DEMO_DIR"
exec "$PY" -m streamlit run app.py --server.port "$PORT"
