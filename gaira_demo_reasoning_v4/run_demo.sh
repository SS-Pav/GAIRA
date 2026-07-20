#!/usr/bin/env bash
# Portable launcher for the GAIRA V6 reasoning demo.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
exec streamlit run app.py "$@"
