#!/usr/bin/env python3
"""GAIRA Demo v2 — data-source self-check (no Streamlit needed).

Prints exactly what the demo will resolve on THIS machine: the data root,
autoresearch root, adenine dir, legacy-CSV source, and per-section real-vs-
placeholder status. Use it to verify a new machine or a relocated drive
before launching the app.

Usage:
    python selfcheck.py
Exit code: 0 if in 'real' mode, 1 otherwise (useful in CI / smoke tests).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a bare script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gaira_core import paths as gpaths  # noqa: E402


def main() -> int:
    s = gpaths.get_data_status()
    print("GAIRA Demo v2 — data-source self-check")
    print("=" * 52)
    print(f"mode                : {s.mode.upper()}")
    print(f"data root           : {s.data_root or '— (not resolved)'}")
    print(f"  mounted/usable    : {s.data_root_mounted}")
    print(f"autoresearch root   : {s.autoresearch_root or '—'}")
    print(f"adenine raw dir     : {s.adenine_dir or '—'}")
    print(f"legacy CSV dir      : {s.legacy_dir}  [{s.legacy_kind}]")
    print(f"sections on real    : {s.real_section_count}/{len(s.checks)}")
    print("-" * 52)
    for section, is_real in s.checks.items():
        mark = "REAL " if is_real else "place"
        print(f"  [{mark}] {section}")
    print("-" * 52)
    if s.mode != "real":
        print("To use your data, set GAIRA_DATA_ROOT to a folder containing")
        print("raw/ and processed/ (see MIGRATION_HARDENING.md), then rerun.")
    return 0 if s.mode == "real" else 1


if __name__ == "__main__":
    raise SystemExit(main())
