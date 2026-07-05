"""CLI entry point for the GAIRA drug-detection layer.

Usage:
    python -m gaira.drug_detection                              # OFF (default) — does nothing
    python -m gaira.drug_detection --enable-drug-detection      # ON — runs on demo OTC spectra
    python -m gaira.drug_detection --enable-drug-detection --config path/to/config.yaml
    python -m gaira.drug_detection --help

The CLI is a thin wrapper intended for smoke-testing the toggle contract.
For real pipeline integration, callers should import
`run_drug_detection_layer` directly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from gaira.drug_detection import (
    run_drug_detection_layer,
    load_config_flag_from_yaml,
)


def _cli():
    parser = argparse.ArgumentParser(
        prog="gaira.drug_detection",
        description="Toggle-aware OTC drug detection layer (parallel to BSV).",
    )
    parser.add_argument("--enable-drug-detection", action="store_true",
                           help="Enable the drug-detection layer (default: OFF).")
    parser.add_argument("--config", type=Path, default=None,
                           help=("YAML config file with `enable_drug_detection: true`. "
                                  "If both --enable-drug-detection and --config are given, "
                                  "the flag takes precedence."))
    parser.add_argument("--demo", action="store_true",
                           help="Run a synthetic demo spectrum (zeros) to show the output contract.")
    args = parser.parse_args()

    # Resolve enable flag
    enable = args.enable_drug_detection
    if not enable and args.config is not None:
        enable = load_config_flag_from_yaml(args.config)

    # Demo input — synthetic zero spectrum on canonical master axis
    master_x = np.arange(400, 1801, dtype=float)
    y = np.zeros_like(master_x)
    if args.demo:
        # Small synthetic signal (no real drug bands — expected: NOT_DETECTED even when enabled)
        y = np.random.default_rng(0).normal(scale=1e-3, size=len(master_x))

    result = run_drug_detection_layer(
        y_pp=y, master_x=master_x, enable_drug_detection=enable,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
