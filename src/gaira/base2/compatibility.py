"""Backward-compatibility helpers for gaira_base_2 ↔ gaira_base.

Guarantees, per the backward-compatibility doctrine §3.1:
  * frozen gaira_base pilot files are never modified
  * gaira_base_2 produces an 8-axis projection that is STRUCTURALLY
    COMPATIBLE with gaira_base (same axis names, same [0,1] range,
    same bounded-semantics)
  * NOT byte-identical to gaira_base values — gaira_base_2 computes
    a different object

This module:
  * exposes the SHA-256 manifest of the 15 frozen pilot files that
    must remain unchanged (regression gate)
  * does NOT provide any "reproduce gaira_base values" utility —
    that would be an overclaim
"""
from __future__ import annotations

import hashlib
from pathlib import Path

# The 15 frozen gaira_base pilot files (from architecture freeze).
# Any change to these files triggers the hard rollback regression gate.
GAIRA_BASE_FROZEN_PILOT_FILES: tuple[Path, ...] = tuple(
    Path(f"/Volumes/SSD_Rad/GAIRA_BUILD/gaira_build_axes_v1/outputs/{p}")
    for p in (
        "pilot1/tables/pilot1_hcc_axis_effect_sizes.csv",
        "pilot1/tables/pilot1_hcc_batch_summary.csv",
        "pilot1/tables/pilot1_hcc_cohort_summary.csv",
        "pilot1/tables/pilot1_hcc_per_spectrum_bsv.csv",
        "pilot1/tables/pilot1_hcc_per_spectrum_delta_bsv.csv",
        "pilot2b/tables/pilot2b_cca_raw_axis_effect_sizes.csv",
        "pilot2b/tables/pilot2b_cca_raw_batch_summary.csv",
        "pilot2b/tables/pilot2b_cca_raw_cohort_summary.csv",
        "pilot2b/tables/pilot2b_cca_raw_per_spectrum_bsv.csv",
        "pilot2b/tables/pilot2b_cca_raw_per_spectrum_delta_bsv.csv",
        "pilot3/tables/pilot3_lm_raw_axis_effect_sizes.csv",
        "pilot3/tables/pilot3_lm_raw_batch_summary.csv",
        "pilot3/tables/pilot3_lm_raw_cohort_summary.csv",
        "pilot3/tables/pilot3_lm_raw_per_spectrum_bsv.csv",
        "pilot3/tables/pilot3_lm_raw_per_spectrum_delta_bsv.csv",
    )
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_pilot_manifest() -> dict[str, str]:
    """Current SHA-256 of each frozen pilot file.

    Used by `test_gaira_base_frozen_files_unchanged` to detect any
    drift vs. a recorded manifest.
    """
    out: dict[str, str] = {}
    for p in GAIRA_BASE_FROZEN_PILOT_FILES:
        if not p.exists():
            out[str(p)] = "MISSING"
            continue
        out[str(p)] = sha256_file(p)
    return out
