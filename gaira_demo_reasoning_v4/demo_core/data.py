"""Dataset catalog + cached-projection loader for the demo.

All projections are the FROZEN-atlas coordinates produced by the earlier validation
studies (results/v5_rebuild/...). The demo reads these cached 24-D coordinates and
runs them through the live engine + MSS layer; it never re-projects or re-fits.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

from .engine_bridge import REPO, K

SPIKE = REPO / "results/v5_rebuild/spike_validation/tables"
REF_ART = Path(__file__).resolve().parents[1] / "reference_artifacts"


def matched_raman_sers_pairs():
    """The committed pure-Raman ↔ pure-Ag-SERS matched-pair artifact (Gobbato 2025,
    51 analytes) built by tools/build_matched_pairs.py. Returns the parsed dict, or
    None on a checkout where it was not generated."""
    p = REF_ART / "matched_raman_sers_pairs.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


@dataclass
class Calibration:
    key: str
    title: str
    analyte: str
    projection: str        # phase3_projection_<name>.csv
    level_col: str         # dose / concentration column
    mechanism: str         # "redistribution" | "scaling" | "depletion"
    target_theme: str
    target_motif: str
    story: str


# The three canonical calibration studies (BSV validation Parts 3/5/13).
CALIBRATIONS = [
    Calibration(
        key="adenine", title="Adenine dose-response", analyte="adenine",
        projection="ils_adenine", level_col="conc_uM", mechanism="redistribution",
        target_theme="nucleic_purine", target_motif="purine_ring_breathing",
        story="A strong Ag adsorber. As dose rises the purine motif climbs, but the "
              "underlying components REDISTRIBUTE (different components dominate at "
              "different doses) rather than a single one scaling."),
    Calibration(
        key="ergothioneine", title="Ergothioneine dose-response", analyte="ergothioneine",
        projection="ergothioneine", level_col="conc_uM", mechanism="scaling",
        target_theme="sulfur_antioxidant", target_motif="sulfur_heterocycle_thione",
        story="The cleanest calibrant. A thione that chemisorbs to Ag; the sulfur motif "
              "scales monotonically and a single component carries most of the change."),
    Calibration(
        key="uricase", title="Uricase depletion", analyte="urate",
        projection="uricase", level_col="", mechanism="depletion",
        target_theme="nucleic_purine", target_motif="oxopurine_carbonyl",
        story="An enzymatic knock-OUT: uricase removes urate. The oxopurine motif is the "
              "read-out; the before/after difference isolates purine-specific evidence."),
]


def calibration(key):
    return next(c for c in CALIBRATIONS if c.key == key)


def load_projection(name):
    """Return (Z [n,24], meta DataFrame) for a cached frozen-atlas projection."""
    df = pd.read_csv(SPIKE / f"phase3_projection_{name}.csv")
    cols = [f"c{j}" for j in range(K)]
    Z = df[cols].values.astype(float)
    meta = df.drop(columns=[c for c in cols if c in df.columns])
    return Z, meta


def available_projections():
    return sorted(p.stem.replace("phase3_projection_", "")
                  for p in SPIKE.glob("phase3_projection_*.csv"))


# Biological cohorts (Page 6) — resolved lazily; the demo degrades gracefully if a
# cohort's cached projection is not present on this machine.
BIOLOGICAL_COHORTS = [
    {"key": "serum_baseline", "title": "Serum baseline", "domain": "serum",
     "note": "Ag-colloid serum background — the matrix every serum spike sits on."},
    {"key": "spiked_serum", "title": "Spiked serum", "domain": "serum",
     "note": "Serum with controlled analyte additions (recoverability stress test)."},
    {"key": "pure_sers", "title": "Pure-analyte SERS", "domain": "buffer",
     "note": "Single analytes on Ag colloid, no matrix."},
]


def cohort_available(key):
    return (SPIKE / f"phase3_projection_{key}.csv").exists()
