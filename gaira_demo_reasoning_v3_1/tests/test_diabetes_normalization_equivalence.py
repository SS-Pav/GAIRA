"""Separate normalization from upstream-engine effects."""
from pathlib import Path
import json, numpy as np, pandas as pd
DEMO = Path(__file__).resolve().parent.parent
EQ = DEMO / "data" / "generated" / "diabetes_equivalence"


def test_engine_difference_is_g10_only():
    s = json.loads((EQ / "equivalence_summary.json").read_text())
    assert s["engine_effect_axes_nonzero"] == ["G10_sulfur_thiol_redox"]
    assert s["historical_1304_vs_1322_nonzero_axes"] == ["G10_sulfur_thiol_redox"]


def test_normalization_reproduced_exactly():
    s = json.loads((EQ / "equivalence_summary.json").read_text())
    assert s["historical_zscore_reproduction_max_abs"] <= 1e-9


def test_normalization_is_the_rebalancer():
    # raw redox dominates (rank 1); cohort/robust normalization de-ranks it.
    s = json.loads((EQ / "equivalence_summary.json").read_text())
    r = s["redox_rank_by_variant"]
    assert r["raw"] == 1
    assert r["robust_cohort_z"] >= 3   # normalization moves redox off the top
