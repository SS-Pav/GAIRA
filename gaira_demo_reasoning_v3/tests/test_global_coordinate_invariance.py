"""Cohort invariance: a sample's global coordinates must not depend on the
comparison cohort. Cohort-relative coordinates MUST change."""
from pathlib import Path

import numpy as np
import pandas as pd

DEMO_ROOT = Path(__file__).resolve().parent.parent

from gaira_core import config as cfg
from gaira_core import global_coordinates as gc
from gaira_core import coordinate_validation as cv


def _ref():
    df = cv.load_reference_samples()
    assert df is not None, "reference samples artifact missing"
    return df


def test_global_coords_invariant_to_cohort():
    calib = gc.load_calibration()
    assert calib is not None and calib.is_valid()
    df = _ref()
    # Take representative samples from each dataset
    samples = (df.groupby("dataset").head(3))
    serum = [r for _, r in df[df.dataset == "serum_liver"].head(20).iterrows()]
    ev = [r for _, r in df[df.dataset == "ev_diabetes"].head(20).iterrows()]

    def raw_of(row):
        return {a: float(row[f"raw_{a}"]) for a in cfg.BSV_AXES}

    max_dev = 0.0
    for _, row in samples.iterrows():
        raw = raw_of(row)
        # comparison sets: alone, own-cohort proxy (serum), different-cohort (ev),
        # and a mixed EV+serum set
        comp_sets = [
            [],
            [raw_of(x) for x in serum[:10]],
            [raw_of(x) for x in ev[:10]],
            [raw_of(x) for x in (serum[:5] + ev[:5])],
        ]
        res = cv.invariance_check(raw, calib, comp_sets, atol=1e-9)
        assert res["global_invariant"], f"global coords not invariant: {res}"
        max_dev = max(max_dev, res["global_max_deviation"])
    assert max_dev <= 1e-9


def test_cohort_relative_DOES_change():
    df = _ref()
    serum = [{a: float(r[f"raw_{a}"]) for a in cfg.BSV_AXES}
             for _, r in df[df.dataset == "serum_liver"].head(10).iterrows()]
    ev = [{a: float(r[f"raw_{a}"]) for a in cfg.BSV_AXES}
          for _, r in df[df.dataset == "ev_diabetes"].head(10).iterrows()]
    sample = serum[0]
    z_alone = gc.cohort_relative_zscores([sample])[0]
    z_with_ev = gc.cohort_relative_zscores([sample] + ev)[0]
    dev = max(abs(z_alone[a] - z_with_ev[a]) for a in cfg.BSV_AXES)
    assert dev > 1e-6, "cohort-relative coords should change with comparison set"
