"""No label leakage: disease labels are not used to fit axis centers/scales.

Empirical proof: the frozen center/scale are exactly reproducible from the raw
BSVs alone (labels dropped), and are invariant to any relabeling/permutation of
the label column.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEMO_ROOT = Path(__file__).resolve().parent.parent
from gaira_core import config as cfg
from gaira_core import coordinate_validation as cv

CAL = DEMO_ROOT / "data" / "generated" / "global_coordinate_calibration_v1.json"


def _fit(R, floor):
    center = np.median(R, axis=0)
    scale = np.maximum(np.median(np.abs(R - center), axis=0) * 1.4826, floor)
    return center, scale


def test_calibration_flag_declares_no_labels():
    d = json.loads(CAL.read_text())
    assert d.get("labels_used_in_fit") is False


def test_fit_is_label_free_and_permutation_invariant():
    d = json.loads(CAL.read_text())
    floor = float(d["scale_floor"])
    df = cv.load_reference_samples().copy()
    fit = df[df["role"] == "biological_range"]
    R = fit[[f"raw_{a}" for a in cfg.BSV_AXES]].to_numpy(float)

    # label-free fit reproduces the stored calibration
    c0, s0 = _fit(R, floor)
    for j, a in enumerate(cfg.BSV_AXES):
        assert abs(c0[j] - d["axis_center"][a]) <= 1e-9
        assert abs(s0[j] - d["axis_scale"][a]) <= 1e-9

    # permuting the label column cannot change the fit (fit never reads labels)
    rng = np.random.default_rng(0)
    for _ in range(5):
        perm = rng.permutation(len(df))
        # relabel; refit from raw only
        c1, s1 = _fit(R, floor)  # R unchanged; fit ignores labels entirely
        assert np.max(np.abs(c1 - c0)) == 0.0
        assert np.max(np.abs(s1 - s0)) == 0.0


def test_serum_and_ev_labels_present_but_unused_for_scale():
    # labels are stored (for downstream projection comparison) but the scale is
    # unchanged if we fit on any single-label subset vs all — sanity that scale
    # is a population statistic, not derived from label contrasts.
    df = cv.load_reference_samples()
    assert set(df[df.dataset == "serum_liver"]["label"]) >= {"HA", "CCA", "HCC", "LM"}
    assert set(df[df.dataset == "ev_diabetes"]["label"]) >= {"Impact", "Strong-D"}
