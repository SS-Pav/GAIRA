"""Signed coordinates must remain visible (no zero-origin radar clipping)."""
from pathlib import Path
import numpy as np, pandas as pd
DEMO = Path(__file__).resolve().parent.parent
from gaira_core import config as cfg
from gaira_core import v3_1_views as v
from gaira_core import coordinate_validation as cv
AXES = list(cfg.BSV_AXES)


def test_frozen_global_ev_means_contain_negatives():
    df = cv.load_reference_samples()
    ev = df[df.dataset == "ev_diabetes"]
    means = {a: float(ev[f"global_{a}"].mean()) for a in AXES}
    assert min(means.values()) < 0, "signed global coords should include negative axis means"


def test_diverging_figure_preserves_sign_and_symmetric_range():
    cohorts = {"OWD": {a: (-1.0 if i % 2 else 1.0) for i, a in enumerate(AXES)},
               "NWD": {a: 0.5 for a in AXES}}
    fig = v.diverging_figure(cohorts, title="t", xlabel="z")
    xs = np.concatenate([np.asarray(tr.x, float) for tr in fig.data])
    assert xs.min() < 0 and xs.max() > 0, "negative values must be present in plotted data"
    rng = fig.layout.xaxis.range
    assert rng[0] < 0 < rng[1] and abs(rng[0] + rng[1]) < 1e-9, "x-range must be symmetric about 0"


def test_zero_reference_line_present():
    fig = v.diverging_figure({"A": {a: 0.3 for a in AXES}}, title="t", xlabel="z")
    shapes = fig.layout.shapes or ()
    assert any(getattr(s, "x0", None) == 0 for s in shapes), "zero reference line required"
