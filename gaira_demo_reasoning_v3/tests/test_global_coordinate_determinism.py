"""Determinism: the frozen calibration's center/scale are reproducible from the
reference samples (label-free robust stats), and to_global is deterministic."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEMO_ROOT = Path(__file__).resolve().parent.parent
from gaira_core import config as cfg
from gaira_core import global_coordinates as gc
from gaira_core import coordinate_validation as cv

CAL = DEMO_ROOT / "data" / "generated" / "global_coordinate_calibration_v1.json"


def test_calibration_reproducible_from_reference_raw():
    calib_json = json.loads(CAL.read_text())
    floor = float(calib_json["scale_floor"])
    df = cv.load_reference_samples()
    assert df is not None
    # calibration is fit on the biological population only
    fit = df[df["role"] == "biological_range"]
    R = fit[[f"raw_{a}" for a in cfg.BSV_AXES]].to_numpy(float)
    center = np.median(R, axis=0)
    scale = np.maximum(np.median(np.abs(R - center), axis=0) * 1.4826, floor)
    for j, a in enumerate(cfg.BSV_AXES):
        assert abs(center[j] - calib_json["axis_center"][a]) <= 1e-9, f"center drift {a}"
        assert abs(scale[j] - calib_json["axis_scale"][a]) <= 1e-9, f"scale drift {a}"


def test_content_hash_present_and_stable():
    d = json.loads(CAL.read_text())
    assert "content_sha256" in d and len(d["content_sha256"]) == 64
    # recompute content hash excluding the runtime-only fields
    import hashlib
    content = {k: v for k, v in d.items() if k not in ("content_sha256", "build_timestamp")}
    h = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    assert h == d["content_sha256"], "content hash does not match stored numeric content"


def test_to_global_deterministic():
    calib = gc.load_calibration()
    df = cv.load_reference_samples()
    raw = {a: float(df.iloc[0][f"raw_{a}"]) for a in cfg.BSV_AXES}
    g1 = gc.global_unbounded_vector(raw, calib)
    g2 = gc.global_unbounded_vector(raw, calib)
    assert np.max(np.abs(g1 - g2)) == 0.0
