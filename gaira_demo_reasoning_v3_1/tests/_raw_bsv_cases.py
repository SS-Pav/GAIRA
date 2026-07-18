"""Shared deterministic raw-BSV test cases (used by baseline gen + regression test).

Uses whatever `gaira_core` is on sys.path, so it can be run against V2 (to make
the baseline) or V3 (in the test). All inputs are real files or fixed-seed
synthetic spectra, so results are process-independent.
"""
from __future__ import annotations

import json as _json

import numpy as np

WN_MIN, WN_MAX, WN_N = 400.0, 1800.0, 1401
GRID = np.linspace(WN_MIN, WN_MAX, WN_N)


def _interp(wn, y):
    order = np.argsort(wn)
    return np.interp(GRID, np.asarray(wn)[order], np.asarray(y)[order], left=0.0, right=0.0)


def standard_raw_bsvs() -> dict:
    from gaira_core import config as cfg
    from gaira_core.report_builder import build_report
    from gaira_core import data_loader as dl
    import pandas as pd

    out: dict[str, dict] = {}

    def rec(name, wn, y, substrate, domain):
        rep = build_report(sample_id=name, title=name, domain=domain,
                           substrate=substrate, wavenumber=np.asarray(wn, float),
                           intensity=np.asarray(y, float))
        out[name] = {a: float(rep["bsv"][a]) for a in cfg.BSV_AXES}

    # adenine — 6 real concentrations
    from gaira_core.data_loader import _ADENINE_FILES, _read_adenine_csv, _crop_and_interp
    for fname, label, _ in _ADENINE_FILES:
        p = cfg.ADENINE_RAW_DIR / fname
        parsed = _read_adenine_csv(p) if p.exists() else None
        if parsed is None:
            continue
        wn, y = _crop_and_interp(*parsed)
        rec(f"adenine::{label}", wn, y, "Ag colloid SERS", "calibration")

    # serum — first patient per cohort
    sp_path = cfg.LIVER_PATIENT_TABLES / "patient_level_mean_spectra.csv"
    if sp_path.exists():
        sp = pd.read_csv(sp_path)
        wn_cols = [c for c in sp.columns if c.startswith("wn_")]
        wn = np.array([int(c[3:]) for c in wn_cols], float)
        for coh in ("HA", "CCA", "HCC", "LM"):
            sub = sp[sp["class_label_display"] == coh]
            if len(sub):
                r = sub.iloc[0]
                rec(f"serum::{coh}::{r['sample_id']}", wn,
                    np.clip(r[wn_cols].to_numpy(float), 0, None), "Ag colloid SERS", "serum")

    # EV — first sample per cohort
    ev_path = cfg.EV_DIABETES_TABLES / "sample_query_spectra.csv"
    if ev_path.exists():
        ev = pd.read_csv(ev_path)
        for coh in ("Impact", "Strong-D"):
            sub = ev[ev["class_label"] == coh]
            if len(sub):
                r = sub.iloc[0]
                wn = np.asarray(_json.loads(r["wavenumbers_json"]), float)
                y = np.asarray(_json.loads(r["intensity_json"]), float)
                rec(f"ev::{coh}::{r['sample_id']}", GRID,
                    np.clip(_interp(wn, y), 0, None), "Ag colloid SERS", "extracellular_vesicle")

    # synthetic — fixed seeds (process-independent)
    wn, y = dl.synth_reference_spectrum("adenine", seed=12345)
    rec("synth::adenine", wn, y, "Ag colloid SERS", "calibration")
    wn, y = dl.synth_reference_spectrum("ergothioneine", seed=54321)
    rec("synth::ergothioneine", wn, y, "Raman", "calibration")

    return out


if __name__ == "__main__":
    import sys
    print(_json.dumps(standard_raw_bsvs()))
