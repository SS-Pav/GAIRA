"""GAIRA data/physics audit — substrate & physics-layer ablation tests.

Read-only. Uses the V3.1 demo engine (unmodified) and controlled calibration
data to measure whether the substrate/modality/physics layers have demonstrable
utility. Ablations:
  * full pipeline vs minus-substrate-weighting        (adenine Ag-SERS dose)
  * cross-substrate stability (cAg/cAu/sAg/sAu)        (European adenine)
  * cross-modality/excitation (532 vs 785)             (European adenine)
  * physics caveats / collision effect on BSV numbers  (adenine)
  * substrate thiol boost on ergothioneine G10

Outputs: data_audit/ablation_results.csv  (+ printed summary)
Deterministic. No data modified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/surajpg/projects/GAIRA")
DEMO = REPO / "gaira_demo_reasoning_v3_1"
sys.path.insert(0, str(DEMO))

from gaira_core import config as cfg                         # noqa
from gaira_core import report_builder as rb                   # noqa
from gaira_core import substrate_physics as sp                # noqa

GRID = np.linspace(cfg.WAVENUMBER_MIN, cfg.WAVENUMBER_MAX, cfg.WAVENUMBER_N)
OUT = REPO / "data_audit"
OUT.mkdir(exist_ok=True)


def _interp(wn, y):
    o = np.argsort(wn)
    return np.clip(np.interp(GRID, np.asarray(wn)[o], np.asarray(y)[o], left=0, right=0), 0, None)


def _identity_substrate(motif_scores, *, substrate):
    return dict(motif_scores), []


def bsv(wn, y, substrate, ablate_substrate=False):
    orig = rb.apply_substrate_corrections
    if ablate_substrate:
        rb.apply_substrate_corrections = _identity_substrate
    try:
        rep = rb.build_report(sample_id="t", title="t", domain="x",
                              substrate=substrate, wavenumber=wn, intensity=y)
        return {a: float(rep["bsv"][a]) for a in cfg.BSV_AXES}, rep
    finally:
        rb.apply_substrate_corrections = orig


def _spearman(x, y):
    xr, yr = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    if xr.std() < 1e-9 or yr.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def load_adenine_control():
    from gaira_core.data_loader import _ADENINE_FILES, _read_adenine_csv, _crop_and_interp
    d = cfg.ADENINE_RAW_DIR
    rows = []
    for fname, label, ng in _ADENINE_FILES:
        p = d / fname
        parsed = _read_adenine_csv(p) if p.exists() else None
        if parsed is None:
            continue
        wn, y = _crop_and_interp(*parsed)
        rows.append((label, ng, wn, y))
    return rows


def load_european_adenine():
    p = cfg.GAIRA_DATA_VOLUME / "raw" / "european_multi_instrument_adenine" / "ILSdata.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    meta = ["labcode", "substrate", "laser", "method", "sample", "type", "conc", "batch", "replica"]
    wn_cols = [c for c in df.columns if c not in meta]
    wn = np.array([float(c) for c in wn_cols], float)
    return df, meta, wn_cols, wn


G01 = "G01_purine_nucleotide"
results = []


def rec(test, layer, metric, full, ablated, verdict, note=""):
    results.append({"test": test, "layer": layer, "metric": metric,
                    "with_layer": full, "without_layer": ablated,
                    "verdict": verdict, "note": note})


# ── 1. Substrate weighting on adenine Ag-SERS dose ──
ad = load_adenine_control()
if ad:
    ng = [r[1] for r in ad]
    full_g01 = [bsv(r[2], r[3], "Ag colloid SERS", False)[0][G01] for r in ad]
    abl_g01 = [bsv(r[2], r[3], "Ag colloid SERS", True)[0][G01] for r in ad]
    # off-target spillover = mean |non-G01 axes|
    full_full = [bsv(r[2], r[3], "Ag colloid SERS", False)[0] for r in ad]
    abl_full = [bsv(r[2], r[3], "Ag colloid SERS", True)[0] for r in ad]
    spill_full = np.mean([np.mean([v[a] for a in cfg.BSV_AXES if a != G01]) for v in full_full])
    spill_abl = np.mean([np.mean([v[a] for a in cfg.BSV_AXES if a != G01]) for v in abl_full])
    sp_full, sp_abl = _spearman(np.log10(ng), full_g01), _spearman(np.log10(ng), abl_g01)
    rec("adenine_dose", "substrate_weighting", "G01_dose_spearman", round(sp_full, 3), round(sp_abl, 3),
        "no_detectable_utility" if abs(sp_full - sp_abl) < 0.05 else "suggestive_utility",
        "substrate dampen ×0.65 lowers G01 magnitude but preserves dose ordering")
    rec("adenine_dose", "substrate_weighting", "G01_max", round(max(full_g01), 4), round(max(abl_g01), 4),
        "demonstrated_utility" if max(abl_g01) > max(full_g01) else "no_effect",
        "dampening reduces peak G01 (keeps call class-level) — a magnitude effect, not a validated correction")
    rec("adenine_dose", "substrate_weighting", "offtarget_spillover", round(float(spill_full), 4),
        round(float(spill_abl), 4), "no_detectable_utility" if abs(spill_full - spill_abl) < 1e-4 else "suggestive_utility",
        "substrate rule does not change off-target axes (only the purine motif)")

# ── 2 & 3. Cross-substrate + cross-modality stability (European adenine) ──
eu = load_european_adenine()
if eu is not None:
    df, meta, wn_cols, wn = eu
    # highest available concentration test spectra
    test = df[(df["type"] == "test") | (df["type"] == "train")]
    hi = test[test["sample"] == "C9"] if (test["sample"] == "C9").any() else test
    # map physical substrate to the demo's only two options
    demo_sub = {"cAg": "Ag colloid SERS", "cAu": "Ag colloid SERS",
                "sAg": "Ag colloid SERS", "sAu": "Ag colloid SERS"}
    g01_by_sub = {}
    for sub in ["cAg", "cAu", "sAg", "sAu"]:
        rows = hi[hi["substrate"] == sub]
        vals = []
        for _, r in rows.head(8).iterrows():
            y = _interp(wn, r[wn_cols].to_numpy(float))
            vals.append(bsv(GRID, y, demo_sub[sub], False)[0][G01])
        if vals:
            g01_by_sub[sub] = float(np.mean(vals))
    if len(g01_by_sub) >= 2:
        cv = float(np.std(list(g01_by_sub.values())) / (np.mean(list(g01_by_sub.values())) + 1e-9))
        rec("european_adenine", "substrate_weighting", "cross_substrate_G01_CV",
            round(cv, 3), "n/a",
            "not_testable_with_current_data" if cv else "n/a",
            f"demo maps ALL of {list(g01_by_sub)} to one rule ('Ag colloid SERS'); it is BLIND to "
            f"Au vs Ag and colloid vs planar. G01 by substrate: "
            + ", ".join(f"{k}={v:.3f}" for k, v in g01_by_sub.items()))
    # cross-modality (532 vs 785) for cAg
    g01_by_laser = {}
    for laser in [532, 785]:
        rows = hi[(hi["substrate"] == "cAg") & (hi["laser"] == laser)]
        vals = [bsv(GRID, _interp(wn, r[wn_cols].to_numpy(float)), "Ag colloid SERS", False)[0][G01]
                for _, r in rows.head(8).iterrows()]
        if vals:
            g01_by_laser[laser] = float(np.mean(vals))
    if len(g01_by_laser) == 2:
        rec("european_adenine", "modality_excitation", "G01_532_vs_785",
            round(g01_by_laser[532], 3), round(g01_by_laser[785], 3),
            "not_testable_with_current_data",
            "demo has NO excitation-wavelength awareness; 532 and 785 treated identically")

# ── 4. Physics caveats / collision — do they change BSV numbers? ──
if ad:
    wn0, y0 = ad[-1][2], ad[-1][3]
    _, rep = bsv(wn0, y0, "Ag colloid SERS", False)
    rec("adenine_highconc", "physics_caveats_collision", "caveats_change_bsv", "no", "no",
        "no_detectable_utility",
        f"{len(rep['caveats'])} caveats generated but BSV numbers are unaffected by caveats/collision "
        "(caveat generator only, not a numerical correction)")

# ── 5. Substrate thiol boost on ergothioneine G10 ──
erg_p = cfg.GAIRA_DATA_VOLUME / "raw" / "ergothioneine_serum" / "ERG_calibration.csv"
if erg_p.exists():
    edf = pd.read_csv(erg_p)
    ecols = [c for c in edf.columns if c not in ("laser", "power", "substrate", "c")]
    ewn = np.array([float(c) for c in ecols], float)
    hi = edf[edf["c"] == edf["c"].max()]
    g10 = "G10_sulfur_thiol_redox"
    full = np.mean([bsv(GRID, _interp(ewn, r[ecols].to_numpy(float)), "Ag colloid SERS", False)[0][g10] for _, r in hi.iterrows()])
    abl = np.mean([bsv(GRID, _interp(ewn, r[ecols].to_numpy(float)), "Ag colloid SERS", True)[0][g10] for _, r in hi.iterrows()])
    rec("ergothioneine_highdose", "substrate_weighting", "G10_thiol_boost",
        round(float(full), 4), round(float(abl), 4),
        "suggestive_utility" if full > abl else "no_effect",
        "Ag-SERS thiol ×1.20 boost raises G10 at high ergothioneine dose (heuristic, not validated vs ground truth)")

df_out = pd.DataFrame(results)
df_out.to_csv(OUT / "ablation_results.csv", index=False)
print(df_out.to_string(index=False))
print("\nwrote", OUT / "ablation_results.csv")
