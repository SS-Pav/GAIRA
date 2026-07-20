"""Calibration analysis helpers for Page 4 (the strongest page).

Turns a cached frozen-atlas projection into dose-resolved reasoning: per-dose mean
coordinates, MSS-motif and theme evolution, Langmuir dose-response fits, and BSV-space
trajectories. All quantities flow through the live frozen engine + MSS layer — nothing
is re-fit or re-projected. The only fitted object is the Langmuir curve OVERLAID on the
engine's dose-response (a visual summary, not part of the engine).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

from . import data as D

# canonical strong-adsorber acquisition for adenine (clean, recoverable regime)
ADENINE_METHOD = "cAg@785"


@dataclass
class DoseSeries:
    analyte: str
    level_col: str
    levels: np.ndarray                 # sorted unique dose levels
    mean_coord: np.ndarray             # (n_levels, 24) per-dose mean coordinates
    coords_by_level: list              # list of (n_reps, 24) arrays, aligned to levels
    n_per_level: np.ndarray
    condition: str = ""                # acquisition description


def build_dose_series(cal, method=None):
    """Build a per-dose series for adenine/ergothioneine (has a numeric level column)."""
    Z, meta = D.load_projection(cal.projection)
    lc = cal.level_col
    if method and "method" in meta.columns:
        m = meta["method"] == method
        Z, meta = Z[m.values], meta[m].reset_index(drop=True)
    lv = np.asarray(meta[lc], float)
    levels = np.array(sorted(np.unique(lv[np.isfinite(lv)])))
    coords_by_level, means, n = [], [], []
    for d in levels:
        sub = Z[lv == d]
        coords_by_level.append(sub)
        means.append(sub.mean(0))
        n.append(len(sub))
    cond = ""
    if "substrate" in meta.columns:
        cond = f"{method or meta['substrate'].iloc[0]}"
    return DoseSeries(cal.analyte, lc, levels, np.array(means), coords_by_level,
                      np.array(n), cond)


def theme_series(bridge, series, theme_id):
    """Per-dose theme composition (mean coord through the engine) + all replicate points."""
    mean_scores = np.array([bridge.infer(mc).bsv.composition[theme_id]
                            for mc in series.mean_coord])
    rep_levels, rep_scores = [], []
    for d, reps in zip(series.levels, series.coords_by_level):
        for r in reps:
            rep_levels.append(d)
            rep_scores.append(bridge.infer(r).bsv.composition[theme_id])
    return mean_scores, np.array(rep_levels), np.array(rep_scores)


def motif_evolution(bridge, series, motif_ids, value="elevation"):
    """Per-dose MSS motif value (elevation or composition) for each motif id.
    Returns dict motif_id -> array over levels."""
    out = {m: [] for m in motif_ids}
    for mc in series.mean_coord:
        _, acts = bridge.bsv_and_mss(mc)
        by_id = {a.id: a for a in acts}
        for m in motif_ids:
            out[m].append(getattr(by_id[m], value))
    return {m: np.array(v) for m, v in out.items()}


def component_series(bridge, series):
    """Per-dose 24-component mean share (n_levels, 24)."""
    return np.array([bridge.infer(mc).bsv.component_coord for mc in series.mean_coord])


def langmuir_fit(x, y):
    """Fit saturating a*x/(K+x)+b; return (xfit, yfit, params, r2, K) or None."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(np.unique(x)) < 3:
        return None
    f = lambda t, a, k, b: a * t / (k + t) + b
    try:
        p0 = [np.ptp(y), np.median(x[x > 0]) if (x > 0).any() else 1.0, y.min()]
        popt, _ = curve_fit(f, x, y, p0=p0, maxfev=10000)
        xf = np.linspace(x.min(), x.max(), 100)
        yf = f(xf, *popt)
        ss = 1 - np.sum((y - f(x, *popt)) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12)
        return xf, yf, popt, float(ss), float(popt[1])
    except Exception:
        return None


def bsv_theme_vectors(bridge, coords):
    """Stack biochemical-theme composition vectors for a set of coordinate rows."""
    themes = bridge.bio_themes
    return np.array([[bridge.infer(c).bsv.composition[t] for t in themes] for c in coords])


def trajectory_2d(vectors, ref=None):
    """PCA to 2D for visualisation only (fit on the supplied vectors, optionally
    including a reference cloud). Deterministic (sign-fixed)."""
    X = vectors if ref is None else np.vstack([vectors, ref])
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comp = Vt[:2]
    # deterministic sign: largest-abs loading positive
    for i in range(2):
        if comp[i][np.argmax(np.abs(comp[i]))] < 0:
            comp[i] = -comp[i]
    proj = (vectors - X.mean(0)) @ comp.T
    var = (S[:2] ** 2) / (S ** 2).sum()
    return proj, var


# ── uricase depletion (condition-based, not dose-based) ──
def uricase_conditions(bridge):
    """Mean coordinates per condition for the uricase depletion study."""
    Z, meta = D.load_projection("uricase")
    order = ["serum_reference", "spiked", "spiked+uricase"]
    out = {}
    for cond in order:
        m = (meta["condition"] == cond).values
        if m.any():
            out[cond] = Z[m].mean(0)
    return out


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(np.unique(x)) < 3 or y.std() < 1e-12:
        return np.nan
    return float(spearmanr(x, y)[0])


def joint_trajectories(bridge):
    """Adenine / ergothioneine / uricase trajectories projected into ONE shared BSV
    PCA space, for the side-by-side trajectory-class comparison."""
    ade = build_dose_series(D.calibration("adenine"), method=ADENINE_METHOD)
    erg = build_dose_series(D.calibration("ergothioneine"))
    uri = uricase_conditions(bridge)
    v_ade = bsv_theme_vectors(bridge, ade.mean_coord)
    v_erg = bsv_theme_vectors(bridge, erg.mean_coord)
    uri_order = [c for c in ("serum_reference", "spiked", "spiked+uricase") if c in uri]
    v_uri = bsv_theme_vectors(bridge, np.array([uri[c] for c in uri_order]))
    allv = np.vstack([v_ade, v_erg, v_uri])
    mean = allv.mean(0)
    U, S, Vt = np.linalg.svd(allv - mean, full_matrices=False)
    comp = Vt[:2]
    for i in range(2):
        if comp[i][np.argmax(np.abs(comp[i]))] < 0:
            comp[i] = -comp[i]
    P = lambda V: (V - mean) @ comp.T
    return {
        "adenine": {"proj": P(v_ade), "levels": ade.levels, "class": "redistribution"},
        "ergothioneine": {"proj": P(v_erg), "levels": erg.levels, "class": "scaling"},
        "uricase": {"proj": P(v_uri), "labels": uri_order, "class": "depletion"},
    }
