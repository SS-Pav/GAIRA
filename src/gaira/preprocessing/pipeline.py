"""GAIRA V5 — canonical deterministic preprocessing candidates (Phase 1).

No pipeline is assumed correct. Each is a candidate to be benchmarked for
comparability. Steps: crop to common window -> baseline -> smooth -> resample
to a shared grid -> normalize. All deterministic.
"""
from __future__ import annotations
import numpy as np


def common_grid(lo=520.0, hi=1750.0, step=2.0):
    return np.arange(lo, hi + step / 2, step)


def crop(wn, y, lo, hi):
    m = (wn >= lo) & (wn <= hi)
    return wn[m], y[m]


def resample(wn, y, grid):
    o = np.argsort(wn)
    return np.interp(grid, wn[o], y[o], left=np.nan, right=np.nan)


# ── baseline ──────────────────────────────────────────────────────────
def baseline_asls(y, lam=1e5, p=0.01, n_iter=8):
    try:
        from scipy.sparse import diags
        from scipy.sparse.linalg import spsolve
    except Exception:
        x = np.arange(len(y)); return np.polyval(np.polyfit(x, y, 3), x)
    n = len(y)
    D = diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n)).tocsc()
    DTD = D.T @ D
    w = np.ones(n); z = y.copy()
    for _ in range(n_iter):
        W = diags(w, 0, shape=(n, n)).tocsc()
        z = spsolve((W + lam * DTD).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def baseline_poly(y, deg=3):
    x = np.arange(len(y))
    return np.polyval(np.polyfit(x, y, deg), x)


def baseline_none(y):
    return np.zeros_like(y)


BASELINES = {"none": baseline_none, "asls": baseline_asls, "poly3": lambda y: baseline_poly(y, 3)}


# ── smoothing ─────────────────────────────────────────────────────────
def smooth_none(y):
    return y


def smooth_savgol(y, window=9, poly=3):
    try:
        from scipy.signal import savgol_filter
        w = window if window % 2 else window + 1
        if len(y) <= w:
            return y
        return savgol_filter(y, w, poly)
    except Exception:
        return y


SMOOTHERS = {"none": smooth_none, "savgol": smooth_savgol}


# ── normalization ─────────────────────────────────────────────────────
def norm_l2(y):
    n = np.linalg.norm(y); return y / n if n > 1e-12 else y


def norm_area(y):
    a = np.nansum(np.abs(y)); return y / a if a > 1e-12 else y


def norm_snv(y):
    mu, sd = np.nanmean(y), np.nanstd(y)
    return (y - mu) / sd if sd > 1e-12 else y - mu


def norm_robust(y):
    lo, hi = np.nanpercentile(y, 5), np.nanpercentile(y, 95)
    return (y - lo) / (hi - lo) if (hi - lo) > 1e-12 else y - lo


NORMALIZERS = {"l2": norm_l2, "area": norm_area, "snv": norm_snv, "robust": norm_robust}


# ── named candidate pipelines ─────────────────────────────────────────
PIPELINES = {
    "P0_raw_l2":            dict(baseline="none",  smooth="none",   norm="l2"),
    "P1_asls_l2":          dict(baseline="asls",  smooth="none",   norm="l2"),
    "P2_asls_savgol_l2":   dict(baseline="asls",  smooth="savgol", norm="l2"),
    "P3_asls_savgol_snv":  dict(baseline="asls",  smooth="savgol", norm="snv"),
    "P4_poly_savgol_l2":   dict(baseline="poly3", smooth="savgol", norm="l2"),
    "P5_asls_savgol_area": dict(baseline="asls",  smooth="savgol", norm="area"),
}


def preprocess(wn, y, config, grid=None, window=(520.0, 1750.0)):
    """Apply a named-config pipeline. Returns intensity on the shared grid
    (NaN outside the spectrum's own range). config is a PIPELINES value."""
    grid = common_grid(*window) if grid is None else grid
    wn2, y2 = crop(np.asarray(wn, float), np.asarray(y, float), window[0], window[1])
    if len(wn2) < 20:
        return np.full(len(grid), np.nan)
    z = BASELINES[config["baseline"]](y2)
    y2 = y2 - z
    y2 = SMOOTHERS[config["smooth"]](y2)
    yg = resample(wn2, y2, grid)
    # normalize on the finite portion
    finite = np.isfinite(yg)
    if finite.sum() < 20:
        return np.full(len(grid), np.nan)
    out = np.full(len(grid), np.nan)
    out[finite] = NORMALIZERS[config["norm"]](yg[finite])
    return out
