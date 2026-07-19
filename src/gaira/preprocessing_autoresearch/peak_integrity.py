"""Stage B0 — peak detection, correspondence and spectral-integrity guards.

Reuses the peak picker from src/gaira/representation/metrics.py. Adds:
  * peak retention / invention relative to a REFERENCE preprocessing (raw+L2),
  * peak-width change and band-position stability,
  * matched vs mismatched-analyte vs random-peak correspondence (Controls 1-2),
  * negative-lobe and edge-artefact burden (over-subtraction guards).

Note on the two peak guards: because peak prominence is measured relative to each
spectrum's own range, OVER-SMOOTHING preserves peak count and position while
broadening bands. `retention` therefore detects peak DISAPPEARANCE, while
`width_ratio` detects BROADENING. Both are required, and both are enforced in
pareto.REJECT (min_peak_retention and max_width_ratio).
"""
from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import linear_sum_assignment

from ..representation.metrics import peaks as _peaks  # reuse

TOL = 12.0          # cm-1 correspondence tolerance
PROM_FRAC = 0.10    # prominence threshold as a fraction of spectral range


def detect(y, grid, prom_frac=PROM_FRAC, min_sep_cm=8.0):
    y = np.nan_to_num(np.asarray(y, float))
    rng = float(y.max() - y.min())
    if rng <= 0:
        return np.array([]), np.array([]), np.array([])
    dx = float(np.median(np.diff(grid)))
    idx, props = find_peaks(y, prominence=prom_frac * rng,
                            distance=max(1, int(round(min_sep_cm / dx))))
    if len(idx) == 0:
        return np.array([]), np.array([]), np.array([])
    w = peak_widths(y, idx, rel_height=0.5)[0] * dx
    return grid[idx], props["prominences"], w


def match(p1, p2, tol=TOL):
    """Optimal assignment; returns (n_matched, mean_abs_shift)."""
    if len(p1) == 0 or len(p2) == 0:
        return 0, np.nan
    C = np.abs(np.asarray(p1)[:, None] - np.asarray(p2)[None, :])
    cost = np.where(C <= tol, C, 1e6)
    r, c = linear_sum_assignment(cost)
    ok = cost[r, c] < 1e6
    if ok.sum() == 0:
        return 0, np.nan
    return int(ok.sum()), float(np.mean(C[r[ok], c[ok]]))


def retention_invention(y_ref, y_new, grid, tol=TOL):
    """Peak retention and invention of a processed spectrum vs a reference one."""
    pr, prom_r, wr = detect(y_ref, grid)
    pn, prom_n, wn = detect(y_new, grid)
    if len(pr) == 0:
        return {"retention": np.nan, "invention": np.nan, "width_ratio": np.nan,
                "n_ref": 0, "n_new": len(pn)}
    n_m, _ = match(pr, pn, tol)
    retention = n_m / len(pr)
    # invented = high-prominence NEW peaks with no reference counterpart
    if len(pn):
        strong = pn[prom_n >= np.percentile(prom_n, 50)]
        n_ms, _ = match(strong, pr, tol)
        invention = 1.0 - (n_ms / max(1, len(strong)))
    else:
        invention = 0.0
    wr_ratio = (np.median(wn) / np.median(wr)) if (len(wn) and len(wr)) else np.nan
    return {"retention": float(retention), "invention": float(invention),
            "width_ratio": float(wr_ratio), "n_ref": int(len(pr)), "n_new": int(len(pn))}


def correspondence_with_nulls(R, S, analytes, grid, rng, tol=TOL, n_random=20):
    """Controls 1-2: matched vs mismatched-analyte vs random-peak correspondence.
    R, S: (n_analytes, n_bins) aligned Raman / Ag-SERS feature matrices."""
    lo, hi = float(grid.min()), float(grid.max())
    pk_r = [detect(r, grid)[0] for r in R]
    pk_s = [detect(s, grid)[0] for s in S]
    matched, mismatched, random_ = [], [], []
    n = len(analytes)
    for i in range(n):
        if len(pk_r[i]) == 0:
            continue
        m, _ = match(pk_r[i], pk_s[i], tol)
        matched.append(m / len(pk_r[i]))
        others = [j for j in range(n) if j != i]
        if others:
            mm = [match(pk_r[i], pk_s[j], tol)[0] / len(pk_r[i]) for j in others]
            mismatched.append(float(np.mean(mm)))
        dens = len(pk_s[i]) if len(pk_s[i]) else 1
        rr = []
        for _ in range(n_random):
            fake = np.sort(rng.uniform(lo, hi, size=dens))
            rr.append(match(pk_r[i], fake, tol)[0] / len(pk_r[i]))
        random_.append(float(np.mean(rr)))
    if not matched:
        return {"matched": np.nan, "mismatched": np.nan, "random": np.nan,
                "effect_vs_mismatched": np.nan, "effect_size": np.nan}
    matched = np.array(matched); mismatched = np.array(mismatched); random_ = np.array(random_)
    diff = matched - mismatched
    eff = float(diff.mean() / (diff.std() + 1e-9))
    return {"matched": float(matched.mean()), "mismatched": float(mismatched.mean()),
            "random": float(random_.mean()),
            "effect_vs_mismatched": float(diff.mean()), "effect_size": eff,
            "n_analytes": int(len(matched))}


def artefact_burden(X, grid, edge_cm=40.0):
    """Over-subtraction / edge-artefact guards."""
    Xn = np.nan_to_num(X)
    pos = np.clip(Xn, 0, None).sum(axis=1) + 1e-12
    neg = np.clip(-Xn, 0, None).sum(axis=1)
    dx = float(np.median(np.diff(grid))); k = max(1, int(edge_cm / dx))
    edge = (np.abs(Xn[:, :k]).mean(axis=1) + np.abs(Xn[:, -k:]).mean(axis=1)) / 2
    body = np.abs(Xn[:, k:-k]).mean(axis=1) + 1e-12
    return {"negative_lobe_burden": float(np.median(neg / (pos + neg))),
            "edge_artefact_ratio": float(np.median(edge / body))}


def effective_rank(X):
    Xc = np.nan_to_num(X) - np.nan_to_num(X).mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    t = s.sum()
    if t < 1e-12:
        return 1.0
    p = s / t
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))
