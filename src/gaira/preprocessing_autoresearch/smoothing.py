"""Stage B0 — baseline correction and smoothing primitives.

Extends src/gaira/preprocessing/pipeline.py (ASLS, Savitzky-Golay, polynomial) with
airPLS, arPLS, rubber-band, morphological baselines and Gaussian / Whittaker /
wavelet-style denoising. All deterministic and per-spectrum (fold-independent),
so they can be cached safely. Bounded parameter ranges only.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.ndimage import gaussian_filter1d, grey_opening
from scipy.signal import savgol_filter

from ..preprocessing.pipeline import baseline_asls, baseline_poly, baseline_none  # reuse


# ── additional baselines ──────────────────────────────────────────────
def _diff2(n):
    D = diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n)).tocsc()
    return (D.T @ D).tocsc()


def baseline_airpls(y, lam=1e5, n_iter=15):
    n = len(y); DTD = _diff2(n)
    w = np.ones(n); z = y.copy()
    for it in range(1, n_iter + 1):
        W = diags(w, 0, shape=(n, n)).tocsc()
        z = spsolve((W + lam * DTD).tocsc(), w * y)
        d = y - z
        dn = d[d < 0]
        if dn.size == 0 or np.abs(dn).sum() < 1e-3 * np.abs(y).sum():
            break
        w = np.zeros(n)
        w[d < 0] = np.exp(it * np.abs(d[d < 0]) / (np.abs(dn).sum() + 1e-12))
    return z


def baseline_arpls(y, lam=1e5, ratio=1e-3, n_iter=20):
    n = len(y); DTD = _diff2(n)
    w = np.ones(n); z = y.copy()
    for _ in range(n_iter):
        W = diags(w, 0, shape=(n, n)).tocsc()
        z = spsolve((W + lam * DTD).tocsc(), w * y)
        d = y - z
        dn = d[d < 0]
        if dn.size == 0:
            break
        m, s = dn.mean(), dn.std() + 1e-12
        wt = 1.0 / (1.0 + np.exp(2 * (d - (2 * s - m)) / s))
        if np.linalg.norm(w - wt) / (np.linalg.norm(w) + 1e-12) < ratio:
            w = wt; break
        w = wt
    return z


def baseline_rubberband(y):
    """Convex-hull ('rubber band') baseline."""
    x = np.arange(len(y))
    pts = np.column_stack([x, y])
    try:
        from scipy.spatial import ConvexHull
        h = ConvexHull(pts).vertices
    except Exception:
        return baseline_poly(y, 2)
    h = np.roll(h, -np.argmin(pts[h, 0]))
    lower = h[: np.argmax(pts[h, 0]) + 1]
    lower = np.sort(lower)
    return np.interp(x, x[lower], y[lower])


def baseline_morphological(y, size=61):
    """Grey-scale opening baseline (structuring element bounded to broad features)."""
    size = int(size) | 1
    return grey_opening(y, size=size, mode="nearest")


BASELINES = {
    "none": lambda y, **k: baseline_none(y),
    "asls": lambda y, lam=1e5, p=0.01, **k: baseline_asls(y, lam=lam, p=p),
    "airpls": lambda y, lam=1e5, **k: baseline_airpls(y, lam=lam),
    "arpls": lambda y, lam=1e5, **k: baseline_arpls(y, lam=lam),
    "rubberband": lambda y, **k: baseline_rubberband(y),
    "poly3": lambda y, **k: baseline_poly(y, 3),
    "morph": lambda y, size=61, **k: baseline_morphological(y, size=size),
}


# ── smoothing / denoising ─────────────────────────────────────────────
def smooth_none(y, **k):
    return y


def smooth_savgol(y, window=9, poly=3, **k):
    w = int(window) | 1
    if len(y) <= w or poly >= w:
        return y
    return savgol_filter(y, w, int(poly))


def smooth_gaussian(y, sigma=1.0, **k):
    return gaussian_filter1d(y, float(sigma), mode="nearest")


def smooth_whittaker(y, lam=10.0, **k):
    n = len(y)
    W = diags(np.ones(n), 0, shape=(n, n)).tocsc()
    return spsolve((W + lam * _diff2(n)).tocsc(), y)


def smooth_wavelet(y, level=2, **k):
    """Wavelet-style denoising via an à-trous Haar decomposition with soft
    thresholding of the finest detail levels (no pywt dependency)."""
    a = y.astype(float).copy()
    details = []
    for j in range(int(level)):
        sm = gaussian_filter1d(a, 2 ** j, mode="nearest")
        details.append(a - sm)
        a = sm
    out = a
    for d in details:
        sigma = 1.4826 * np.median(np.abs(d - np.median(d))) + 1e-12
        thr = sigma * np.sqrt(2 * np.log(max(len(y), 2)))
        out = out + np.sign(d) * np.maximum(np.abs(d) - thr, 0.0)
    return out


SMOOTHERS = {"none": smooth_none, "savgol": smooth_savgol, "gaussian": smooth_gaussian,
             "whittaker": smooth_whittaker, "wavelet": smooth_wavelet}
