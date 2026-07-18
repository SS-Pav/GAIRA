"""GAIRA Demo v1 — preprocessing (baseline + smoothing + L2).

Lightweight stand-in for the production pipeline. Not optimised for
accuracy — adequate to make the End-to-End Workflow tab show a real
processed spectrum given any input.
"""
from __future__ import annotations

import numpy as np


def smooth_savgol(y: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
    if window % 2 == 0:
        window += 1
    if window <= poly + 1:
        return y.copy()
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(y, window, poly)
    except Exception:
        # Fallback: moving average
        k = max(3, window // 2 | 1)
        kernel = np.ones(k) / k
        return np.convolve(y, kernel, mode="same")


def asls_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.01,
                   n_iter: int = 8) -> np.ndarray:
    """Asymmetric Least Squares baseline (Eilers & Boelens 2005)."""
    try:
        from scipy.sparse import diags, eye
        from scipy.sparse.linalg import spsolve
    except Exception:
        # Fallback: very gentle polynomial detrend
        x = np.arange(len(y))
        coef = np.polyfit(x, y, deg=3)
        return np.polyval(coef, x)

    n = len(y)
    D = diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n)).tocsc()
    DT = D.T
    DTD = DT @ D
    w = np.ones(n)
    z = y.copy()
    for _ in range(n_iter):
        W = diags(w, 0, shape=(n, n)).tocsc()
        Z = (W + lam * DTD).tocsc()
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def l2_normalize(y: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(y))
    return y / n if n > 1e-12 else y


def preprocess(wavenumber: np.ndarray, intensity: np.ndarray) -> dict:
    """Return dict with raw_intensity, baseline, processed_intensity."""
    raw = np.asarray(intensity, dtype=float)
    smoothed = smooth_savgol(raw)
    baseline = asls_baseline(smoothed)
    sub = smoothed - baseline
    # Re-zero floor then L2 normalize.
    sub = np.clip(sub, 0.0, None)
    processed = l2_normalize(sub)
    return {
        "wavenumber": np.asarray(wavenumber, dtype=float),
        "raw_intensity": raw,
        "baseline": baseline,
        "smoothed": smoothed,
        "processed_intensity": processed,
        "summary": "ASLS baseline + Savitzky–Golay smoothing + L2 normalization",
    }
