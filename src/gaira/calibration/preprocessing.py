"""Calibration preprocessing — applies the same stack the raw-CSV path uses.

Mirrors gaira.spectral.preprocessing._preprocess_raw:
  1. Crop to fingerprint region (>= 400 cm⁻¹ by default)
  2. Interpolate to a master axis if one is provided
  3. AsLS baseline correction (lambda=1e5, p=0.001, 10 iter)
  4. Savitzky-Golay smoothing (window=11, order=3)
  5. L2 vector normalization

Duplicated here rather than imported from spectral.preprocessing to keep
the calibration module self-contained. Identical numerics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter


@dataclass
class CalibrationPreprocessed:
    X: np.ndarray                 # (n_spectra, n_wn_fp) L2-normalized
    wavenumbers: np.ndarray        # (n_wn_fp,) cropped axis
    pipeline: str                  # "calibration_asls_sg_l2"
    crop_range: tuple[float, float]


def _asls_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.001,
                    n_iter: int = 10) -> np.ndarray:
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    L = len(y)
    D = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.T)
    w = np.ones(L)
    for _ in range(n_iter):
        W = sparse.spdiags(w, 0, L, L)
        Z = (W + D).tocsc()
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def preprocess_calibration(
    X_raw: np.ndarray,
    wavenumbers: np.ndarray,
    crop_range: tuple[float, float] = (400.0, 1800.0),
    asls_lambda: float = 1e5,
    asls_p: float = 0.001,
    asls_iter: int = 10,
    sg_window: int = 11,
    sg_order: int = 3,
) -> CalibrationPreprocessed:
    """Apply AsLS + SG + L2 pipeline to raw calibration spectra."""
    wn = np.asarray(wavenumbers, dtype=float)
    X = np.asarray(X_raw, dtype=float).copy()

    # 1. Crop to fingerprint region
    lo, hi = crop_range
    mask = (wn >= lo) & (wn <= hi)
    if mask.sum() < 50:
        raise ValueError(
            f"Crop range {crop_range} leaves only {mask.sum()} points "
            f"(axis covers {wn.min():.1f} to {wn.max():.1f})."
        )
    wn_fp = wn[mask]
    X = X[:, mask]

    # 2. AsLS baseline per spectrum
    for i in range(X.shape[0]):
        X[i] = X[i] - _asls_baseline(X[i], lam=asls_lambda, p=asls_p, n_iter=asls_iter)

    # 3. Savitzky-Golay smoothing
    if sg_window >= 5 and X.shape[1] >= sg_window:
        for i in range(X.shape[0]):
            X[i] = savgol_filter(X[i], window_length=sg_window, polyorder=sg_order)

    # 4. L2 vector normalization
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    X = X / norms

    return CalibrationPreprocessed(
        X=X, wavenumbers=wn_fp,
        pipeline="calibration_asls_sg_l2",
        crop_range=crop_range,
    )
