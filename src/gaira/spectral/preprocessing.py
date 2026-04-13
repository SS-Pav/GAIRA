"""
Spectral preprocessing — local query-time pipeline.

For NPZ-based datasets: data is already preprocessed during ingestion
(polynomial baseline, interpolation, vector normalization). We apply
L2 normalization here as the query-time step.

For raw CSV datasets (HCC holdout): apply the full preprocessing stack
used in the successful spectral query v1 work:
  1. Crop to fingerprint region
  2. Interpolate to shared wavenumber axis
  3. AsLS baseline correction (lambda=1e5, p=0.001)
  4. Savitzky-Golay smoothing (window=11, order=3)
  5. L2 vector normalization
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from gaira.spectral.dataset_loader import SpectralDataset


@dataclass
class PreprocessingSummary:
    normalization: str
    baseline: str
    smoothing: str
    notes: str
    pipeline: str  # "npz_l2" or "raw_asls_sg_l2"


def _asls_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.001,
                    n_iter: int = 10) -> np.ndarray:
    """Asymmetric Least Squares baseline estimation (Eilers & Boelens 2005)."""
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


def preprocess(ds: SpectralDataset) -> tuple[np.ndarray, PreprocessingSummary]:
    """Apply query-time preprocessing.

    For NPZ-ingested datasets: L2 normalization only (already preprocessed).
    For raw CSV datasets (HCC holdout): full AsLS + SG + L2 pipeline.
    """
    if ds.dataset_id == "hcc_holdout_vornoli2020":
        return _preprocess_raw(ds)
    return _preprocess_npz(ds)


def _preprocess_npz(ds: SpectralDataset) -> tuple[np.ndarray, PreprocessingSummary]:
    """NPZ data: already baseline-corrected and smoothed during ingestion."""
    norms = np.linalg.norm(ds.X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    X_norm = ds.X / norms

    return X_norm, PreprocessingSummary(
        normalization="L2 vector norm",
        baseline="Pre-applied (polynomial, from embedding pipeline v2)",
        smoothing="Pre-applied (from embedding pipeline)",
        notes="Spectra were baseline-corrected and smoothed during ingestion. "
              "L2 normalization applied at query time.",
        pipeline="npz_l2",
    )


def _preprocess_raw(ds: SpectralDataset) -> tuple[np.ndarray, PreprocessingSummary]:
    """Raw CSV data: apply full spectral-query-v1 preprocessing stack.

    Pipeline: AsLS baseline → SG smoothing → L2 normalization.
    Matches the preprocessing used in the successful spectral query v1 HCC analysis.
    """
    X = ds.X.copy()
    n_spectra = X.shape[0]

    # 1. AsLS baseline correction per spectrum
    for i in range(n_spectra):
        baseline = _asls_baseline(X[i], lam=1e5, p=0.001, n_iter=10)
        X[i] = X[i] - baseline

    # 2. Savitzky-Golay smoothing
    for i in range(n_spectra):
        X[i] = savgol_filter(X[i], window_length=11, polyorder=3)

    # 3. L2 vector normalization
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    X_norm = X / norms

    return X_norm, PreprocessingSummary(
        normalization="L2 vector norm",
        baseline="AsLS (lambda=1e5, p=0.001, 10 iterations)",
        smoothing="Savitzky-Golay (window=11, order=3)",
        notes="Full local preprocessing applied at query time: AsLS baseline correction, "
              "SG smoothing, L2 normalization. Matches spectral query v1 pipeline.",
        pipeline="raw_asls_sg_l2",
    )
