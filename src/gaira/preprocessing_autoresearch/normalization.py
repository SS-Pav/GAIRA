"""Stage B0 — normalization, derivatives, replicate aggregation, analysis scaling.

Normalizers reuse src/gaira/preprocessing/pipeline.py where one already exists.
SNV is included ONLY as a declared negative/control baseline (the spectral audit
showed it collapses Ag-SERS replicate agreement) and is never preferred merely
because it raises a similarity metric.

Visualization-only scalings live in `viz_scale` and are never used for metrics.
"""
from __future__ import annotations
import numpy as np

from ..preprocessing.pipeline import norm_l2, norm_area, norm_snv, norm_robust  # reuse


def norm_none(y):
    return y


def norm_max(y):
    m = np.nanmax(np.abs(y))
    return y / m if m > 1e-12 else y


def norm_p95(y):
    p = np.nanpercentile(np.abs(y), 95)
    return y / p if p > 1e-12 else y


NORMALIZERS = {"none": norm_none, "l2": norm_l2, "area": norm_area, "max": norm_max,
               "p95": norm_p95, "robust": norm_robust, "snv": norm_snv}
CONTROL_ONLY = {"snv"}          # declared negative control


# ── derivatives (must follow denoising) ──
def derivative(Y, order=0):
    if order == 0:
        return Y
    D = np.gradient(np.nan_to_num(Y), axis=-1)
    if order == 2:
        D = np.gradient(D, axis=-1)
    return D


def concat_intensity_derivative(Y):
    d = derivative(Y, 1)
    def _u(A):
        n = np.linalg.norm(A, axis=-1, keepdims=True)
        return A / (n + 1e-12)
    return np.concatenate([_u(np.nan_to_num(Y)), _u(d)], axis=-1)


# ── replicate aggregation (within one analyte x modality group only) ──
def aggregate(X, method="mean"):
    """Aggregate technical replicates of ONE analyte-condition group."""
    X = np.nan_to_num(np.atleast_2d(X))
    if len(X) == 1 or method in (None, "none"):
        return X.mean(axis=0)
    if method == "mean":
        return X.mean(axis=0)
    if method == "median":
        return np.median(X, axis=0)
    if method == "ivw":
        # weight each REPLICATE by the inverse of its own noise level (residual
        # variance about the group median); per-bin weights would degenerate to
        # the plain mean because they are shared by all replicates at that bin.
        med = np.median(X, axis=0)
        v = np.var(X - med, axis=1, ddof=1) + 1e-9
        w = 1.0 / v
        w = w / w.sum()
        return np.sum(X * w[:, None], axis=0)
    if method == "huber":                    # robust location per bin
        mu = np.median(X, axis=0)
        for _ in range(8):
            r = X - mu
            s = 1.4826 * np.median(np.abs(r), axis=0) + 1e-9
            w = 1.0 / (1.0 + (np.abs(r) / (1.345 * s)) ** 2)
            mu = np.sum(w * X, axis=0) / (np.sum(w, axis=0) + 1e-12)
        return mu
    if method == "consensus":                # drop replicate outliers, then mean
        mu = np.median(X, axis=0)
        d = np.array([1 - _cos(x, mu) for x in X])
        keep = d <= (np.median(d) + 2 * (1.4826 * np.median(np.abs(d - np.median(d))) + 1e-9))
        return X[keep].mean(axis=0) if keep.sum() >= 1 else mu
    return X.mean(axis=0)


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


AGGREGATORS = ["mean", "median", "ivw", "huber", "consensus"]


# ── analysis scaling fitted on TRAIN only ──
class QuantileScaler:
    """Match a modality's intensity distribution to a training-derived reference."""
    def __init__(self, n_q=51):
        self.n_q = n_q; self.ref = None; self.src = None

    def fit(self, X_src_train, X_ref_train):
        q = np.linspace(0, 100, self.n_q)
        self.src = np.percentile(np.nan_to_num(X_src_train).ravel(), q)
        self.ref = np.percentile(np.nan_to_num(X_ref_train).ravel(), q)
        return self

    def transform(self, X):
        if self.ref is None:
            return X
        Xn = np.nan_to_num(X)
        return np.interp(Xn, self.src, self.ref)


# ── visualization-only scalings (NEVER used in metrics) ──
def viz_scale(y, mode="max1"):
    y = np.nan_to_num(y)
    if mode == "max1":
        m = np.max(np.abs(y)); return y / m if m > 1e-12 else y
    if mode == "area1":
        a = np.sum(np.abs(y)); return y / a if a > 1e-12 else y
    return y
