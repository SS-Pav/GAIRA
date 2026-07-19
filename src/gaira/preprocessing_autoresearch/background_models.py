"""Stage B0 — Ag-SERS common-background models (the primary study component).

Every model is fitted STRICTLY on training-fold Ag-SERS spectra and then applied
unchanged to validation/test spectra. No analyte labels are used to construct a
background vector, and no Raman information ever touches an Ag-SERS spectrum.

Models: none · global/fold mean subtraction · scaled mean subtraction (alpha via
least squares, NNLS, robust Huber, or low-percentile band fit) · low-rank (PCA)
component removal · robust low-rank decomposition.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import nnls


@dataclass
class BackgroundModel:
    method: str
    params: dict = field(default_factory=dict)
    b: np.ndarray | None = None        # background vector (mean)
    comps: np.ndarray | None = None    # low-rank components (k, n_bins)
    fitted_on: int = 0

    # ── fitting (training Ag-SERS only) ──
    def fit(self, X_train_sers):
        X = np.nan_to_num(np.atleast_2d(X_train_sers))
        self.fitted_on = int(len(X))
        if self.method == "none":
            return self
        self.b = X.mean(axis=0)
        if self.method in ("lowrank", "robust_lowrank"):
            k = int(self.params.get("k", 1))
            Xc = X - self.b
            if self.method == "robust_lowrank":
                # iteratively down-weight high-residual spectra (robust to outliers)
                w = np.ones(len(X))
                for _ in range(5):
                    mu = np.average(X, axis=0, weights=w)
                    r = np.linalg.norm(X - mu, axis=1)
                    s = np.median(r) + 1e-12
                    w = 1.0 / (1.0 + (r / (1.345 * s)) ** 2)
                self.b = np.average(X, axis=0, weights=w)
                Xc = (X - self.b) * w[:, None]
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            self.comps = Vt[:k]
        return self

    # ── application ──
    def transform(self, X, is_sers_mask=None):
        """Remove the fitted background from the Ag-SERS rows only."""
        X = np.nan_to_num(np.atleast_2d(X)).astype(float).copy()
        if self.method == "none" or self.b is None:
            return X
        idx = np.arange(len(X)) if is_sers_mask is None else np.where(np.asarray(is_sers_mask))[0]
        for i in idx:
            X[i] = self._remove(X[i])
        return X

    def _remove(self, x):
        m = self.method
        if m in ("mean", "fold_mean", "global_mean"):
            return x - self.b
        if m == "scaled_mean":
            a = self._alpha(x)
            return x - a * self.b
        if m in ("lowrank", "robust_lowrank"):
            r = x - self.b
            if self.comps is not None:
                for v in self.comps:
                    v = v / (np.linalg.norm(v) + 1e-12)
                    r = r - np.dot(r, v) * v
            return r
        return x

    def _alpha(self, x):
        """Scale of the background in this spectrum (estimator is a fitted param)."""
        est = self.params.get("alpha", "ls")
        b = self.b
        bb = float(np.dot(b, b)) + 1e-12
        if est == "ls":
            return float(np.dot(x, b) / bb)
        if est == "nnls":
            a, _ = nnls(b[:, None], x)
            return float(a[0])
        if est == "robust":
            a = float(np.dot(x, b) / bb)
            for _ in range(10):
                r = x - a * b
                s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
                w = 1.0 / (1.0 + (r / (1.345 * s)) ** 2)
                a = float(np.sum(w * x * b) / (np.sum(w * b * b) + 1e-12))
            return a
        if est == "percentile":
            # fit on the low-signal (background-dominated) bins only
            q = np.percentile(x, self.params.get("pct", 40))
            m = x <= q
            if m.sum() < 10:
                return float(np.dot(x, b) / bb)
            return float(np.dot(x[m], b[m]) / (np.dot(b[m], b[m]) + 1e-12))
        return float(np.dot(x, b) / bb)

    def variance_explained(self, X):
        """Fraction of Ag-SERS variance removed by this background model (Control 4)."""
        X = np.nan_to_num(np.atleast_2d(X))
        if self.method == "none" or self.b is None:
            return 0.0
        R = self.transform(X)
        v0 = float(np.var(X)) + 1e-12
        return float(1.0 - np.var(R) / v0)

    def to_dict(self):
        return {"method": self.method, "params": dict(self.params),
                "n_components": (0 if self.comps is None else int(len(self.comps))),
                "fitted_on_n_spectra": self.fitted_on}


def make(method, **params):
    return BackgroundModel(method=method, params=params)


CANDIDATES = [
    ("none", {}),
    ("mean", {}),
    ("scaled_mean", {"alpha": "ls"}),
    ("scaled_mean", {"alpha": "nnls"}),
    ("scaled_mean", {"alpha": "robust"}),
    ("scaled_mean", {"alpha": "percentile", "pct": 40}),
    ("lowrank", {"k": 1}),
    ("lowrank", {"k": 2}),
    ("lowrank", {"k": 3}),
    ("lowrank", {"k": 5}),
    ("robust_lowrank", {"k": 2}),
]
