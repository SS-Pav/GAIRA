"""I4 — non-negative basis (NMF) representation (§7).

Learns a SHARED non-negative basis on joint training data, then represents spectra
as non-negative activation vectors. Requires non-negative input — this module must
be given a non-negative preprocessing (e.g. L2 on baseline-corrected spectra, clipped
at 0). It raises on signed input rather than silently transforming it. Basis spectra
map back to wavenumber space directly. Fitted on training data only.
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from .base import Representation


class NMFRepresentation(Representation):
    def __init__(self, grid, basis):
        super().__init__(name="I4_nmf_basis", branch="interpretable", grid=grid,
                         modality_specific=False, params={"n_basis": basis.shape[0]})
        self.basis = basis            # (n_basis, n_bins), >=0
        self.n_features = basis.shape[0]
        from sklearn.decomposition import NMF as _N
        self._nmf = _N(n_components=basis.shape[0], init="custom", max_iter=1)

    def transform(self, X, modality=None):
        from scipy.optimize import nnls
        X = np.clip(np.nan_to_num(np.atleast_2d(X)), 0, None)
        W = np.vstack([nnls(self.basis.T, x)[0] for x in X])
        n = np.linalg.norm(W, axis=1, keepdims=True)
        return W / (n + 1e-12)

    def feature_wavenumbers(self):
        return [float(self.grid[int(np.argmax(b))]) for b in self.basis]


def _check_nonneg(X, max_neg_mass=0.15):
    """Accept ESSENTIALLY non-negative data (small baseline undershoots → clipped);
    reject genuinely signed data (SNV/derivative, ~half the mass negative)."""
    Xn = np.nan_to_num(X)
    neg_mass = np.abs(np.clip(Xn, None, 0)).sum() / (np.abs(Xn).sum() + 1e-12)
    if neg_mass > max_neg_mass:
        raise ValueError(f"I4 NMF requires (essentially) non-negative input; negative mass "
                         f"fraction={neg_mass:.2f} > {max_neg_mass} (this looks like SNV/derivative). "
                         "Use a non-negative preprocessing such as L2 on baseline-corrected spectra.")
    return neg_mass


def fit_nmf_basis(X_train_nonneg, grid, n_basis=16, seed=0):
    _check_nonneg(X_train_nonneg)
    Xt = normalize(np.clip(np.nan_to_num(X_train_nonneg), 0, None))
    m = NMF(n_components=n_basis, init="nndsvda", random_state=seed, max_iter=1000).fit(Xt)
    return NMFRepresentation(grid, m.components_)
