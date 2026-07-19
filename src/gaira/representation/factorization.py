"""NMF / sparse PCA / factor analysis (Phase 2 §10).

NMF requires non-negative input — it is applied ONLY to non-negative
representations (L2 on non-negative spectra), never to signed SNV or
derivative reps. The guard raises if that rule is violated.
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import NMF, SparsePCA, FactorAnalysis


def is_nonnegative(X, tol=1e-8):
    return bool(np.nanmin(X) >= -tol)


def fit_nmf(X, n_components=6, seed=0):
    if not is_nonnegative(X):
        raise ValueError("NMF requires non-negative input; got signed representation "
                         "(SNV/derivative). Do not apply NMF here.")
    Xc = np.clip(np.nan_to_num(X), 0, None)
    m = NMF(n_components=n_components, init="nndsvda", random_state=seed, max_iter=1000).fit(Xc)
    W = m.transform(Xc)
    return {"components": m.components_, "W": W, "reconstruction_err": float(m.reconstruction_err_),
            "n_components": n_components}


def fit_sparse_pca(X, n_components=6, seed=0, alpha=1.0):
    m = SparsePCA(n_components=n_components, alpha=alpha, random_state=seed, max_iter=200).fit(X)
    dens = float(np.mean(np.abs(m.components_) > 1e-6))
    return {"components": m.components_, "nonzero_fraction": dens, "n_components": n_components}


def fit_factor_analysis(X, n_components=6, seed=0):
    m = FactorAnalysis(n_components=n_components, random_state=seed).fit(X)
    return {"components": m.components_, "noise_variance": m.noise_variance_.tolist(),
            "n_components": n_components}
