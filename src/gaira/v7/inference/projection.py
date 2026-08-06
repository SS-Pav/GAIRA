"""GAIRA V7 — Phase 05, Step 1: direct projection onto the frozen CSM basis.

The canonical inference representation. A spectrum becomes a 49-dimensional non-negative CSM
activation vector and nothing else is fitted — no clustering, no factorisation, no manifold
learning, no embedding. Phase 04 measured why this is the right layer: chemistry-class
generalisation to unseen molecules rises 0.608 → 0.855 from raw spectrum to CSM, then falls to
0.405 at the theme layer.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

EPS = 1e-12


def project(X: np.ndarray, CSM: np.ndarray) -> np.ndarray:
    """Non-negative projection onto the frozen CSM dictionary.

    NNLS rather than a penalised estimator: at inference the question is what mixture of frozen
    components explains this spectrum, and a Raman mixture cannot contain a negative component.
    No regularisation parameter means no quantity that could be tuned per spectrum.
    """
    X = np.atleast_2d(np.asarray(X, float))
    return np.vstack([nnls(CSM.T, np.clip(x, 0, None))[0] for x in X])


def diagnostics(X: np.ndarray, A: np.ndarray, CSM: np.ndarray) -> dict:
    """Everything Step 1 must return, per spectrum.

    These are also the raw material for open-set rejection: a spectrum the atlas cannot explain
    shows up here first, in the residual, before any similarity is computed.
    """
    X = np.atleast_2d(np.asarray(X, float))
    R = A @ CSM
    resid = ((X - R) ** 2).sum(axis=1)
    total = (X ** 2).sum(axis=1) + EPS
    n = A.shape[1]
    P = A / (A.sum(axis=1, keepdims=True) + EPS)
    ent = -(np.where(P > 0, P * np.log(P + EPS), 0.0)).sum(axis=1) / np.log(n)
    l1 = np.abs(A).sum(axis=1)
    l2 = np.linalg.norm(A, axis=1)
    hoyer = (np.sqrt(n) - l1 / (l2 + EPS)) / (np.sqrt(n) - 1 + EPS)
    return {
        "reconstruction": R,
        "residual": np.sqrt(resid),
        "residual_fraction": resid / total,
        "explained_variance": np.clip(1.0 - resid / total, 0.0, None),
        "component_sparsity": hoyer,
        "n_active_csms": (A > 1e-9).sum(axis=1),
        "activation_entropy": ent,
    }
