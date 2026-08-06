"""GAIRA V7 — Phase 04, Part A: how a spectrum projects onto the frozen LSM dictionary.

Six candidate estimators, benchmarked rather than assumed. NNLS is the incumbent because
Raman mixtures are physically non-negative — a component cannot contribute negative intensity —
but non-negativity alone does not make it the best conditioned estimator on a dictionary of 50
correlated motifs, and that is an empirical question.

**Nothing here fits anything.** Every estimator solves for activations against a fixed
dictionary. `alpha` values are read from the frozen engine config, never tuned against the
spectrum being projected.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls
from sklearn.linear_model import ARDRegression, ElasticNet, Lasso, OrthogonalMatchingPursuit, Ridge

EPS = 1e-12
METHODS = ("nnls", "lasso", "elastic_net", "ridge", "ard_bayesian", "omp")


def _clip(a):
    return np.clip(np.asarray(a, float).ravel(), 0.0, None)


def project_nnls(x: np.ndarray, D: np.ndarray, **kw) -> np.ndarray:
    """Non-negative least squares. Physically the default: no component may contribute
    negative intensity to a Raman mixture."""
    return nnls(D.T, x)[0]


def project_lasso(x: np.ndarray, D: np.ndarray, alpha: float = 1e-4, **kw) -> np.ndarray:
    m = Lasso(alpha=alpha, positive=True, max_iter=20000, fit_intercept=False)
    m.fit(D.T, x)
    return _clip(m.coef_)


def project_elastic_net(x: np.ndarray, D: np.ndarray, alpha: float = 1e-4,
                        l1_ratio: float = 0.5, **kw) -> np.ndarray:
    m = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, positive=True, max_iter=20000,
                   fit_intercept=False)
    m.fit(D.T, x)
    return _clip(m.coef_)


def project_ridge(x: np.ndarray, D: np.ndarray, alpha: float = 1e-3, **kw) -> np.ndarray:
    """Included as the negative control: it has no non-negativity and no sparsity, so if it
    wins the benchmark, the case for the physical constraints is weaker than assumed."""
    m = Ridge(alpha=alpha, fit_intercept=False)
    m.fit(D.T, x)
    return _clip(m.coef_)


def project_ard(x: np.ndarray, D: np.ndarray, **kw) -> np.ndarray:
    """Automatic relevance determination — a sparse Bayesian projection whose per-coefficient
    precisions give a natural uncertainty, at the cost of no non-negativity constraint."""
    m = ARDRegression(max_iter=300, fit_intercept=False)
    m.fit(D.T, x)
    return _clip(m.coef_)


def project_omp(x: np.ndarray, D: np.ndarray, n_nonzero: int = 8, **kw) -> np.ndarray:
    m = OrthogonalMatchingPursuit(n_nonzero_coefs=min(n_nonzero, D.shape[0] - 1),
                                  fit_intercept=False)
    m.fit(D.T, x)
    return _clip(m.coef_)


DISPATCH = {"nnls": project_nnls, "lasso": project_lasso, "elastic_net": project_elastic_net,
            "ridge": project_ridge, "ard_bayesian": project_ard, "omp": project_omp}


def project(X: np.ndarray, D: np.ndarray, method: str = "nnls", **kw) -> np.ndarray:
    """Activations of every row of `X` against dictionary `D`. Deterministic."""
    X = np.atleast_2d(np.asarray(X, float))
    return np.vstack([DISPATCH[method](x, D, **kw) for x in X])


# ── benchmark criteria ───────────────────────────────────────────────────────
def reconstruction_ev(X: np.ndarray, A: np.ndarray, D: np.ndarray) -> np.ndarray:
    R = A @ D
    num = ((X - R) ** 2).sum(axis=1)
    den = (X ** 2).sum(axis=1) + EPS
    return np.clip(1.0 - num / den, 0.0, None)


def sparsity(A: np.ndarray, thresh: float = 1e-6) -> float:
    return float((np.abs(A) > thresh).sum(axis=1).mean())


def negativity(A: np.ndarray) -> float:
    """Share of coefficient mass an unconstrained estimator puts below zero.

    Zero for the constrained methods by construction; reported so the physical cost of the
    unconstrained ones is visible rather than assumed.
    """
    neg = np.clip(-A, 0, None).sum()
    return float(neg / (np.abs(A).sum() + EPS))


def replicate_consistency(A: np.ndarray, groups: np.ndarray) -> float:
    """Mean cosine between activation vectors of replicates of the same molecule.

    The property that matters most for an inference engine: two measurements of one substance
    must land in the same place. A projection that reconstructs beautifully but scatters
    replicates is useless downstream.
    """
    N = A / (np.linalg.norm(A, axis=1, keepdims=True) + EPS)
    out = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if idx.size < 2:
            continue
        C = N[idx] @ N[idx].T
        out.append(float(C[np.triu_indices(idx.size, 1)].mean()))
    return float(np.mean(out)) if out else float("nan")


def noise_stability(x: np.ndarray, D: np.ndarray, method: str, sigma: float = 0.02,
                    n: int = 12, seed: int = 0, **kw) -> float:
    """Cosine between the activation of a spectrum and of the same spectrum plus noise.

    Deterministic given the seed. Additive Gaussian noise at 2% of the spectrum's own scale is
    a mild perturbation; an estimator whose answer moves under it is not usable on real data.
    """
    rng = np.random.default_rng(seed)
    a0 = DISPATCH[method](x, D, **kw)
    scale = sigma * float(np.abs(x).max())
    out = []
    for _ in range(n):
        a = DISPATCH[method](np.clip(x + rng.normal(0, scale, x.shape), 0, None), D, **kw)
        out.append(float(a0 @ a / (np.linalg.norm(a0) * np.linalg.norm(a) + EPS)))
    return float(np.mean(out))


def condition_diagnostics(D: np.ndarray) -> dict:
    """How badly conditioned the dictionary is — the reason the estimator choice matters.

    A dictionary of 50 correlated Raman motifs is not an orthogonal basis; the coherence and
    condition number say how much of the activation is determined by the data and how much by
    the estimator's prior.
    """
    N = D / (np.linalg.norm(D, axis=1, keepdims=True) + EPS)
    G = N @ N.T
    off = np.abs(G[np.triu_indices(D.shape[0], 1)])
    s = np.linalg.svd(D, compute_uv=False)
    return {"max_coherence": float(off.max()), "mean_coherence": float(off.mean()),
            "condition_number": float(s.max() / (s.min() + EPS)),
            "effective_rank": float(np.exp(-(lambda p: (p * np.log(p + EPS)).sum())(
                s ** 2 / (s ** 2).sum())))}
