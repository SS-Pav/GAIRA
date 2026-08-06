"""GAIRA V7 — Phase 03: five candidate soft-membership models.

Every model produces `S ∈ ℝ₊^{M×K}` with rows summing to one (contract C-08): a CSM's
membership is a *distribution* over themes, not a label. Soft membership is not a convenience —
shared biochemical structure genuinely belongs to more than one theme, and forcing one parent
destroys exactly the information the theme layer exists to carry.

None of these models sees a chemistry label. Labels enter only after `K` is fixed and the
themes are validated.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import NMF
from sklearn.mixture import GaussianMixture

EPS = 1e-12
MODELS = ("archetypal", "sparse_nmf", "fuzzy_cmeans", "diffusion_gmm", "graph_regularised_nmf")
SEED = 0


def _rownorm(S: np.ndarray) -> np.ndarray:
    S = np.clip(np.asarray(S, float), 0.0, None)
    return S / (S.sum(axis=1, keepdims=True) + EPS)


# ── 1. archetypal analysis ───────────────────────────────────────────────────
def archetypal(X: np.ndarray, K: int, n_iter: int = 300, seed: int = SEED) -> dict:
    """Themes as convex extremes of the CSM cloud; memberships are convex weights.

    Archetypal analysis is the model whose assumptions match the object best. An archetype is
    a convex combination of observed CSMs, so every theme is a spectrum that *could* exist in
    this corpus rather than an arbitrary direction; and memberships come out convex —
    non-negative and summing to one — by construction rather than by post-hoc normalisation.
    It also naturally produces themes at the edges of the space, which is what "a theme" means
    when the space is a continuum: not a cluster centre, but a pole.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    K = min(K, n)
    # deterministic furthest-point initialisation
    idx = [int(np.argmax(np.linalg.norm(X - X.mean(0), axis=1)))]
    while len(idx) < K:
        d = np.min([np.linalg.norm(X - X[j], axis=1) for j in idx], axis=0)
        idx.append(int(np.argmax(d)))
    B = np.zeros((K, n))
    B[np.arange(K), idx] = 1.0
    Z = B @ X
    S = _rownorm(np.abs(rng.normal(size=(n, K))) + 0.1)

    for _ in range(n_iter):
        for _ in range(3):                      # S-step: project each CSM onto the simplex
            G = (S @ Z - X) @ Z.T
            S = _rownorm(np.clip(S - 0.5 * G / (np.abs(G).max() + EPS) * 0.1, 0, None))
        for _ in range(3):                      # B-step: archetypes stay convex in the data
            G = S.T @ (S @ (B @ X) - X) @ X.T
            B = _rownorm(np.clip(B - 0.5 * G / (np.abs(G).max() + EPS) * 0.1, 0, None))
        Z = B @ X
    return {"S": _rownorm(S), "themes": Z, "model": "archetypal"}


# ── 2. sparse NMF on the CSM basis ───────────────────────────────────────────
def sparse_nmf(X: np.ndarray, K: int, alpha: float = 0.02, seed: int = SEED) -> dict:
    """Themes as non-negative spectral parts; memberships from the row-normalised loadings."""
    m = NMF(n_components=min(K, X.shape[0]), init="nndsvda", random_state=seed,
            max_iter=3000, alpha_W=alpha, l1_ratio=1.0)
    W = m.fit_transform(np.clip(X, 0, None))
    return {"S": _rownorm(W), "themes": m.components_, "model": "sparse_nmf"}


# ── 3. relational fuzzy c-means on the frozen geometry ───────────────────────
def fuzzy_cmeans(D: np.ndarray, X: np.ndarray, K: int, m: float = 1.8,
                 n_iter: int = 300, seed: int = SEED) -> dict:
    """Fuzzy c-means driven by the Phase 02.5 distance matrix rather than by the spectra.

    The only candidate that uses the *geometry* as its objective. `m = 1.8` is the standard
    fuzzifier; at m → 1 it collapses to hard k-means, which the continuum result forbids.
    """
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    K = min(K, n)
    U = _rownorm(rng.uniform(size=(n, K)) + 0.1)
    for _ in range(n_iter):
        Um = U ** m
        # relational centres: distance from each point to each cluster in the given metric
        num = Um.T @ (D ** 2)
        den = Um.sum(axis=0)[:, None]
        dk = (num / (den + EPS)).T - 0.5 * ((Um.T @ (D ** 2) @ Um).diagonal()
                                            / (den.ravel() ** 2 + EPS))[None, :]
        dk = np.clip(dk, EPS, None)
        inv = dk ** (-1.0 / (m - 1.0))
        U = _rownorm(inv)
    themes = (U ** m).T @ X / ((U ** m).sum(axis=0)[:, None] + EPS)
    return {"S": U, "themes": themes, "model": "fuzzy_cmeans"}


# ── 4. Gaussian mixture on diffusion coordinates ─────────────────────────────
def diffusion_gmm(coords: np.ndarray, X: np.ndarray, K: int, seed: int = SEED) -> dict:
    """Responsibilities of a mixture fitted in the diffusion space.

    Diffusion coordinates are slow modes of a random walk on the similarity graph, so a mixture
    there models themes as regions of a *manifold* rather than as balls in the ambient 676-bin
    space — the right shape for a continuum with denser regions.
    """
    C = np.asarray(coords, float)
    K = min(K, C.shape[0] - 1)
    g = GaussianMixture(n_components=K, covariance_type="diag", random_state=seed,
                        n_init=5, max_iter=500, reg_covar=1e-5).fit(C)
    S = _rownorm(g.predict_proba(C))
    themes = S.T @ X / (S.sum(axis=0)[:, None] + EPS)
    return {"S": S, "themes": themes, "model": "diffusion_gmm"}


# ── 5. graph-regularised NMF ─────────────────────────────────────────────────
def graph_regularised_nmf(X: np.ndarray, A: np.ndarray, K: int, lam: float = 0.5,
                          n_iter: int = 400, seed: int = SEED) -> dict:
    """NMF on the spectra with a Laplacian penalty from the Phase 02.5 neighbour graph.

    The hybrid: spectral identity from the factorisation, neighbourhood structure from the
    graph. `lam` trades them off; it is swept in the sensitivity arm rather than tuned here.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    K = min(K, n)
    Xp = np.clip(X, 0, None)
    W = np.abs(rng.normal(size=(n, K))) + 0.1
    H = np.abs(rng.normal(size=(K, d))) + 0.1
    Adj = np.clip(np.asarray(A, float), 0, None)
    np.fill_diagonal(Adj, 0.0)
    Dg = np.diag(Adj.sum(axis=1))
    for _ in range(n_iter):
        H *= (W.T @ Xp) / (W.T @ W @ H + EPS)
        W *= (Xp @ H.T + lam * Adj @ W) / (W @ H @ H.T + lam * Dg @ W + EPS)
    return {"S": _rownorm(W), "themes": H, "model": "graph_regularised_nmf"}


def fit(model: str, K: int, X: np.ndarray, D: np.ndarray, coords: np.ndarray,
        A: np.ndarray, **kw) -> dict:
    if model == "archetypal":
        return archetypal(X, K, **kw)
    if model == "sparse_nmf":
        return sparse_nmf(X, K, **kw)
    if model == "fuzzy_cmeans":
        return fuzzy_cmeans(D, X, K, **kw)
    if model == "diffusion_gmm":
        return diffusion_gmm(coords, X, K, **kw)
    if model == "graph_regularised_nmf":
        return graph_regularised_nmf(X, A, K, **kw)
    raise ValueError(f"unknown model {model}")
