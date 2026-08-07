"""GAIRA V7 — Phase 07: candidate factorisations of the Chemistry Evidence matrix.

The object factorised is `Ev ∈ ℝ₊^{375×16}` — spectra × **chemistry evidence**. Not spectra, not
CSM activations, not a similarity matrix, not a graph. A biochemical *programme* is therefore a
pattern of chemistry co-occurrence: which chemistries tend to be evidenced together.

    Ev ≈ W P,   W ∈ ℝ₊^{375×K}  (per-spectrum programme activation)
                P ∈ ℝ₊^{K×16}   (which chemistries each programme uses)

This is a different object from the archived Meta Components (A-15), which factorised spectra ×
*motif* activations and retained 0.185 of the CSM layer's information before being discarded.
Whether the difference is enough is the open question this phase exists to answer, and the answer
is allowed to be no.

Six families are fitted. PCA and ICA are **controls only** — they are signed and therefore cannot
be biochemical programmes under P-02, but they bound what any linear method can reconstruct.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12
K_GRID = tuple(range(2, 17))
FAMILIES = ("nmf", "sparse_nmf", "orthogonal_nmf", "semi_nmf", "pca_control", "ica_control")
# Sparse NMF needs its penalty swept: at alpha = 0.2 the L1 term drives the loadings to zero and
# the family scores EV -0.401 at every K, which would silently exclude one of the six required
# candidates for a reason that is a hyperparameter rather than a result.
SPARSE_ALPHAS = (0.005, 0.02, 0.05, 0.1)
NON_NEGATIVE = ("nmf", "sparse_nmf", "orthogonal_nmf")
CONTROLS = ("pca_control", "ica_control")
SEED = 0


def _rownorm(M):
    return np.asarray(M, float) / (np.linalg.norm(M, axis=1, keepdims=True) + EPS)


def fit(family: str, Ev: np.ndarray, K: int, seed: int = SEED, alpha: float = 0.2,
        n_iter: int = 400) -> dict:
    """Fit one factorisation. Deterministic given the seed."""
    Ev = np.clip(np.asarray(Ev, float), 0.0, None)
    if family == "nmf":
        from sklearn.decomposition import NMF
        m = NMF(n_components=K, init="nndsvda", random_state=seed, max_iter=4000, tol=1e-6)
        W = m.fit_transform(Ev)
        return {"family": family, "K": K, "W": W, "P": m.components_}
    if family == "sparse_nmf":
        from sklearn.decomposition import NMF
        m = NMF(n_components=K, init="nndsvda", random_state=seed, max_iter=4000, tol=1e-6,
                alpha_H=alpha, l1_ratio=1.0)
        W = m.fit_transform(Ev)
        return {"family": family, "K": K, "W": W, "P": m.components_, "alpha": alpha}
    if family == "orthogonal_nmf":
        return _orthogonal_nmf(Ev, K, seed, n_iter)
    if family == "semi_nmf":
        return _semi_nmf(Ev, K, seed, n_iter)
    if family == "pca_control":
        from sklearn.decomposition import PCA
        m = PCA(n_components=K, random_state=seed)
        W = m.fit_transform(Ev)
        return {"family": family, "K": K, "W": W, "P": m.components_, "mean": m.mean_,
                "signed": True}
    if family == "ica_control":
        from sklearn.decomposition import FastICA
        m = FastICA(n_components=K, random_state=seed, max_iter=2000, tol=1e-4,
                    whiten="unit-variance")
        W = m.fit_transform(Ev)
        return {"family": family, "K": K, "W": W, "P": m.mixing_.T, "mean": m.mean_,
                "signed": True}
    raise ValueError(f"unknown family {family}")


def _orthogonal_nmf(Ev, K, seed, n_iter) -> dict:
    """NMF with an orthogonality penalty on the programme loadings.

    `min ‖Ev − WP‖² + λ‖PPᵀ − I‖²` by multiplicative updates. Orthogonality is what
    *disentangled* means here: two programmes that load on the same chemistries are two names for
    one programme. The penalty is a preference, not a constraint — programmes may still overlap
    where the chemistry genuinely does.
    """
    from sklearn.decomposition import NMF
    base = NMF(n_components=K, init="nndsvda", random_state=seed, max_iter=1000).fit(Ev)
    W, P = base.transform(Ev) + EPS, base.components_ + EPS
    lam = 0.1
    for _ in range(n_iter):
        W *= (Ev @ P.T) / (W @ P @ P.T + EPS)
        num = W.T @ Ev + lam * P
        den = W.T @ W @ P + lam * (P @ P.T @ P) + EPS
        P *= num / den
        W, P = np.clip(W, 0, None), np.clip(P, 0, None)
    return {"family": "orthogonal_nmf", "K": K, "W": W, "P": P, "lambda": lam}


def _semi_nmf(Ev, K, seed, n_iter) -> dict:
    """Semi-NMF: `W ≥ 0`, `P` unconstrained (Ding et al. 2010).

    Activations stay non-negative — a spectrum cannot contain a negative amount of a programme —
    while a programme may *subtract* chemistry evidence. That is a weaker physical claim than
    full NMF and it is included to measure what the non-negativity of `P` costs in reconstruction.
    """
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(seed)
    km = KMeans(n_clusters=K, random_state=seed, n_init=20).fit(Ev)
    W = np.zeros((len(Ev), K))
    W[np.arange(len(Ev)), km.labels_] = 1.0
    W += 0.2
    for _ in range(n_iter):
        P = np.linalg.pinv(W.T @ W) @ W.T @ Ev
        A = Ev @ P.T
        B = P @ P.T
        Ap, An = np.clip(A, 0, None), np.clip(-A, 0, None)
        Bp, Bn = np.clip(B, 0, None), np.clip(-B, 0, None)
        W *= np.sqrt((Ap + W @ Bn) / (An + W @ Bp + EPS))
        W = np.clip(W, 0, None)
    P = np.linalg.pinv(W.T @ W) @ W.T @ Ev
    return {"family": "semi_nmf", "K": K, "W": W, "P": P, "signed_loadings": True}


def project(model: dict, Ev_new: np.ndarray) -> np.ndarray:
    """Frozen projection of new Chemistry Evidence onto the learned programmes.

    This is the inference path: no fitting, no randomness, and the answer for a spectrum depends
    only on that spectrum's chemistry evidence and the frozen programme dictionary.
    """
    Ev_new = np.atleast_2d(np.asarray(Ev_new, float))
    P = model["P"]
    if model["family"] in CONTROLS:
        return (Ev_new - model["mean"]) @ np.linalg.pinv(P)
    if model["family"] == "semi_nmf":
        Wn = np.clip(Ev_new @ np.linalg.pinv(P), 0.0, None)
        return Wn
    from scipy.optimize import nnls
    return np.vstack([nnls(P.T, np.clip(e, 0, None))[0] for e in Ev_new])


def reconstruct(model: dict, W: np.ndarray) -> np.ndarray:
    R = np.asarray(W, float) @ model["P"]
    if model["family"] in CONTROLS:
        R = R + model["mean"]
    return R


# ── properties of a factorisation ────────────────────────────────────────────
def reconstruction(Ev, model, W=None) -> dict:
    W = model["W"] if W is None else W
    R = reconstruct(model, W)
    resid = Ev - R
    ss = (Ev - Ev.mean(axis=0)) ** 2
    return {"rmse": float(np.sqrt((resid ** 2).mean())),
            "explained_variance": float(1.0 - (resid ** 2).sum() / (ss.sum() + EPS)),
            "mean_cosine": float((_rownorm(Ev) * _rownorm(np.clip(R, 0, None))).sum(axis=1).mean()),
            "relative_frobenius": float(np.linalg.norm(resid) / (np.linalg.norm(Ev) + EPS))}


def per_axis_reconstruction(Ev, model, axis_names, W=None) -> "pd.DataFrame":
    import pandas as pd
    W = model["W"] if W is None else W
    R = reconstruct(model, W)
    rows = []
    for k, n in enumerate(axis_names):
        e, r = Ev[:, k], R[:, k]
        ss = ((e - e.mean()) ** 2).sum()
        rows.append({"chemistry_axis": n, "rmse": float(np.sqrt(((e - r) ** 2).mean())),
                     "explained_variance": float(1.0 - ((e - r) ** 2).sum() / (ss + EPS)),
                     "mean_evidence": float(e.mean())})
    return pd.DataFrame(rows)


def sparsity(P) -> float:
    """Mean Hoyer sparsity of the programme loadings. 1 = one chemistry each, 0 = uniform."""
    n = P.shape[1]
    out = []
    for p in np.abs(np.asarray(P, float)):
        l1, l2 = p.sum(), np.linalg.norm(p)
        out.append((np.sqrt(n) - l1 / (l2 + EPS)) / (np.sqrt(n) - 1 + EPS))
    return float(np.mean(out))


def overlap(P) -> np.ndarray:
    """Pairwise cosine between programme loadings — the disentanglement matrix."""
    N = _rownorm(np.abs(np.asarray(P, float)))
    return np.clip(N @ N.T, 0.0, 1.0)


def max_single_axis_share(P) -> float:
    """The largest share any one programme places on a single chemistry axis.

    The direct test of "a programme is not one chemistry class": at 1.0 the programme is a basis
    vector of the input and the factorisation has learned a permutation.
    """
    A = np.abs(np.asarray(P, float))
    share = A / (A.sum(axis=1, keepdims=True) + EPS)
    return float(share.max())


def redundancy(P) -> float:
    if P.shape[0] < 2:
        return 0.0
    O = overlap(P)
    return float(O[np.triu_indices(P.shape[0], 1)].max())


def mean_overlap(P) -> float:
    if P.shape[0] < 2:
        return 0.0
    O = overlap(P)
    return float(O[np.triu_indices(P.shape[0], 1)].mean())


def activation_entropy(W) -> float:
    A = np.clip(np.asarray(W, float), 0, None)
    Q = A / (A.sum(axis=1, keepdims=True) + EPS)
    K = A.shape[1]
    return float((-(np.where(Q > 0, Q * np.log(Q + EPS), 0.0)).sum(axis=1) /
                  np.log(max(K, 2))).mean())


def dominance(W) -> float:
    """Share of spectra whose top programme is the single most-used programme.

    A factorisation where one programme wins for most spectra is a background, not a programme
    set. Phase 04.5's MC-03 dominated 233 of 375 spectra and that was the tell.
    """
    top = np.argmax(np.clip(np.asarray(W, float), 0, None), axis=1)
    _, n = np.unique(top, return_counts=True)
    return float(n.max() / n.sum())


def effective_rank(M) -> float:
    X = np.asarray(M, float) - np.asarray(M, float).mean(axis=0)
    s = np.linalg.svd(X, compute_uv=False)
    p = s ** 2 / ((s ** 2).sum() + EPS)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))
