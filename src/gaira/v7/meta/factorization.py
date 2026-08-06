"""GAIRA V7 — Phase 04.5: hierarchical NMF over frozen CSM activations.

The object factorised is `A ∈ ℝ₊^{375×49}` — spectra × frozen CSM activations. **Not** spectra,
**not** a similarity matrix, **not** a graph. A Meta Component is therefore a pattern of *motif
usage*: which frozen CSMs a spectrum tends to switch on together. Two spectra can share a Meta
Component without sharing a band.

    A ≈ W H,    W ∈ ℝ₊^{375×K}  (per-spectrum programme activation)
                H ∈ ℝ₊^{K×49}   (which CSMs each programme uses)

Two variants are fitted and compared: plain NMF, and NMF with a graph-Laplacian penalty over
the frozen Phase 02.5 CSM geometry that encourages *nearby CSMs to co-activate*. The penalty is
one-sided by construction — it rewards smoothness, it never pushes distant CSMs apart, and it
cannot create a cluster the activations do not support.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import NMF

EPS = 1e-12
K_GRID = (2, 3, 4, 5, 6, 8, 10, 12)
VARIANTS = ("plain", "geometry_regularised")
SEED = 0


def _rownorm(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + EPS)


def csm_graph_laplacian(D_csm: np.ndarray, k: int = 5) -> np.ndarray:
    """**Combinatorial** Laplacian `L = D − W` of the frozen CSM k-NN graph.

    Combinatorial, not symmetric-normalised, and the distinction is not cosmetic. The prior is
    supposed to be a pure smoothness reward: `tr(H L Hᵀ) = ½ Σ_ij w_ij ‖h_·i − h_·j‖²`, which
    is exactly zero when every CSM carries the same loading and grows only with *differences*
    between neighbours. The symmetric-normalised Laplacian `I − D^-½ W D^-½` has null space
    `D^½1` rather than `1`, so it penalises a uniform loading in proportion to each CSM's
    degree — a hidden preference for loading on low-degree CSMs that nothing in the design
    called for. The unit test caught it.

    Built from the Phase 02.5 distance matrix, which is frozen and simply read. The graph is
    symmetrised by union rather than intersection so a CSM that is somebody's neighbour is
    connected even when the relation is not reciprocal — the prior is meant to be permissive.
    """
    n = D_csm.shape[0]
    Wg = np.zeros((n, n))
    for i in range(n):
        for j in np.argsort(D_csm[i])[1:k + 1]:
            sigma = float(np.median(D_csm[D_csm > 0])) + EPS
            w = float(np.exp(-(D_csm[i, j] ** 2) / (sigma ** 2)))
            Wg[i, j] = max(Wg[i, j], w)
            Wg[j, i] = max(Wg[j, i], w)
    return np.diag(Wg.sum(axis=1)) - Wg, Wg


def fit_plain(A: np.ndarray, K: int, seed: int = SEED, alpha_H: float = 0.0) -> dict:
    """Plain NMF on the activation matrix. Deterministic given the seed."""
    m = NMF(n_components=K, init="nndsvda", random_state=seed, max_iter=4000,
            alpha_H=alpha_H, l1_ratio=1.0 if alpha_H > 0 else 0.0)
    W = m.fit_transform(np.clip(A, 0, None))
    return {"W": W, "H": m.components_, "variant": "plain", "K": K}


def fit_geometry_regularised(A: np.ndarray, K: int, L: np.ndarray, lam: float = 0.1,
                             seed: int = SEED, n_iter: int = 600) -> dict:
    """NMF with `+ λ·tr(H L Hᵀ)` — nearby CSMs are encouraged to co-activate.

    The Laplacian acts on the **columns of H**, which are the CSMs, so the penalty says: if two
    CSMs are close in the frozen geometry, the programmes should load on them similarly. It is
    a smoothness reward, not a separation force: `tr(H L Hᵀ) = ½ Σ_ij w_ij ‖h_·i − h_·j‖²` has
    no term that grows when distant CSMs are given similar loadings, so the prior can never
    push anything apart or manufacture a cluster.

    Multiplicative updates with the Laplacian split into its positive and negative parts, which
    keeps every factor non-negative at every step without projection.
    """
    rng = np.random.default_rng(seed)
    n, m = A.shape
    Ap = np.clip(A, 0, None)
    base = fit_plain(Ap, K, seed)
    W, H = base["W"].copy() + EPS, base["H"].copy() + EPS
    Lp = np.clip(L, 0, None)          # positive part (degree)
    Ln = np.clip(-L, 0, None)         # negative part (adjacency)
    for _ in range(n_iter):
        W *= (Ap @ H.T) / (W @ H @ H.T + EPS)
        num = W.T @ Ap + lam * (H @ Ln)
        den = W.T @ W @ H + lam * (H @ Lp) + EPS
        H *= num / den
        W = np.clip(W, 0, None)
        H = np.clip(H, 0, None)
    return {"W": W, "H": H, "variant": "geometry_regularised", "K": K, "lambda": lam}


def fit(variant: str, A: np.ndarray, K: int, L: np.ndarray | None = None,
        lam: float = 0.1, seed: int = SEED) -> dict:
    if variant == "plain":
        return fit_plain(A, K, seed)
    if variant == "geometry_regularised":
        if L is None:
            raise ValueError("geometry_regularised needs the CSM Laplacian")
        return fit_geometry_regularised(A, K, L, lam, seed)
    raise ValueError(f"unknown variant {variant}")


def project(A_new: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Frozen Meta Component projection: NNLS of new CSM activations onto the frozen `H`.

    This is the inference path. No fitting, no randomness, and the answer for a spectrum
    depends only on that spectrum's CSM activations and the frozen dictionary.
    """
    from scipy.optimize import nnls
    A_new = np.atleast_2d(np.asarray(A_new, float))
    return np.vstack([nnls(H.T, np.clip(a, 0, None))[0] for a in A_new])


# ── the eleven model-selection metrics ───────────────────────────────────────
def reconstruction_error(A, W, H) -> float:
    return float(np.linalg.norm(A - W @ H) / (np.linalg.norm(A) + EPS))


def explained_variance(A, W, H) -> float:
    return float(max(0.0, 1.0 - ((A - W @ H) ** 2).sum() / ((A ** 2).sum() + EPS)))


def component_sparsity(H: np.ndarray) -> float:
    """Hoyer sparsity of the components, averaged. 1 = one CSM per programme, 0 = uniform."""
    n = H.shape[1]
    out = []
    for h in H:
        l1, l2 = np.abs(h).sum(), np.linalg.norm(h)
        out.append((np.sqrt(n) - l1 / (l2 + EPS)) / (np.sqrt(n) - 1 + EPS))
    return float(np.mean(out))


def activation_entropy(W: np.ndarray) -> float:
    """Mean normalised entropy of each spectrum's programme activation.

    Low entropy means a spectrum uses few programmes — which is what a *programme* should mean.
    Reported rather than optimised: entropy of exactly zero would be a hard assignment.
    """
    P = W / (W.sum(axis=1, keepdims=True) + EPS)
    K = W.shape[1]
    H = -(np.where(P > 0, P * np.log(P + EPS), 0.0)).sum(axis=1)
    return float(np.mean(H) / (np.log(max(K, 2)) + EPS))


def participation_ratio(M: np.ndarray) -> float:
    C = np.cov(np.asarray(M, float).T)
    ev = np.clip(np.linalg.eigvalsh(np.atleast_2d(C)), 0, None)
    return float((ev.sum() ** 2) / ((ev ** 2).sum() + EPS))


def effective_rank(M: np.ndarray) -> float:
    s = np.linalg.svd(np.asarray(M, float), compute_uv=False)
    p = s ** 2 / ((s ** 2).sum() + EPS)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def redundancy(H: np.ndarray) -> float:
    """Maximum pairwise cosine between components — duplicates, not shared chemistry."""
    if H.shape[0] < 2:
        return 0.0
    N = _rownorm(H)
    C = N @ N.T
    return float(C[np.triu_indices(H.shape[0], 1)].max())


def mutual_coherence(H: np.ndarray) -> float:
    """Mean pairwise cosine — how well-conditioned the meta dictionary is for projection."""
    if H.shape[0] < 2:
        return 0.0
    N = _rownorm(H)
    C = N @ N.T
    return float(C[np.triu_indices(H.shape[0], 1)].mean())


def bootstrap_stability(A: np.ndarray, K: int, variant: str, L=None, lam=0.1,
                        n_boot: int = 20, seed: int = SEED) -> dict:
    """Hungarian-matched component recovery under resampling of the spectra."""
    from scipy.optimize import linear_sum_assignment
    base = fit(variant, A, K, L, lam, seed)["H"]
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    scores, per = [], np.zeros(K)
    for _ in range(n_boot):
        idx = np.sort(rng.choice(n, int(0.85 * n), replace=False))
        H = fit(variant, A[idx], K, L, lam, seed)["H"]
        Nb, Nh = _rownorm(base), _rownorm(H)
        C = Nb @ Nh.T
        r, c = linear_sum_assignment(-C)
        scores.append(float(C[r, c].mean()))
        per[r] += C[r, c]
    return {"mean": float(np.mean(scores)), "min": float(np.min(scores)),
            "per_component": (per / max(n_boot, 1)).tolist()}


def consensus_stability(A: np.ndarray, K: int, variant: str, L=None, lam=0.1,
                        n_rep: int = 20, seed: int = SEED) -> float:
    """Co-assignment consensus: how consistently two spectra share a dominant programme.

    Scored as the dispersion of the consensus matrix — 1.0 when every pair is either always or
    never co-assigned, lower when the assignment is unstable. This is the standard consensus
    measure for NMF rank selection and it is orthogonal to component recovery.
    """
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    C, N = np.zeros((n, n)), np.zeros((n, n))
    for r in range(n_rep):
        idx = np.sort(rng.choice(n, int(0.85 * n), replace=False))
        W = fit(variant, A[idx], K, L, lam, seed + r)["W"]
        lab = W.argmax(axis=1)
        for a in range(len(idx)):
            for b in range(len(idx)):
                N[idx[a], idx[b]] += 1
                if lab[a] == lab[b]:
                    C[idx[a], idx[b]] += 1
    M = np.where(N > 0, C / np.maximum(N, 1), 0.0)
    iu = np.triu_indices(n, 1)
    return float((4 * (M[iu] - 0.5) ** 2).mean())


def interpretability(H: np.ndarray, csm_class: list[str], top: int = 5) -> float:
    """Purity of each component's top CSMs by chemistry class, against the base rate.

    **Evaluation only.** Class labels are not used to fit anything; they are revealed here to
    ask whether a programme corresponds to nameable chemistry. A component whose top five CSMs
    span five classes is a programme nobody can name.
    """
    cls = np.asarray(csm_class)
    _, counts = np.unique(cls, return_counts=True)
    base = float(((counts / counts.sum()) ** 2).sum())
    out = []
    for h in H:
        nb = np.argsort(-h)[:top]
        sub = cls[nb]
        out.append(max(np.sum(sub == c) for c in set(sub)) / len(sub))
    return float(np.mean(out) / (base + EPS))
