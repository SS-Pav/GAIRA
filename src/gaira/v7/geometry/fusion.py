"""GAIRA V7 — Phase 02.5: multi-view integration and the Pareto choice between candidates.

Five ways of combining the seven views, compared on seven pre-declared criteria. The winner is
not the one that looks cleanest — a fused geometry can invent structure as easily as it can
reveal it, so "avoids creating false clusters" is scored explicitly against the null.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12
FUSION_METHODS = ("weighted_similarity", "similarity_network_fusion",
                  "multiple_kernel_embedding", "concatenated_features", "graph_consensus")

# Pre-declared Pareto criteria, direction, and weight. Fixed before any fusion was run.
CRITERIA = {
    "neighbourhood_stability": (0.22, +1),
    "null_separation": (0.20, +1),
    "spectroscopic_coherence": (0.16, +1),
    "source_robustness": (0.16, +1),
    "neighbourhood_preservation": (0.12, +1),
    "interpretability": (0.08, +1),
    "simplicity": (0.06, +1),
}


def _sim(Dm):
    Dm = np.asarray(Dm, float)
    S = 1.0 - Dm / (Dm.max() + EPS)
    np.fill_diagonal(S, 1.0)
    return S


def _dist(S):
    Dm = 1.0 - S / (S.max() + EPS)
    np.fill_diagonal(Dm, 0.0)
    return (Dm + Dm.T) / 2


# ── A. weighted similarity fusion ────────────────────────────────────────────
def weighted_similarity(Ds: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    tot = sum(weights.values())
    S = sum((weights[k] / tot) * _sim(Ds[k]) for k in weights)
    return _dist(S)


# ── B. similarity network fusion ─────────────────────────────────────────────
def similarity_network_fusion(Ds: dict[str, np.ndarray], k: int = 8,
                              iters: int = 20) -> np.ndarray:
    """Wang's SNF: each view's similarity is diffused through the others' local structures.

    The point is that a relationship supported by one view alone decays, while one supported by
    several is reinforced — the same logic as the Phase 02 geometric-mean edge weight, applied
    to whole matrices instead of single edges.
    """
    views = list(Ds)
    P = {}
    S = {}
    for v in views:
        W = _sim(Ds[v])
        np.fill_diagonal(W, 0.0)
        P[v] = W / (W.sum(axis=1, keepdims=True) + EPS)
        n = W.shape[0]
        Sl = np.zeros_like(W)
        for i in range(n):
            nb = np.argsort(-W[i])[:k]
            Sl[i, nb] = W[i, nb] / (W[i, nb].sum() + EPS)
        S[v] = Sl
    for _ in range(iters):
        newP = {}
        for v in views:
            others = [P[u] for u in views if u != v]
            newP[v] = S[v] @ (sum(others) / max(len(others), 1)) @ S[v].T
            newP[v] = newP[v] / (newP[v].sum(axis=1, keepdims=True) + EPS)
        P = newP
    fused = sum(P.values()) / len(views)
    fused = (fused + fused.T) / 2
    return _dist(fused)


# ── C. multiple-kernel spectral embedding ────────────────────────────────────
def multiple_kernel_embedding(Ds: dict[str, np.ndarray], n_components: int = 6) -> np.ndarray:
    """Average the view kernels, embed, and take distances in the embedding."""
    from sklearn.manifold import SpectralEmbedding
    K = np.zeros_like(next(iter(Ds.values())))
    for Dm in Ds.values():
        sigma = np.median(Dm[Dm > 0]) + EPS
        K += np.exp(-(Dm ** 2) / (sigma ** 2))
    K /= len(Ds)
    E = SpectralEmbedding(n_components=min(n_components, K.shape[0] - 2),
                          affinity="precomputed", random_state=0).fit_transform(K)
    from scipy.spatial.distance import pdist, squareform
    Dm = squareform(pdist(E))
    return Dm / (Dm.max() + EPS)


# ── D. concatenated standardised features ────────────────────────────────────
def concatenated_features(views: dict[str, np.ndarray]) -> np.ndarray:
    from scipy.spatial.distance import pdist, squareform
    blocks = []
    for V in views.values():
        V = np.asarray(V, float)
        V = (V - V.mean(0)) / (V.std(0) + EPS)
        # scale each block by 1/sqrt(dim) so a 676-column view does not outvote a 20-column one
        blocks.append(V / np.sqrt(V.shape[1]))
    Dm = squareform(pdist(np.hstack(blocks)))
    return Dm / (Dm.max() + EPS)


# ── E. graph consensus across views ──────────────────────────────────────────
def graph_consensus(Ds: dict[str, np.ndarray], k: int = 5) -> np.ndarray:
    """Fraction of views in which two motifs are mutual k-nearest neighbours.

    Rank-based rather than value-based, so views on incomparable scales can vote without being
    forced onto a common scale first. This is the multi-view analogue of Phase 02's
    threshold-consensus estimator.
    """
    n = next(iter(Ds.values())).shape[0]
    C = np.zeros((n, n))
    for Dm in Ds.values():
        nb = [set(np.argsort(Dm[i])[1:k + 1]) for i in range(n)]
        for i in range(n):
            for j in nb[i]:
                if i in nb[j]:
                    C[i, j] += 1
                    C[j, i] += 1
    C /= (2 * len(Ds))
    return _dist(C + np.eye(n))


def score_geometry(Dm: np.ndarray, labels: list[str], sources: list[str],
                   null_Ds: list[np.ndarray], boot_fn, k: int = 5,
                   n_features: int = 1) -> dict:
    """The seven pre-declared criteria for one candidate geometry."""
    from . import metrics as MET
    n = Dm.shape[0]
    lab = np.asarray(labels)
    src = np.asarray(sources)

    coherence = MET.knn_label_coherence(Dm, labels, k)
    src_pred = float(np.mean([
        (src[np.argsort(Dm[i])[1:k + 1]] == src[i]).mean() for i in range(n)]))
    return {
        "neighbourhood_stability": float(boot_fn(Dm)),
        "null_separation": float(MET.null_separation(Dm, null_Ds)),
        "spectroscopic_coherence": coherence,
        # robustness is 1 when neighbours are no more source-alike than chance
        "source_robustness": float(1.0 - max(0.0, src_pred - _chance(src))),
        "neighbourhood_preservation": float(np.mean([
            len(set(np.argsort(Dm[i])[1:k + 1])) / k for i in range(n)])),
        "interpretability": float(1.0),      # set per candidate by the caller
        "simplicity": float(1.0 / (1.0 + np.log10(max(n_features, 1)))),
        "knn_source_predictability": src_pred,
    }


def _chance(src: np.ndarray) -> float:
    _, c = np.unique(src, return_counts=True)
    p = c / c.sum()
    return float((p ** 2).sum())


def pareto_select(rows: list[dict]) -> tuple[str, np.ndarray]:
    """Min-max normalise each criterion across candidates, apply weights, rank."""
    out = np.zeros(len(rows))
    for crit, (w, direction) in CRITERIA.items():
        v = np.array([r[crit] for r in rows], float)
        span = v.max() - v.min()
        z = np.full_like(v, 0.5) if span < EPS else (v - v.min()) / span
        out += w * (z if direction > 0 else 1 - z)
    return rows[int(np.argmax(out))]["method"], out
