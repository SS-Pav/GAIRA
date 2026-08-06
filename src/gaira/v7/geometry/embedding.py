"""GAIRA V7 — Phase 02.5: linear and nonlinear embeddings, and how much to believe them.

PCA, UMAP and diffusion maps are estimators of structure, not the structure. Each is therefore
paired here with the measurement that says whether its output means anything: explained variance
and resampling stability for PCA, a neighbourhood-stability sweep for UMAP, eigenvalue decay for
diffusion maps, and trustworthiness/continuity for all of them.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding, trustworthiness

EPS = 1e-12
SEED = 0


# ── linear ───────────────────────────────────────────────────────────────────
def fit_pca(V: np.ndarray, n_components: int = 10) -> dict:
    """Deterministic PCA with a full SVD — no randomised solver, so the result is exact."""
    V = np.asarray(V, float)
    k = min(n_components, min(V.shape) - 1)
    p = PCA(n_components=k, svd_solver="full", random_state=SEED)
    S = p.fit_transform(V)
    # sign convention: the largest-magnitude loading of each PC is positive, so scores are
    # reproducible across runs and machines rather than flipping arbitrarily
    for j in range(k):
        if p.components_[j][np.argmax(np.abs(p.components_[j]))] < 0:
            p.components_[j] *= -1
            S[:, j] *= -1
    return {"scores": S, "loadings": p.components_,
            "explained_variance_ratio": p.explained_variance_ratio_,
            "cumulative": np.cumsum(p.explained_variance_ratio_), "mean": p.mean_}


def pca_stability(V: np.ndarray, n_components: int = 6, n_boot: int = 40,
                  seed: int = SEED) -> np.ndarray:
    """Per-PC |cosine| between the full-data loading and bootstrap-resampled loadings.

    A PC whose loading vector is not reproducible under resampling is a direction of this
    particular sample, not of motif space.
    """
    V = np.asarray(V, float)
    base = fit_pca(V, n_components)["loadings"]
    rng = np.random.default_rng(seed)
    out = np.zeros((n_boot, base.shape[0]))
    for b in range(n_boot):
        idx = rng.choice(V.shape[0], V.shape[0], replace=True)
        L = fit_pca(V[idx], n_components)["loadings"]
        k = min(L.shape[0], base.shape[0])
        out[b, :k] = np.abs((base[:k] * L[:k]).sum(axis=1) /
                            (np.linalg.norm(base[:k], axis=1) * np.linalg.norm(L[:k], axis=1)
                             + EPS))
    return out.mean(axis=0)


def band_drivers(loading: np.ndarray, grid: np.ndarray, top: int = 6) -> list[dict]:
    """The Raman shifts a PC actually loads on, signed, strongest first."""
    idx = np.argsort(np.abs(loading))[::-1][:top]
    return [{"cm1": float(grid[i]), "loading": float(loading[i]),
             "direction": "+" if loading[i] > 0 else "−"} for i in sorted(idx, key=lambda i: -abs(loading[i]))]


# ── nonlinear ────────────────────────────────────────────────────────────────
def fit_umap(Dm: np.ndarray, n_neighbors: int = 8, min_dist: float = 0.3,
             seed: int = SEED) -> np.ndarray:
    import umap
    r = umap.UMAP(n_neighbors=min(n_neighbors, Dm.shape[0] - 1), min_dist=min_dist,
                  metric="precomputed", random_state=seed, n_components=2,
                  init="spectral", transform_seed=seed)
    return r.fit_transform(Dm)


def umap_stability_sweep(Dm: np.ndarray, neighbours=(5, 8, 12, 20), dists=(0.05, 0.3, 0.7),
                         seeds=(0, 1, 2), k: int = 5) -> list[dict]:
    """Sweep the two parameters that change UMAP's answer, and measure neighbour agreement.

    A single UMAP layout is not evidence. What is evidence is whether a motif keeps the same
    neighbours as the parameters and seed move — reported as the mean Jaccard of k-NN sets
    against the layout's own high-dimensional neighbours, and across seeds.
    """
    true_nb = [set(np.argsort(Dm[i])[1:k + 1]) for i in range(Dm.shape[0])]
    rows = []
    for nn in neighbours:
        for md in dists:
            embs = [fit_umap(Dm, nn, md, s) for s in seeds]
            jac_hd, jac_seed = [], []
            for E in embs:
                Dl = _pairwise(E)
                for i in range(Dm.shape[0]):
                    nb = set(np.argsort(Dl[i])[1:k + 1])
                    jac_hd.append(len(nb & true_nb[i]) / len(nb | true_nb[i]))
            for a in range(len(embs)):
                for b in range(a + 1, len(embs)):
                    Da, Db = _pairwise(embs[a]), _pairwise(embs[b])
                    for i in range(Dm.shape[0]):
                        na = set(np.argsort(Da[i])[1:k + 1])
                        nbs = set(np.argsort(Db[i])[1:k + 1])
                        jac_seed.append(len(na & nbs) / len(na | nbs))
            rows.append({"n_neighbors": nn, "min_dist": md,
                         "knn_jaccard_vs_highdim": float(np.mean(jac_hd)),
                         "knn_jaccard_across_seeds": float(np.mean(jac_seed)),
                         "trustworthiness": float(trustworthiness(
                             Dm, embs[0], n_neighbors=k, metric="precomputed"))})
    return rows


def fit_diffusion_map(Dm: np.ndarray, n_components: int = 5, epsilon: float | None = None,
                      alpha: float = 1.0) -> dict:
    """Diffusion map with the Coifman alpha-normalisation.

    Chosen over t-SNE/UMAP as the *quantitative* nonlinear view because its coordinates have a
    meaning — a diffusion coordinate is a slow mode of a random walk on the similarity graph,
    so a smooth gradient along one is a real continuum rather than a layout artefact. The
    eigenvalue spectrum also says how many coordinates are worth reading.
    """
    Dm = np.asarray(Dm, float)
    if epsilon is None:
        off = Dm[~np.eye(Dm.shape[0], dtype=bool)]
        epsilon = float(np.median(off)) ** 2
    K = np.exp(-(Dm ** 2) / (epsilon + EPS))
    q = K.sum(axis=1)
    K = K / (np.outer(q, q) ** alpha + EPS)
    d = K.sum(axis=1)
    P = K / (d[:, None] + EPS)
    w, v = np.linalg.eig(P)
    order = np.argsort(-w.real)
    w, v = w.real[order], v.real[:, order]
    v = v / (np.linalg.norm(v, axis=0, keepdims=True) + EPS)
    return {"eigenvalues": w[:n_components + 1],
            "coordinates": v[:, 1:n_components + 1] * w[1:n_components + 1],
            "epsilon": epsilon,
            "spectral_gap": float(w[1] - w[2]) if w.size > 2 else float("nan")}


def fit_spectral_embedding(Dm: np.ndarray, n_components: int = 3, seed: int = SEED) -> np.ndarray:
    A = np.exp(-(Dm ** 2) / (np.median(Dm[Dm > 0]) ** 2 + EPS))
    return SpectralEmbedding(n_components=n_components, affinity="precomputed",
                             random_state=seed).fit_transform(A)


# ── embedding quality ────────────────────────────────────────────────────────
def _pairwise(E):
    from scipy.spatial.distance import pdist, squareform
    return squareform(pdist(np.asarray(E, float)))


def continuity(Dm: np.ndarray, E: np.ndarray, k: int = 5) -> float:
    """The dual of trustworthiness: are true neighbours kept close in the embedding?"""
    n = Dm.shape[0]
    Dl = _pairwise(E)
    rank_l = np.argsort(np.argsort(Dl, axis=1), axis=1)
    total = 0.0
    for i in range(n):
        true_nb = np.argsort(Dm[i])[1:k + 1]
        for j in true_nb:
            r = rank_l[i, j]
            if r > k:
                total += r - k
    return float(1 - 2 * total / (n * k * (2 * n - 3 * k - 1) + EPS))


def knn_preservation(Dm: np.ndarray, E: np.ndarray, k: int = 5) -> float:
    Dl = _pairwise(E)
    n = Dm.shape[0]
    return float(np.mean([len(set(np.argsort(Dm[i])[1:k + 1]) & set(np.argsort(Dl[i])[1:k + 1])) / k
                          for i in range(n)]))


def procrustes_stability(embed_fn, n_rep: int = 5) -> float:
    """Mean Procrustes disparity between repeated embeddings — 0 means identical up to
    rotation, scale and reflection."""
    from scipy.spatial import procrustes
    embs = [embed_fn(s) for s in range(n_rep)]
    ds = []
    for a in range(len(embs)):
        for b in range(a + 1, len(embs)):
            ds.append(procrustes(embs[a], embs[b])[2])
    return float(np.mean(ds))
