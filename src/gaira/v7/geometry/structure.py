"""GAIRA V7 — Phase 02.5: graph, hierarchy, and the discrete-versus-continuous question.

Phase 02 already established that this space has no threshold at which a partition is invariant.
That is a statement about *clusters*; it says nothing about neighbourhoods, gradients or
branches. This module measures those, and answers the question Phase 02 could not: is motif
space made of islands, of continua, or of both.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import SpectralClustering
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)

EPS = 1e-12
LINKAGES = ("average", "complete", "ward", "single")
K_RANGE = range(2, 16)


# ── hierarchical and spectral clustering ─────────────────────────────────────
def hierarchical(Dm: np.ndarray, method: str = "average"):
    return linkage(squareform(np.asarray(Dm, float), checks=False), method=method)


def cluster_sweep(Dm: np.ndarray, V: np.ndarray, ks=K_RANGE) -> list[dict]:
    """Cluster quality across K and linkage, on every index that disagrees with the others.

    Silhouette, Davies-Bouldin and Calinski-Harabasz do not agree on this kind of data, and
    that disagreement is informative: a K that wins on all three is a real cluster count, and a
    K that wins on one is a preference.
    """
    rows = []
    for method in LINKAGES:
        Z = hierarchical(Dm, method)
        for k in ks:
            lab = fcluster(Z, t=k, criterion="maxclust")
            if len(set(lab)) < 2:
                continue
            rows.append({"method": method, "k": int(k), "n_clusters": int(len(set(lab))),
                         "silhouette": float(silhouette_score(Dm, lab, metric="precomputed")),
                         "davies_bouldin": float(davies_bouldin_score(V, lab)),
                         "calinski_harabasz": float(calinski_harabasz_score(V, lab)),
                         "largest_cluster": int(np.bincount(lab).max()),
                         "n_singleton_clusters": int((np.bincount(lab)[1:] == 1).sum())})
    for k in ks:
        A = np.exp(-(Dm ** 2) / (np.median(Dm[Dm > 0]) ** 2 + EPS))
        lab = SpectralClustering(n_clusters=k, affinity="precomputed",
                                 random_state=0, n_init=10).fit_predict(A)
        if len(set(lab)) < 2:
            continue
        rows.append({"method": "spectral", "k": int(k), "n_clusters": int(len(set(lab))),
                     "silhouette": float(silhouette_score(Dm, lab, metric="precomputed")),
                     "davies_bouldin": float(davies_bouldin_score(V, lab)),
                     "calinski_harabasz": float(calinski_harabasz_score(V, lab)),
                     "largest_cluster": int(np.bincount(lab).max()),
                     "n_singleton_clusters": int((np.bincount(lab) == 1).sum())})
    return rows


def cluster_bootstrap_ari(Dm: np.ndarray, k: int, method: str = "average",
                          n_boot: int = 50, seed: int = 0) -> dict:
    """ARI and variation of information between the full clustering and resampled ones."""
    n = Dm.shape[0]
    base = fcluster(hierarchical(Dm, method), t=k, criterion="maxclust")
    rng = np.random.default_rng(seed)
    aris, vis = [], []
    for _ in range(n_boot):
        keep = np.sort(rng.choice(n, int(0.85 * n), replace=False))
        lab = fcluster(hierarchical(Dm[np.ix_(keep, keep)], method), t=k, criterion="maxclust")
        aris.append(adjusted_rand_score(base[keep], lab))
        vis.append(_variation_of_information(base[keep], lab))
    return {"mean_ari": float(np.mean(aris)), "min_ari": float(np.min(aris)),
            "mean_vi": float(np.mean(vis))}


def _variation_of_information(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    n = a.size
    hx = hy = ixy = 0.0
    for ca in set(a):
        pa = (a == ca).mean()
        hx -= pa * np.log(pa + EPS)
        for cb in set(b):
            pab = ((a == ca) & (b == cb)).mean()
            if pab > 0:
                ixy += pab * np.log(pab / (pa * (b == cb).mean() + EPS) + EPS)
    for cb in set(b):
        pb = (b == cb).mean()
        hy -= pb * np.log(pb + EPS)
    return float(hx + hy - 2 * ixy)


# ── graph structure ──────────────────────────────────────────────────────────
def knn_graph(Dm: np.ndarray, ids: list[str], k: int = 5, mutual: bool = False) -> nx.Graph:
    """k-NN graph. `mutual=True` keeps only reciprocated edges, which removes the hub effect
    where one popular motif attaches to everything."""
    n = Dm.shape[0]
    G = nx.Graph()
    G.add_nodes_from(ids)
    nb = [set(np.argsort(Dm[i])[1:k + 1]) for i in range(n)]
    for i in range(n):
        for j in nb[i]:
            if mutual and i not in nb[j]:
                continue
            G.add_edge(ids[i], ids[j], weight=float(1.0 / (Dm[i, j] + EPS)),
                       distance=float(Dm[i, j]))
    return G


def graph_roles(G: nx.Graph, Dm: np.ndarray, ids: list[str],
                labels: list[str] | None = None) -> "pd.DataFrame":
    """Degree, betweenness, clustering and local density for every motif.

    High betweenness with low local clustering is the signature of a **bridge** — a motif that
    sits on paths between neighbourhoods without belonging to one. Those motifs matter for
    Phase 03: they are exactly the ones a hard theme assignment would misplace.
    """
    import pandas as pd
    bt = nx.betweenness_centrality(G, weight="distance", normalized=True)
    cl = nx.clustering(G, weight="weight")
    n = len(ids)
    rows = []
    for i, m in enumerate(ids):
        d = np.sort(Dm[i])[1:6]
        rows.append({"motif": m, "degree": G.degree(m),
                     "betweenness": float(bt.get(m, 0.0)),
                     "clustering": float(cl.get(m, 0.0)),
                     "mean_knn_distance": float(d.mean()),
                     "nn1_distance": float(d[0]),
                     "isolation": float(d.mean() / (np.median(Dm[Dm > 0]) + EPS))})
    df = pd.DataFrame(rows)
    df["is_bridge"] = (df.betweenness > df.betweenness.quantile(0.85)) & \
                      (df.clustering < df.clustering.median())
    df["is_hub"] = df.degree > df.degree.quantile(0.90)
    df["is_isolated"] = df.isolation > df.isolation.quantile(0.90)
    if labels is not None:
        df["chemical_class"] = list(labels)      # evaluation only, attached after the fact
    return df


def minimum_spanning_tree(Dm: np.ndarray, ids: list[str]) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(ids)
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(ids[i], ids[j], weight=float(Dm[i, j]))
    return nx.minimum_spanning_tree(G, weight="weight")


def modularity_vs_null(G: nx.Graph, W: np.ndarray, n_null: int = 50, seed: int = 0) -> dict:
    """Observed modularity against a degree-preserving rewiring null."""
    from .nulls import degree_preserving_graph_null
    comms = nx.community.louvain_communities(G, weight="weight", seed=0)
    obs = nx.community.modularity(G, comms, weight="weight")
    ids = sorted(G.nodes)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_null):
        Wn = degree_preserving_graph_null(W, rng)
        Gn = nx.Graph()
        Gn.add_nodes_from(ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if Wn[i, j] > 0:
                    Gn.add_edge(ids[i], ids[j], weight=float(Wn[i, j]))
        if Gn.number_of_edges() == 0:
            continue
        cn = nx.community.louvain_communities(Gn, weight="weight", seed=0)
        draws.append(nx.community.modularity(Gn, cn, weight="weight"))
    draws = np.array(draws) if draws else np.array([0.0])
    return {"observed_modularity": float(obs), "null_mean": float(draws.mean()),
            "null_sd": float(draws.std()), "n_communities": len(comms),
            "z": float((obs - draws.mean()) / (draws.std() + EPS)),
            "p_empirical": float((draws >= obs).mean())}


# ── discrete vs continuous ───────────────────────────────────────────────────
def local_intrinsic_dimension(Dm: np.ndarray, k: int = 10) -> np.ndarray:
    """Maximum-likelihood (Levina-Bickel) local intrinsic dimension per motif.

    A dimension near 1 along a region means motifs there lie on a curve — a gradient. A high
    dimension means they fill space, which is what a set of unrelated islands looks like
    locally.
    """
    n = Dm.shape[0]
    out = np.zeros(n)
    for i in range(n):
        d = np.sort(Dm[i])[1:k + 1]
        d = d[d > 0]
        if d.size < 3:
            out[i] = np.nan
            continue
        out[i] = float((d.size - 1) / np.sum(np.log(d[-1] / d[:-1] + EPS)))
    return out


def density_gap_statistic(Dm: np.ndarray, k: int = 5) -> dict:
    """Dip-like test for multimodality of the k-NN distance distribution.

    Discrete islands produce a bimodal distance distribution — short within-island distances,
    long between-island ones, with a valley between. A continuum produces a unimodal one.
    """
    from scipy.stats import gaussian_kde
    off = Dm[~np.eye(Dm.shape[0], dtype=bool)]
    kde = gaussian_kde(off)
    xs = np.linspace(off.min(), off.max(), 400)
    ys = kde(xs)
    peaks = [i for i in range(1, len(ys) - 1) if ys[i] > ys[i - 1] and ys[i] > ys[i + 1]]
    valleys = [i for i in range(1, len(ys) - 1) if ys[i] < ys[i - 1] and ys[i] < ys[i + 1]]
    depth = 0.0
    if len(peaks) >= 2 and valleys:
        v = min(valleys, key=lambda i: ys[i])
        depth = float(1 - ys[v] / min(ys[peaks[0]], ys[peaks[-1]]))
    return {"n_modes": len(peaks), "valley_depth": depth,
            "bimodal": bool(len(peaks) >= 2 and depth > 0.15)}


def graph_conductance(Dm: np.ndarray, groups: list[list[int]]) -> list[float]:
    """Conductance of each group in the similarity graph — low means a genuine island."""
    S = 1.0 / (Dm + EPS)
    np.fill_diagonal(S, 0.0)
    out = []
    for g in groups:
        g = np.asarray(g)
        mask = np.zeros(Dm.shape[0], bool)
        mask[g] = True
        cut = S[np.ix_(mask, ~mask)].sum()
        vol = S[mask].sum()
        out.append(float(cut / (min(vol, S.sum() - vol) + EPS)))
    return out


def classify_region(conductance: float, mean_lid: float, bimodal: bool,
                    stability: float) -> str:
    """Assign a geometry type from the measurements, on pre-stated thresholds."""
    if stability < 0.4:
        return "unresolved"
    if conductance < 0.35 and bimodal:
        return "discrete"
    if mean_lid < 1.6:
        return "continuous"
    if conductance < 0.55:
        return "branching"
    return "overlapping"
