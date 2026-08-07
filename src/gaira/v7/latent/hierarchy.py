"""GAIRA V7 — Phase 06.5: is Raman space tree-like, graph-like, continuous, or modular?

Phase 02.5 answered this for the 49 *motifs* and found a continuum with one bipartition. This
module asks the same question one level up, for the 154 *molecules* in motif space, where the
answer need not be the same.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def intrinsic_dimension(M: np.ndarray, k: int = 10) -> dict:
    """Two estimators that fail in different ways, reported together.

    Levina–Bickel maximum likelihood on k-NN distances, and a correlation-dimension slope. If
    they disagree substantially the estimate is not trustworthy, and saying so is more useful
    than reporting whichever is prettier.
    """
    from .clustering import cosine_distance
    D = cosine_distance(M)
    np.fill_diagonal(D, np.inf)
    nn = np.sort(D, axis=1)[:, :k]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.log(nn[:, -1][:, None] / np.clip(nn[:, :-1], EPS, None))
        mle = (k - 2) / np.clip(ratios.sum(axis=1), EPS, None)
    mle = mle[np.isfinite(mle)]
    iu = np.triu_indices(len(M), 1)
    d = D[iu]
    rs = np.quantile(d, np.linspace(0.05, 0.5, 12))
    counts = np.array([float((d < r).mean()) for r in rs])
    ok = counts > 0
    slope = np.polyfit(np.log(rs[ok]), np.log(np.clip(counts[ok], EPS, None)), 1)[0] \
        if ok.sum() > 2 else np.nan
    return {"levina_bickel_mle": float(np.median(mle)) if len(mle) else np.nan,
            "correlation_dimension": float(slope),
            "ambient_dimension": int(M.shape[1]),
            "estimators_agree": bool(len(mle) and np.isfinite(slope) and
                                     abs(np.median(mle) - slope) < 0.5 *
                                     max(np.median(mle), slope))}


def gap_statistic(M: np.ndarray) -> dict:
    """Is the pairwise-distance distribution bimodal? A density gap is what 'islands' means."""
    from .clustering import cosine_distance
    d = cosine_distance(M)[np.triu_indices(len(M), 1)]
    hist, edges = np.histogram(d, bins=60, density=True)
    lo, hi = int(0.15 * len(hist)), int(0.85 * len(hist))
    if hi <= lo + 2:
        return {"valley_depth": np.nan, "bimodal": False}
    peak_l = hist[:lo].max() if lo else hist[0]
    peak_r = hist[hi:].max() if hi < len(hist) else hist[-1]
    valley = hist[lo:hi].min()
    depth = float(1.0 - valley / (min(peak_l, peak_r) + EPS))
    return {"valley_depth": depth, "bimodal": bool(depth > 0.30),
            "median_distance": float(np.median(d))}


def modularity_vs_null(M: np.ndarray, k: int = 5, n_null: int = 200, seed: int = 0) -> dict:
    """Greedy-modularity communities on the k-NN graph, against a degree-preserving null."""
    import networkx as nx
    from networkx.algorithms.community import greedy_modularity_communities, modularity
    from .clustering import cosine_distance
    D = cosine_distance(M)
    np.fill_diagonal(D, np.inf)
    G = nx.Graph()
    G.add_nodes_from(range(len(M)))
    for i in range(len(M)):
        for j in np.argsort(D[i])[:k]:
            G.add_edge(i, int(j), weight=float(np.exp(-D[i, j])))
    comms = list(greedy_modularity_communities(G, weight="weight"))
    Q = float(modularity(G, comms, weight="weight"))
    rng = np.random.default_rng(seed)
    null = []
    deg = [d for _, d in G.degree()]
    for _ in range(n_null):
        try:
            H = nx.configuration_model(deg, seed=int(rng.integers(1 << 30)))
            H = nx.Graph(H)
            H.remove_edges_from(nx.selfloop_edges(H))
            null.append(float(modularity(H, list(greedy_modularity_communities(H)))))
        except Exception:                                          # pragma: no cover
            continue
    return {"modularity": Q, "n_communities": len(comms),
            "null_mean": float(np.mean(null)) if null else np.nan,
            "null_sd": float(np.std(null)) if null else np.nan,
            "z_score": float((Q - np.mean(null)) / (np.std(null) + EPS)) if null else np.nan,
            "community_sizes": sorted((len(c) for c in comms), reverse=True),
            "p_value": float((sum(q >= Q for q in null) + 1) / (len(null) + 1)) if null
                       else np.nan}


def cophenetic_fit(M: np.ndarray, method: str = "average") -> dict:
    """How tree-like is the space? Cophenetic correlation of the dendrogram to the distances."""
    from scipy.cluster.hierarchy import cophenet, linkage
    from scipy.spatial.distance import squareform
    from .clustering import cosine_distance
    D = cosine_distance(M)
    d = squareform(D, checks=False)
    out = {}
    for m in ("average", "complete", "ward", "single"):
        try:
            Z = linkage(d, method=m) if m != "ward" else linkage(
                np.asarray(M, float) / (np.linalg.norm(M, axis=1, keepdims=True) + EPS),
                method="ward")
            c, _ = cophenet(Z, d)
            out[m] = float(c)
        except Exception:                                          # pragma: no cover
            out[m] = np.nan
    best = max((v, k) for k, v in out.items() if np.isfinite(v))
    return {"cophenetic_correlation": out, "best_linkage": best[1],
            "best_correlation": best[0],
            "tree_like": bool(best[0] > 0.75)}


def branch_points(M: np.ndarray, method: str = "average", max_k: int = 12) -> "pd.DataFrame":
    """Where the dendrogram splits, and how much each split buys."""
    import pandas as pd
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from .clustering import cosine_distance, internal_indices
    D = cosine_distance(M)
    Z = linkage(squareform(D, checks=False), method=method)
    rows = []
    prev = None
    for K in range(2, max_k + 1):
        lab = fcluster(Z, K, criterion="maxclust") - 1
        iv = internal_indices(M, lab)
        sizes = sorted(np.bincount(lab).tolist(), reverse=True)
        rows.append({"K": K, "silhouette": iv["silhouette"],
                     "largest_cluster": sizes[0], "smallest_cluster": sizes[-1],
                     "split_gain": (iv["silhouette"] - prev) if prev is not None else np.nan})
        prev = iv["silhouette"]
    return pd.DataFrame(rows)


def continuity(M: np.ndarray, k: int = 5) -> dict:
    """Bridge density: what share of molecules sit between neighbourhoods rather than inside one.

    Measured label-free as the fraction whose k nearest neighbours span more than one
    greedy-modularity community — the graph-level analogue of the Phase 02.5 bridge motifs.
    """
    import networkx as nx
    from networkx.algorithms.community import greedy_modularity_communities
    from .clustering import cosine_distance
    D = cosine_distance(M)
    np.fill_diagonal(D, np.inf)
    G = nx.Graph()
    G.add_nodes_from(range(len(M)))
    for i in range(len(M)):
        for j in np.argsort(D[i])[:k]:
            G.add_edge(i, int(j))
    comms = list(greedy_modularity_communities(G))
    of = {n: c for c, s in enumerate(comms) for n in s}
    nn = np.argsort(D, axis=1)[:, :k]
    spans = np.array([len({of[int(j)] for j in nn[i]} | {of[i]}) for i in range(len(M))])
    return {"mean_communities_in_neighbourhood": float(spans.mean()),
            "fraction_bridging": float(np.mean(spans > 1)),
            "n_communities": len(comms)}
