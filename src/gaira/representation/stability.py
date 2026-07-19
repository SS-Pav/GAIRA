"""Consensus (bootstrap) clustering stability (Phase 2 §11).

Resamples analytes with replacement, reclusters, and accumulates a co-assignment
(consensus) matrix over centroids. Stable structure = pairs that cluster together
across resamples. Bootstrapping is by analyte (§13), never by spectrum.
"""
from __future__ import annotations
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


def consensus_clustering(X, analytes, k, n_boot=200, metric="cosine", method="average", seed=0):
    """X: centroid matrix (row per item); analytes: per-row analyte label (bootstrap unit).
    Returns mean consensus (co-cluster probability) among co-sampled pairs + summary."""
    rng = np.random.default_rng(seed)
    analytes = np.asarray(analytes)
    uniq = np.unique(analytes)
    n = X.shape[0]
    co = np.zeros((n, n)); cnt = np.zeros((n, n))
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(analytes == g)[0] for g in samp])
        idx = np.unique(idx)
        if len(idx) < k + 1:
            continue
        D = pdist(X[idx], metric=metric)
        Z = linkage(D, method=method)
        cl = fcluster(Z, t=k, criterion="maxclust")
        for a in range(len(idx)):
            for b in range(a, len(idx)):
                i, j = idx[a], idx[b]
                cnt[i, j] += 1; cnt[j, i] += 1
                if cl[a] == cl[b]:
                    co[i, j] += 1; co[j, i] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        consensus = np.where(cnt > 0, co / cnt, np.nan)
    off = ~np.eye(n, dtype=bool)
    vals = consensus[off & (cnt > 0)]
    # dispersion of consensus: 0 = perfectly stable (all 0/1), 1 = maximally ambiguous
    amb = float(np.nanmean(4 * vals * (1 - vals))) if vals.size else float("nan")
    return {"mean_consensus": float(np.nanmean(vals)) if vals.size else None,
            "consensus_ambiguity": amb, "k": int(k), "n_boot": n_boot,
            "consensus_matrix": consensus}
