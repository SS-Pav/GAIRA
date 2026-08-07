"""GAIRA V7 — Phase 06.5: unsupervised clustering of the frozen CSM manifold.

**No chemistry label enters any function in this module.** Labels are used only afterwards, in
`composition.py`, to interpret what emerged. The distinction is the whole point of the phase: a
geometry optimised to match an ontology cannot then be evidence for that ontology.

The unit of analysis is the **canonical molecule**, not the spectrum. Replicates of one molecule
are near-duplicates in CSM space, and clustering them would manufacture stability — three
spectra of glucose always co-assign, which says nothing about whether glucose belongs with
fructose. This is principle P-11 applied to geometry.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12
K_GRID = (2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24, 30)
ALGORITHMS = ("ward", "average", "complete", "spectral", "hdbscan", "affinity_propagation")
# HDBSCAN and affinity propagation choose their own K; they are swept over their own control
# parameter and reported at whatever K they produce.
FIXED_K_ALGORITHMS = ("ward", "average", "complete", "spectral")


def unit(M):
    return np.asarray(M, float) / (np.linalg.norm(M, axis=1, keepdims=True) + EPS)


def cosine_distance(M):
    """Pairwise cosine distance on non-negative activations, clipped to [0, 2].

    The diagonal is set to exactly zero. Floating-point residue of ~1e-11 is enough for
    scikit-learn to reject a precomputed distance matrix, and a bare `except` around
    `silhouette_score` turned that into NaN across an entire 56-row sweep before it was caught.
    """
    N = unit(M)
    D = np.clip(1.0 - N @ N.T, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    return 0.5 * (D + D.T)


def fit(algorithm: str, M: np.ndarray, K: int | None = None, seed: int = 0,
        param: float | None = None) -> np.ndarray:
    """Return integer labels. Deterministic for every algorithm used here.

    `-1` means *unassigned* and only HDBSCAN produces it. It is preserved rather than forced
    into a cluster: a density-based method saying "this molecule is not in any dense region" is
    information, and Phase 02.5 already established that motif space is a continuum where forced
    assignment is the wrong default.
    """
    from sklearn.cluster import (AffinityPropagation, AgglomerativeClustering, HDBSCAN,
                                 SpectralClustering)
    N = unit(M)
    D = cosine_distance(M)
    if algorithm in ("ward", "average", "complete"):
        if algorithm == "ward":
            # Ward is defined on Euclidean geometry only; applying it to a precomputed cosine
            # matrix is a category error, so it is run on the L2-normalised vectors where
            # Euclidean distance is a monotone function of cosine distance.
            return AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(N)
        return AgglomerativeClustering(n_clusters=K, metric="precomputed",
                                       linkage=algorithm).fit_predict(D)
    if algorithm == "spectral":
        A = np.exp(-(D ** 2) / (2 * float(np.median(D[D > 0])) ** 2 + EPS))
        return SpectralClustering(n_clusters=K, affinity="precomputed", random_state=seed,
                                  assign_labels="kmeans", n_init=20).fit_predict(A)
    if algorithm == "hdbscan":
        return HDBSCAN(min_cluster_size=int(param or 3), metric="precomputed",
                       allow_single_cluster=False).fit_predict(D.astype(float))
    if algorithm == "affinity_propagation":
        S = -D
        pref = np.quantile(S[~np.eye(len(S), dtype=bool)], param if param is not None else 0.5)
        ap = AffinityPropagation(affinity="precomputed", preference=pref, random_state=seed,
                                 damping=0.9, max_iter=1000, convergence_iter=50)
        lab = ap.fit_predict(S)
        return lab if lab is not None and len(set(lab.tolist())) > 1 else np.zeros(len(S), int)
    raise ValueError(f"unknown algorithm {algorithm}")


# ── internal validity, label-free ────────────────────────────────────────────
def internal_indices(M: np.ndarray, lab: np.ndarray) -> dict:
    """Silhouette, Calinski–Harabasz, Davies–Bouldin — none of which see a chemistry label."""
    from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                                 silhouette_score)
    keep = lab >= 0
    out = {"n_clusters": int(len({int(v) for v in lab[keep]})),
           "n_unassigned": int((~keep).sum())}
    if out["n_clusters"] < 2 or keep.sum() < 3:
        return {**out, "silhouette": np.nan, "calinski_harabasz": np.nan,
                "davies_bouldin": np.nan}
    D = cosine_distance(M)[np.ix_(keep, keep)]
    N = unit(M)[keep]
    l = lab[keep]
    # No bare except: an index that cannot be computed is a defect to surface, not a NaN to
    # propagate. Only the genuinely undefined case — a cluster of size 1 for Davies-Bouldin —
    # is tolerated, and it is named.
    sil = float(silhouette_score(D, l, metric="precomputed"))
    ch = float(calinski_harabasz_score(N, l))
    try:
        db = float(davies_bouldin_score(N, l))
    except ValueError:
        db = np.nan          # undefined when a cluster has a single member
    return {**out, "silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db}


def membership_entropy(lab: np.ndarray) -> float:
    """Normalised entropy of the cluster-size distribution. 1 = balanced, 0 = one giant."""
    v, n = np.unique(lab[lab >= 0], return_counts=True)
    if len(v) < 2:
        return 0.0
    p = n / n.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(v)))


def neighbour_preservation(M: np.ndarray, lab: np.ndarray, k: int = 5) -> float:
    """Fraction of each molecule's k nearest neighbours that share its cluster.

    Label-free: it asks whether the partition respects local geometry, not whether it respects
    chemistry. A partition that cuts through dense neighbourhoods scores low here even if it
    happens to align with an ontology.
    """
    D = cosine_distance(M)
    np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, axis=1)[:, :k]
    keep = lab >= 0
    if keep.sum() == 0:
        return np.nan
    return float(np.mean([(lab[nn[i]] == lab[i]).mean() for i in range(len(M)) if keep[i]]))


# ── stability under resampling ───────────────────────────────────────────────
def bootstrap_stability(M: np.ndarray, algorithm: str, K: int | None, n_boot: int = 40,
                        frac: float = 0.85, seed: int = 0, param=None) -> dict:
    """Resample molecules, re-cluster, and measure agreement on the shared members.

    Reported as ARI on the intersection and as a co-assignment consensus. A K that is an
    artefact of the algorithm rather than of the data falls apart here, which is the only
    question Section 1 is really asking.
    """
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(seed)
    n = len(M)
    base = fit(algorithm, M, K, seed, param)
    aris, co, cnt = [], np.zeros((n, n)), np.zeros((n, n))
    ks = []
    for b in range(n_boot):
        idx = np.sort(rng.choice(n, int(frac * n), replace=False))
        try:
            lab = fit(algorithm, M[idx], K, seed, param)
        except Exception:                                          # pragma: no cover
            continue
        ks.append(len({int(v) for v in lab if v >= 0}))
        ok = lab >= 0
        aris.append(adjusted_rand_score(base[idx][ok], lab[ok]))
        ii = idx[ok]
        same = (lab[ok][:, None] == lab[ok][None, :]).astype(float)
        co[np.ix_(ii, ii)] += same
        cnt[np.ix_(ii, ii)] += 1
    C = np.where(cnt > 0, co / np.maximum(cnt, 1), 0.0)
    iu = np.triu_indices(n, 1)
    return {"bootstrap_ari_mean": float(np.mean(aris)) if aris else np.nan,
            "bootstrap_ari_sd": float(np.std(aris)) if aris else np.nan,
            "consensus_dispersion": float((4 * (C[iu] - 0.5) ** 2).mean()),
            "mean_coassignment": float(C[iu].mean()),
            "k_realised_mean": float(np.mean(ks)) if ks else np.nan,
            "k_realised_sd": float(np.std(ks)) if ks else np.nan,
            "consensus": C}


def cluster_survival(M: np.ndarray, algorithm: str, K: int | None, n_boot: int = 40,
                     frac: float = 0.85, seed: int = 0, param=None) -> dict:
    """Per-cluster recovery: how often each base cluster reappears as a coherent group.

    A partition whose mean stability is respectable can still contain one cluster that never
    survives. Reporting only the mean hides exactly the cluster a reader would care about.
    """
    rng = np.random.default_rng(seed + 1)
    n = len(M)
    base = fit(algorithm, M, K, seed, param)
    ids = sorted({int(v) for v in base if v >= 0})
    jac = {c: [] for c in ids}
    for _ in range(n_boot):
        idx = np.sort(rng.choice(n, int(frac * n), replace=False))
        try:
            lab = fit(algorithm, M[idx], K, seed, param)
        except Exception:                                          # pragma: no cover
            continue
        for c in ids:
            members = set(idx[base[idx] == c].tolist())
            if not members:
                continue
            best = 0.0
            for d in {int(v) for v in lab if v >= 0}:
                cand = set(idx[lab == d].tolist())
                u = len(members | cand)
                if u:
                    best = max(best, len(members & cand) / u)
            jac[c].append(best)
    surv = {c: float(np.mean(v)) if v else np.nan for c, v in jac.items()}
    vals = [v for v in surv.values() if np.isfinite(v)]
    return {"per_cluster_jaccard": surv,
            "min_survival": float(np.min(vals)) if vals else np.nan,
            "mean_survival": float(np.mean(vals)) if vals else np.nan,
            "n_clusters_below_0.5": int(sum(v < 0.5 for v in vals))}


def sweep(M: np.ndarray, k_grid=K_GRID, algorithms=FIXED_K_ALGORITHMS, n_boot: int = 40,
          seed: int = 0, log=None) -> "pd.DataFrame":
    """The Section 1 grid: every algorithm at every K, with stability and internal validity."""
    import pandas as pd
    rows = []
    for algo in algorithms:
        for K in k_grid:
            if K >= len(M):
                continue
            try:
                lab = fit(algo, M, K, seed)
            except Exception as exc:                               # pragma: no cover
                rows.append({"algorithm": algo, "K": K, "usable": False, "error": str(exc)[:60]})
                continue
            iv = internal_indices(M, lab)
            bs = bootstrap_stability(M, algo, K, n_boot=n_boot, seed=seed)
            sv = cluster_survival(M, algo, K, n_boot=n_boot, seed=seed)
            rows.append({"algorithm": algo, "K": K, "usable": True, **iv,
                         "membership_entropy": membership_entropy(lab),
                         "neighbour_preservation": neighbour_preservation(M, lab),
                         **{k: v for k, v in bs.items() if k != "consensus"},
                         "min_cluster_survival": sv["min_survival"],
                         "mean_cluster_survival": sv["mean_survival"],
                         "n_clusters_unstable": sv["n_clusters_below_0.5"]})
            if log:
                r = rows[-1]
                log(f"    {algo:10s} K={K:2d}  sil {r['silhouette']:+.3f}  "
                    f"bootARI {r['bootstrap_ari_mean']:.3f}  survival "
                    f"{r['mean_cluster_survival']:.3f} (min {r['min_cluster_survival']:.3f})  "
                    f"nbr {r['neighbour_preservation']:.3f}")
    return pd.DataFrame(rows)
