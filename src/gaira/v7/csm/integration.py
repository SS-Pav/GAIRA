"""GAIRA V7 — Phase 02: the five integration candidates and the composite that chooses.

The point of running five methods is an auditable choice. The comparison table is published
whichever candidate wins (pre-registration §5); a table showing *why* the winner won is the
deliverable, not a footnote.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import NMF

from .graph import build_graph, consensus_partition

METHODS = ("graph_community", "consensus_clustering", "spectral",
           "meta_nmf", "hybrid")

# composite weights, pre-registered (§5)
CRITERIA = {
    "consensus_stability": (0.20, +1),
    "within_cohesion": (0.15, +1),
    "between_separation": (0.15, +1),
    "chemical_coherence": (0.10, +1),
    "retained_lsm_information": (0.15, +1),
    "heldout_reconstruction": (0.10, +1),
    "hyperparameter_sensitivity": (0.05, -1),
    "singleton_fraction": (0.05, -1),
    "csm_redundancy": (0.05, -1),
}
M_SWEEP = range(2, 26)
PLATEAU_TOLERANCE = 0.02
EPS = 1e-12


def _labels_to_groups(labels) -> list[list[int]]:
    out: dict[int, list[int]] = {}
    for i, k in enumerate(labels):
        out.setdefault(int(k), []).append(i)
    return [sorted(v) for _, v in sorted(out.items())]


# ── the five candidates ──────────────────────────────────────────────────────
def run_graph_community(W, motif_ids, classes, types, tau) -> list[list[int]]:
    """Louvain modularity on the thresholded graph. Chooses its own M."""
    G = build_graph(W, motif_ids, classes, types, tau)
    part = consensus_partition(G)
    idx = {m: i for i, m in enumerate(motif_ids)}
    groups: dict[int, list[int]] = {}
    for node, k in part.items():
        groups.setdefault(k, []).append(idx[node])
    return [sorted(v) for _, v in sorted(groups.items())]


def run_consensus_clustering(W, M, resamples: int = 50, seed: int = 0) -> list[list[int]]:
    """Average linkage on 1 - w, over a consensus matrix built from 50 subsampled runs.

    Well-understood stability semantics and a dendrogram, at the cost of ignoring graph
    topology entirely — it sees only the pairwise weights.
    """
    n = W.shape[0]
    rng = np.random.default_rng(seed)
    C, N = np.zeros((n, n)), np.zeros((n, n))
    for _ in range(resamples):
        keep = np.sort(rng.choice(n, size=max(3, int(0.8 * n)), replace=False))
        D = 1.0 - W[np.ix_(keep, keep)]
        np.fill_diagonal(D, 0.0)
        lab = fcluster(linkage(squareform(D, checks=False), method="average"),
                       t=min(M, len(keep)), criterion="maxclust")
        for a in range(len(keep)):
            for b in range(len(keep)):
                N[keep[a], keep[b]] += 1
                if lab[a] == lab[b]:
                    C[keep[a], keep[b]] += 1
    Cm = np.where(N > 0, C / np.maximum(N, 1), 0.0)
    D = 1.0 - Cm
    np.fill_diagonal(D, 0.0)
    lab = fcluster(linkage(squareform((D + D.T) / 2, checks=False), method="average"),
                   t=M, criterion="maxclust")
    return _labels_to_groups(lab)


def run_spectral(W, M, seed: int = 0) -> list[list[int]]:
    """Eigen-decomposition of the graph Laplacian. Handles non-convex structure; needs M."""
    A = W.copy()
    np.fill_diagonal(A, 0.0)
    if A.sum() <= 0:
        return [[i] for i in range(A.shape[0])]
    sc = SpectralClustering(n_clusters=M, affinity="precomputed", random_state=seed,
                            assign_labels="kmeans", n_init=10)
    return _labels_to_groups(sc.fit_predict(A))


def run_meta_nmf(A_mol, M, seed: int = 0) -> list[list[int]]:
    """Sparse non-negative factorisation of the activation matrix, A ~= U V.

    Its structural weakness is stated in the architecture and is the reason it is a candidate
    rather than the plan: it sees ONE of the seven edge channels — activation co-occurrence —
    and discards shape, bands, positions, bootstrap behaviour, substitutability and provenance.
    """
    model = NMF(n_components=M, init="nndsvda", random_state=seed, max_iter=2000,
                alpha_H=0.01, l1_ratio=1.0)
    model.fit(A_mol)
    V = model.components_                      # M x n_lsm
    return _labels_to_groups(V.argmax(axis=0))


def run_hybrid(W, motif_ids, classes, types, tau, A_mol, seed: int = 0) -> list[list[int]]:
    """Graph communities establish WHICH motifs correspond; a constrained non-negative refit
    establishes HOW MUCH each contributes.

    Keeps the multi-feature evidence of the graph and recovers the soft overlapping membership
    that a hard partition throws away. Motifs whose refit mass moves decisively to another
    community are reassigned; the rest keep their community.
    """
    groups = run_graph_community(W, motif_ids, classes, types, tau)
    n = W.shape[0]
    M = len(groups)
    if M < 2:
        return groups
    H0 = np.zeros((M, n))
    for k, g in enumerate(groups):
        H0[k, g] = 1.0
    W0 = np.maximum(A_mol @ H0.T, EPS)
    model = NMF(n_components=M, init="custom", random_state=seed, max_iter=2000,
                alpha_H=0.01, l1_ratio=1.0)
    model.fit_transform(np.maximum(A_mol, 0.0), W=W0, H=np.maximum(H0, EPS))
    return _labels_to_groups(model.components_.argmax(axis=0))


# ── scoring ──────────────────────────────────────────────────────────────────
def score_partition(groups: list[list[int]], W: np.ndarray, H: np.ndarray,
                    classes: list[str], X: np.ndarray, folds: np.ndarray,
                    consensus_fn, stability: float, sensitivity: float) -> dict:
    """The nine pre-registered criteria for one candidate partition."""
    from .consensus import consensus_spectrum

    n = H.shape[0]
    lab = np.zeros(n, int)
    for k, g in enumerate(groups):
        lab[g] = k
    M = len(groups)

    within = [np.mean([W[a, b] for i, a in enumerate(g) for b in g[i + 1:]])
              for g in groups if len(g) > 1]
    between = [W[a, b] for i in range(n) for b in range(i + 1, n) if lab[i] != lab[b]
               for a in [i]]
    csms = np.array([consensus_spectrum(H[g], np.ones(len(g))) for g in groups])
    csms = csms / (np.linalg.norm(csms, axis=1, keepdims=True) + EPS)

    # retained LSM information: can the CSM set rebuild the LSM set?
    from scipy.optimize import nnls
    res = tot = 0.0
    for h in H:
        c = nnls(csms.T, h)[0]
        res += float(((h - c @ csms) ** 2).sum()); tot += float((h ** 2).sum())
    retained = max(0.0, 1.0 - res / (tot + EPS))

    # held-out molecule reconstruction, analyte-grouped folds
    ho = []
    for f in sorted(set(folds)):
        te = folds == f
        r = t = 0.0
        for x in X[te]:
            c = nnls(csms.T, x)[0]
            r += float(((x - c @ csms) ** 2).sum()); t += float((x ** 2).sum())
        ho.append(max(0.0, 1.0 - r / (t + EPS)))

    # chemical coherence: a group is coherent if its classes share a broad chemistry
    coh = []
    for g in groups:
        cs = [classes[i] for i in g]
        coh.append(max(cs.count(c) for c in set(cs)) / len(cs))

    R = csms @ csms.T
    np.fill_diagonal(R, 0.0)

    return {
        "M": M,
        "consensus_stability": float(stability),
        "within_cohesion": float(np.mean(within)) if within else 0.0,
        "between_separation": float(1.0 - np.mean(between)) if between else 1.0,
        "chemical_coherence": float(np.mean(coh)),
        "retained_lsm_information": float(retained),
        "heldout_reconstruction": float(np.mean(ho)),
        "hyperparameter_sensitivity": float(sensitivity),
        "singleton_fraction": float(np.mean([len(g) == 1 for g in groups])),
        "csm_redundancy": float(R.max()) if M > 1 else 0.0,
    }


def composite(rows: list[dict]) -> np.ndarray:
    """Min-max normalise each criterion across candidates, apply direction, weight, sum.

    Normalising across candidates rather than to an absolute scale is deliberate: the question
    is which method is best on this graph, and an absolute scale would need constants nobody
    can justify in advance.
    """
    out = np.zeros(len(rows))
    for crit, (w, direction) in CRITERIA.items():
        v = np.array([r[crit] for r in rows], float)
        span = v.max() - v.min()
        z = np.full_like(v, 0.5) if span < EPS else (v - v.min()) / span
        out += w * (z if direction > 0 else 1.0 - z)
    return out


def select_M(rows: list[dict], tolerance: float = PLATEAU_TOLERANCE) -> dict:
    """Smallest M on the contiguous Pareto plateau containing the maximum.

    Contiguity is carried over from the Phase 01 k_c correction, where a literal reading of
    "plateau" admitted a non-contiguous set and selected k=1 for a 20-molecule class whose
    composite peaked at k=9.
    """
    comp = composite(rows)
    order = np.argsort([r["M"] for r in rows])
    rows = [rows[i] for i in order]
    comp = comp[order]
    peak = int(np.argmax(comp))
    lo = peak
    while lo - 1 >= 0 and comp[lo - 1] >= comp[peak] - tolerance:
        lo -= 1
    hi = peak
    while hi + 1 < len(comp) and comp[hi + 1] >= comp[peak] - tolerance:
        hi += 1
    return {"M": int(rows[lo]["M"]), "peak_M": int(rows[peak]["M"]),
            "plateau": [int(rows[lo]["M"]), int(rows[hi]["M"])],
            "composite": float(comp[lo]),
            "rationale": (f"composite peaks at M={rows[peak]['M']} ({comp[peak]:.4f}); the "
                          f"contiguous plateau within {tolerance} spans M="
                          f"{rows[lo]['M']}–{rows[hi]['M']}; smallest on the plateau selected")}
