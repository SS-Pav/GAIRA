"""GAIRA V7 — Phase 02.5: null geometries.

Every geometric claim in this phase is a claim that observed structure exceeds what generic
Raman statistics produce. Six nulls, each destroying one specific kind of structure while
preserving the rest, so that when a claim survives it is clear *what* it survived.

Phase 02 established the scale that matters: its band-permutation null had a mean edge weight
of 0.156 against an observed 0.174, and only 61 of 1225 pairs cleared p < 0.01. Structure in
this space is not obvious, and nothing here should be believed without a null beside it.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12
NULLS = ("band_position", "intensity_permutation", "class_label", "molecule_activation",
         "source_label", "degree_preserving_graph")


def band_position_null(H: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Circularly shift each motif. Preserves peak count, prominences and envelope; destroys
    the *relationship between* motifs' band positions."""
    n_bins = H.shape[1]
    return np.array([np.roll(h, int(rng.integers(30, n_bins - 30))) for h in H])


def intensity_permutation_null(H: np.ndarray, grid: np.ndarray,
                               rng: np.random.Generator) -> np.ndarray:
    """Keep peak positions, scramble relative intensities among them.

    Tests whether a relationship is carried by *which* bands a motif has, or by how strongly it
    has them. Two motifs sharing a band list but not its weighting survive this null; two that
    only agreed on the tallest peak do not.
    """
    from scipy.signal import find_peaks
    out = np.zeros_like(H)
    for i, h in enumerate(H):
        x = h / (h.max() + EPS)
        idx, _ = find_peaks(x, prominence=0.03)
        if idx.size < 2:
            out[i] = h
            continue
        scale = np.ones(H.shape[1])
        perm = rng.permutation(idx)
        for src, dst in zip(idx, perm):
            scale[max(0, dst - 6):dst + 7] = x[src] / (x[dst] + EPS)
        out[i] = np.clip(h * scale, 0, None)
    return out


def label_permutation(labels: list[str], rng: np.random.Generator) -> np.ndarray:
    """Shuffle labels over a FIXED geometry. Used only to calibrate enrichment statistics."""
    return rng.permutation(np.asarray(labels))


def molecule_activation_null(A_mol: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute each motif's activation profile across molecules, preserving its prevalence.

    Column-wise, so a motif that activates broadly still activates broadly — what is destroyed
    is *which* molecules, i.e. any co-response between motifs.
    """
    out = np.asarray(A_mol, float).copy()
    for j in range(out.shape[1]):
        out[:, j] = rng.permutation(out[:, j])
    return out


def degree_preserving_graph_null(W: np.ndarray, rng: np.random.Generator,
                                 n_swap_multiplier: int = 10) -> np.ndarray:
    """Rewire the graph by double-edge swaps, preserving every node's degree exactly.

    Modularity is high in graphs that merely have heterogeneous degrees, so without a genuinely
    degree-preserving null "this graph is modular" is close to unfalsifiable. An earlier version
    of this function permuted all weights across all positions, which changes the topology
    completely and is a *weight-permutation* null wearing a degree-preserving label — it made
    the observed modularity look far more exceptional than it is.

    The topology is rewired with networkx's double-edge swap (degree preserved exactly), then
    the observed edge weights are reassigned at random to the new edges, so the weight
    distribution is preserved too.
    """
    import networkx as nx
    n = W.shape[0]
    Wb = (W > 0).astype(int)
    G = nx.from_numpy_array(Wb)
    G.remove_edges_from(nx.selfloop_edges(G))
    m = G.number_of_edges()
    if m >= 2:
        try:
            nx.double_edge_swap(G, nswap=n_swap_multiplier * m, max_tries=200 * m,
                                seed=int(rng.integers(0, 2 ** 31 - 1)))
        except nx.NetworkXAlgorithmError:      # too few swappable edges — report as-is
            pass
    weights = W[np.triu_indices(n, 1)]
    weights = weights[weights > 0]
    weights = rng.permutation(weights)
    out = np.zeros((n, n))
    for k, (i, j) in enumerate(G.edges()):
        w = float(weights[k % max(weights.size, 1)]) if weights.size else 1.0
        out[i, j] = out[j, i] = w
    np.fill_diagonal(out, 0.0)
    return out


def null_distance_ensemble(H, grid, dist_fn, kind: str, n: int = 40, seed: int = 0,
                           **kw) -> list[np.ndarray]:
    """`n` null distance matrices under one null model, for use with `metrics.null_separation`."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        if kind == "band_position":
            Hn = band_position_null(H, rng)
        elif kind == "intensity_permutation":
            Hn = intensity_permutation_null(H, grid, rng)
        else:
            raise ValueError(f"{kind} is not a spectral null")
        out.append(np.asarray(dist_fn(Hn), float))
    return out


def enrichment_pvalue(observed: float, labels: list[str], stat_fn, n_perm: int = 500,
                      seed: int = 0) -> tuple[float, float, float]:
    """One-sided empirical p for a label statistic against the label-permutation null.

    Returns (p, null mean, null sd). This is the only legitimate way to say "this community is
    enriched for lipids": the geometry is fixed, the labels move.
    """
    rng = np.random.default_rng(seed)
    draws = np.array([stat_fn(label_permutation(labels, rng)) for _ in range(n_perm)])
    p = float((draws >= observed).mean())
    return p, float(draws.mean()), float(draws.std())
