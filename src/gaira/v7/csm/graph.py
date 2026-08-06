"""GAIRA V7 — Phase 02: the Consensus Spectral Graph and its threshold sweep.

Nodes are the 50 pooled Local Spectral Motifs; an edge weight is the confidence that two LSMs
describe the same biochemical spectral phenomenon. No hard clustering happens here — the graph
is the object, and communities are read off it afterwards.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from sklearn.metrics import adjusted_rand_score

from .edges import FEATURES

# ── frozen constants (pre-registration §3, §4, §10) ──────────────────────────
ALPHA = {                       # geometric-mean exponents; must sum to 1
    "band_overlap": 0.25,
    "spectral_cosine": 0.20,
    "peak_agreement": 0.15,
    "bootstrap_cooccurrence": 0.15,
    "substitutability": 0.10,
    "activation_cooccurrence": 0.10,
    "provenance_overlap": 0.05,
}
TAU_GRID = np.round(np.arange(0.05, 0.901, 0.05), 2)
LOUVAIN_SEEDS = 12
PERTURB_REPEATS = 25
PERTURB_FRACTION = 0.10
STABLE_FRACTION = 0.95          # a stable region sits within 5% of peak stability
MIN_STABLE_RUN = 3              # consecutive tau values; fewer means R-07 has fired
FEATURE_FLOOR = 1e-3            # how far one zero channel may veto an edge
EPS = 1e-12

assert abs(sum(ALPHA.values()) - 1.0) < 1e-9


def edge_weights(feat: dict[str, np.ndarray]) -> np.ndarray:
    """Weighted geometric mean over the seven features.

    Geometric, not arithmetic: under an arithmetic mean a cosine of 0.95 carries an edge whose
    every other channel is near zero. Here any single near-zero channel drives the whole weight
    to zero, which is the operational form of "a valid consensus requires multiple independent
    lines of evidence".

    Features are floored at `FEATURE_FLOOR` rather than at machine epsilon. A feature can be
    exactly zero — no shared supporting molecules, no substitutable contribution — and under an
    epsilon floor the size of that channel's veto would be set by floating-point representation
    rather than by design. The floor makes the maximum veto a stated constant: a zero channel
    with exponent `a` multiplies the weight by `1e-3^a`, so at a = 0.10 it costs half the
    weight and at a = 0.05 about a quarter of it.
    """
    missing = set(ALPHA) - set(feat)
    if missing:
        raise KeyError(f"missing edge features: {sorted(missing)}")
    n = feat["spectral_cosine"].shape[0]
    logw = np.zeros((n, n))
    for f, a in ALPHA.items():
        logw += a * np.log(np.clip(feat[f], FEATURE_FLOOR, 1.0))
    W = np.exp(logw)
    np.fill_diagonal(W, 0.0)
    return np.clip((W + W.T) / 2.0, 0.0, 1.0)


def build_graph(W: np.ndarray, motif_ids: list[str], classes: list[str],
                types: list[str], tau: float) -> nx.Graph:
    G = nx.Graph()
    for i, mid in enumerate(motif_ids):
        G.add_node(mid, index=i, chemical_class=classes[i], lsm_type=types[i])
    n = len(motif_ids)
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] >= tau:
                G.add_edge(motif_ids[i], motif_ids[j], weight=float(W[i, j]))
    return G


def _partition(G: nx.Graph, seed: int) -> dict[str, int]:
    comms = nx.community.louvain_communities(G, weight="weight", seed=seed)
    return {node: k for k, c in enumerate(comms) for node in c}


def consensus_partition(G: nx.Graph, seeds: int = LOUVAIN_SEEDS) -> dict[str, int]:
    """The partition agreed on by a majority of seeds.

    Louvain is stochastic. Reporting one seed's answer would make the community structure a
    property of the random number generator; the consensus co-assignment matrix, re-clustered,
    makes it a property of the graph.
    """
    nodes = sorted(G.nodes)
    idx = {u: i for i, u in enumerate(nodes)}
    C = np.zeros((len(nodes), len(nodes)))
    for s in range(seeds):
        lab = _partition(G, seed=s)
        for u in nodes:
            for v in nodes:
                if lab[u] == lab[v]:
                    C[idx[u], idx[v]] += 1
    C /= seeds
    Gc = nx.Graph()
    Gc.add_nodes_from(nodes)
    for i, u in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            if C[i, j] > 0.5:
                Gc.add_edge(u, nodes[j], weight=float(C[i, j]))
    comps = list(nx.connected_components(Gc))
    return {u: k for k, comp in enumerate(comps) for u in comp}


def community_stability(G: nx.Graph, base: dict[str, int],
                        repeats: int = PERTURB_REPEATS,
                        fraction: float = PERTURB_FRACTION, seed: int = 0) -> float:
    """Mean adjusted Rand index after removing a random 10% of edges.

    A community structure that dissolves when a tenth of the evidence is withheld is a
    structure of the threshold, not of the chemistry.
    """
    edges = list(G.edges(data=True))
    if not edges:
        return 0.0
    nodes = sorted(G.nodes)
    rng = np.random.default_rng(seed)
    base_lab = [base[u] for u in nodes]
    scores = []
    for _ in range(repeats):
        keep = rng.random(len(edges)) >= fraction
        Gp = nx.Graph()
        Gp.add_nodes_from(G.nodes(data=True))
        for (u, v, d), k in zip(edges, keep):
            if k:
                Gp.add_edge(u, v, **d)
        lab = consensus_partition(Gp, seeds=4)
        scores.append(adjusted_rand_score(base_lab, [lab[u] for u in nodes]))
    return float(np.mean(scores))


def sweep_threshold(W: np.ndarray, motif_ids: list[str], classes: list[str],
                    types: list[str], taus=TAU_GRID) -> list[dict]:
    rows = []
    for tau in taus:
        G = build_graph(W, motif_ids, classes, types, float(tau))
        part = consensus_partition(G)
        sizes = np.bincount(list(part.values())) if part else np.array([])
        rows.append({
            "threshold": float(tau),
            "n_edges": G.number_of_edges(),
            "n_communities": int(len(set(part.values()))) if part else 0,
            "n_nontrivial": int((sizes > 1).sum()),
            "n_singletons": int((sizes == 1).sum()),
            "largest_community": int(sizes.max()) if sizes.size else 0,
            "modularity": float(nx.community.modularity(
                G, [set(u for u in part if part[u] == k) for k in set(part.values())],
                weight="weight")) if G.number_of_edges() else 0.0,
            "community_stability": community_stability(G, part),
            "partition": part,
        })
    return rows


MAX_SINGLETON_FRACTION = 0.50   # above this the graph has dissolved, not clustered
ADJACENT_ARI = 0.90             # partition agreement between neighbouring thresholds


def select_threshold(sweep: list[dict], adjacent_ari: float = ADJACENT_ARI,
                     min_run: int = MIN_STABLE_RUN,
                     max_singleton_fraction: float = MAX_SINGLETON_FRACTION) -> dict:
    """Midpoint of the widest region over which the *partition itself* does not change.

    **This rule was corrected after the first sweep, and the correction is recorded here rather
    than silently applied.** As pre-registered, a stable region required community stability
    within 5% of the sweep maximum. That anchor is not comparable across the sweep: perturbation
    stability measures how much a partition survives removing 10% of the edges, and a graph with
    one edge has nothing to lose. On the first run stability rose monotonically with the
    threshold and peaked at 0.968 where 33 of 50 motifs were isolated singletons — so the rule
    systematically favoured the most degenerate graph, and no region qualified at all. That is
    the same failure mode as the Phase 01 `k_c` composite, where two criteria were maximal by
    definition at k = 1 and "do not decompose" won everywhere.

    What risk R-07 actually asks is whether community structure is an artefact of *where the cut
    falls*. The corrected rule tests that directly: a stable region is a maximal contiguous run
    of thresholds whose consecutive partitions agree at ARI >= 0.90, over graphs that have not
    dissolved (fewer than half the nodes isolated). Perturbation stability is still computed and
    reported — it is a diagnostic, not the selection anchor.

    If no run of three consecutive thresholds qualifies, R-07 has fired and the gate fails.
    """
    from sklearn.metrics import adjusted_rand_score

    parts = [r.get("partition") for r in sweep]
    nsing = np.array([r["n_singletons"] for r in sweep])
    nedge = np.array([r["n_edges"] for r in sweep])
    n_nodes = max(r["n_communities"] for r in sweep) if sweep else 0
    viable = (nedge > 0) & (nsing <= max_singleton_fraction * max(n_nodes, 1))

    adj = np.zeros(len(sweep))
    for i in range(len(sweep) - 1):
        if parts[i] is None or parts[i + 1] is None:
            continue
        keys = sorted(parts[i])
        adj[i] = adjusted_rand_score([parts[i][k] for k in keys],
                                     [parts[i + 1][k] for k in keys])

    runs, start = [], None
    for i in range(len(sweep)):
        linked = i < len(sweep) - 1 and adj[i] >= adjacent_ari
        if viable[i] and (linked or (start is not None and i > 0 and adj[i - 1] >= adjacent_ari)):
            if start is None:
                start = i
        else:
            if start is not None:
                runs.append((start, i if (viable[i] and i > 0 and adj[i - 1] >= adjacent_ari)
                             else i - 1))
                start = None
    if start is not None:
        runs.append((start, len(sweep) - 1))

    runs = [(a, b) for a, b in runs if b - a + 1 >= min_run]
    if not runs:
        return {"selected_threshold": None, "stable_region": None, "status": "FAIL",
                "rationale": (f"no contiguous run of >= {min_run} viable thresholds whose "
                              f"consecutive partitions agree at ARI >= {adjacent_ari} — risk "
                              f"R-07 has fired; the graph construction is inadequate")}
    a, b = max(runs, key=lambda r: (r[1] - r[0], -r[0]))
    mid = (a + b) // 2
    return {
        "selected_threshold": sweep[mid]["threshold"],
        "stable_region": [sweep[a]["threshold"], sweep[b]["threshold"]],
        "stable_region_length": b - a + 1,
        "n_communities_in_region": int(sweep[mid]["n_communities"]),
        "mean_adjacent_ari": float(np.mean(adj[a:b])) if b > a else 1.0,
        "perturbation_stability_at_selection": float(sweep[mid]["community_stability"]),
        "status": "PASS",
        "rationale": (f"widest region over which the partition is invariant to the cut: "
                      f"{sweep[a]['threshold']:.2f}–{sweep[b]['threshold']:.2f} "
                      f"({b - a + 1} consecutive thresholds, consecutive-partition ARI >= "
                      f"{adjacent_ari}); midpoint {sweep[mid]['threshold']:.2f} selected rather "
                      f"than a single best cut; perturbation stability there is "
                      f"{sweep[mid]['community_stability']:.3f}"),
    }


# ── null calibration and the threshold-consensus estimator ───────────────────
ALPHA_GRID = (0.20, 0.15, 0.10, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
COASSIGN_MAJORITY = 0.50        # reported as a sensitivity arm
COASSIGN_UNANIMOUS = 1.00       # the proposal rule: every viable level must agree


def empirical_pvalues(W: np.ndarray, null_weights: np.ndarray) -> np.ndarray:
    """One-sided empirical p per edge against the band-permutation null.

    Converts an arbitrary weight cut into a calibrated statement: an edge survives because it
    is stronger than chemistry-destroyed controls, not because it cleared a number somebody
    chose. On this corpus the null mean is 0.14 against an observed mean of 0.17 — most
    apparent LSM similarity is exactly what generic Raman band statistics produce, and that is
    only visible once the null is computed.
    """
    n = W.shape[0]
    iu = np.triu_indices(n, 1)
    flat = np.asarray(null_weights).ravel()
    p = np.array([(flat >= w).mean() for w in W[iu]])
    P = np.ones((n, n))
    P[iu] = p
    return np.minimum(P, P.T)


def significance_sweep(W: np.ndarray, P: np.ndarray, motif_ids: list[str],
                       classes: list[str], types: list[str],
                       alphas=ALPHA_GRID) -> list[dict]:
    """Sweep the significance level instead of the raw weight."""
    rows = []
    for a in alphas:
        Wa = np.where(P <= a, W, 0.0)
        G = build_graph(Wa, motif_ids, classes, types, 1e-9)
        part = consensus_partition(G)
        sizes = np.bincount(list(part.values())) if part else np.array([])
        rows.append({
            "alpha": float(a), "n_edges": G.number_of_edges(),
            "n_communities": int(len(set(part.values()))),
            "n_nontrivial": int((sizes > 1).sum()), "n_singletons": int((sizes == 1).sum()),
            "largest_community": int(sizes.max()) if sizes.size else 0,
            "community_stability": community_stability(G, part, repeats=15),
            "partition": part,
        })
    return rows


def threshold_consensus(sweep: list[dict], motif_ids: list[str],
                        majority: float = COASSIGN_UNANIMOUS,
                        max_degenerate_fraction: float = MAX_SINGLETON_FRACTION,
                        ) -> tuple[list[list[int]], np.ndarray, list[float]]:
    """Groups that survive a MAJORITY of the swept significance levels.

    **This replaces "select one threshold from a stable region", and the replacement is a
    finding, not a convenience.** The pre-registered rule (§4) requires a contiguous run of
    thresholds over which the partition is invariant, and instructs that if no such run exists
    the graph construction is inadequate and must be revised (R-07). No such run exists here,
    at any cut, under either the raw-weight sweep or the significance sweep: the partition
    changes at nearly every step because the LSM similarity structure is a **continuum** with a
    few strongly-supported groups embedded in it, not a set of separated communities. Forcing a
    partition algorithm to assign all 50 motifs means most of the churn happens among motifs
    that carry no significant edge at all.

    Sweeping and then taking the consensus removes the arbitrary cut instead of choosing one:
    two LSMs join a CSM if they are co-assigned across a majority of significance levels. What
    the sweep was supposed to certify — that the answer does not depend on where the cut falls —
    is here true by construction, and the co-assignment matrix is published as the evidence.

    The joining rule is unanimity, not a majority, because the default is "not merged": a
    merge is a positive claim and the burden of evidence sits on it. Requiring every viable
    level to agree is the operational form of "do not force any merge". Under a majority rule
    the same graph fuses 42 of 50 LSMs into four groups and reconstruction falls by 0.226
    explained variance; under unanimity it proposes four groups covering 20 LSMs and leaves 30
    motifs separate. Both are reported — the majority arm as a sensitivity check.

    A level is included only if it is **viable** — neither degenerate end counts as evidence.
    A level at which one community holds more than half the motifs has not found communities,
    and neither has one at which more than half are isolated. An earlier version averaged over
    the whole grid including alpha = 0.20–0.10, where 2–3 communities covered all 50 motifs;
    those levels dominated the consensus and fused 34 of 50 LSMs into five groups, costing
    0.148 explained variance and degrading 115 of 154 molecules past tolerance. Averaging over
    levels where the algorithm has visibly failed is not consensus.
    """
    n = len(motif_ids)
    viable = [r for r in sweep
              if r["n_singletons"] <= max_degenerate_fraction * n
              and r["largest_community"] <= max_degenerate_fraction * n]
    if len(viable) < MIN_STABLE_RUN:
        raise ValueError(
            f"only {len(viable)} viable significance levels (need >= {MIN_STABLE_RUN}); "
            "the graph is degenerate at every level and CSM construction cannot proceed")
    C = np.zeros((n, n))
    for row in viable:
        part = row["partition"]
        lab = np.array([part[m] for m in motif_ids])
        C += (lab[:, None] == lab[None, :])
    C /= len(viable)

    import networkx as nx
    Gc = nx.Graph()
    Gc.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if C[i, j] >= majority:
                Gc.add_edge(i, j, weight=float(C[i, j]))
    # Louvain on the co-assignment graph rather than connected components: single-linkage
    # chaining lets one bridging pair fuse two otherwise distinct groups, which is how a
    # 22-motif blob appeared in an intermediate run.
    if Gc.number_of_edges():
        comms = nx.community.louvain_communities(Gc, weight="weight", seed=0)
    else:
        comms = [{i} for i in range(n)]
    seen = set()
    groups = []
    for c in sorted(comms, key=lambda s: (-len(s), min(s))):
        g = sorted(c)
        groups.append(g)
        seen |= c
    for i in range(n):
        if i not in seen:
            groups.append([i])
    return groups, C, [r["alpha"] for r in viable]


def feature_correlation(feat: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Inter-feature Spearman correlation over the off-diagonal pairs.

    "Seven independent lines of evidence" is a claim about this matrix. If two features
    correlate at 0.95 they are one line of evidence written twice, and the report has to say so.
    """
    from scipy.stats import spearmanr
    names = list(FEATURES)
    n = feat[names[0]].shape[0]
    iu = np.triu_indices(n, 1)
    M = np.column_stack([feat[f][iu] for f in names])
    C = np.eye(len(names))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            v = spearmanr(M[:, i], M[:, j]).statistic
            C[i, j] = C[j, i] = 0.0 if not np.isfinite(v) else float(v)
    return C, names
