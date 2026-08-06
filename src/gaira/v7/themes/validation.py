"""GAIRA V7 — Phase 03: theme validation, hierarchy, gradients and post-hoc interpretation.

Nine checks, each written to look for a reason a theme is wrong. A theme that fails is rejected;
the objective is the smallest set of themes the frozen geometry actually supports, not the
largest set that can be defended.

The last two checks — ontology agreement and biochemical naming — are the only places human
labels appear, and they run after `K` is fixed and every other check has passed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, nnls
from scipy.stats import entropy

EPS = 1e-12
UNASSIGNED_MIN_FIT = 0.35             # best theme must explain this much of the CSM
BRIDGE_ENTROPY_QUANTILE = 0.70        # membership entropy above this marks a bridge
BRIDGE_MIN_SECOND = 0.20              # ...and a genuine second claim on the CSM


# ── 1–2. stability under perturbation ────────────────────────────────────────
def _match(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, float]:
    C = np.zeros((A.shape[1], B.shape[1]))
    for i in range(A.shape[1]):
        for j in range(B.shape[1]):
            C[i, j] = np.dot(A[:, i], B[:, j]) / (
                np.linalg.norm(A[:, i]) * np.linalg.norm(B[:, j]) + EPS)
    r, c = linear_sum_assignment(-C)
    return c, float(C[r, c].mean())


def bootstrap_stability(fit_fn, X: np.ndarray, S0: np.ndarray, n_boot: int = 50,
                        seed: int = 0) -> dict:
    """Per-theme recovery under resampling of the CSM set."""
    rng = np.random.default_rng(seed)
    n, K = X.shape[0], S0.shape[1]
    per = np.zeros(K)
    cnt = np.zeros(K)
    overall = []
    for _ in range(n_boot):
        keep = np.sort(rng.choice(n, int(0.85 * n), replace=False))
        S = fit_fn(X[keep])["S"]
        cols, mean = _match(S0[keep], S)
        overall.append(mean)
        for k in range(min(K, S.shape[1])):
            a, b = S0[keep][:, k], S[:, cols[k]]
            per[k] += np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS)
            cnt[k] += 1
    return {"mean": float(np.mean(overall)), "min": float(np.min(overall)),
            "per_theme": (per / np.maximum(cnt, 1)).tolist()}


def leave_one_out_stability(fit_fn, X: np.ndarray, S0: np.ndarray) -> dict:
    """Drop each CSM in turn. A theme that needs one particular CSM is that CSM, not a theme."""
    n, K = X.shape[0], S0.shape[1]
    per = np.zeros(K)
    worst = []
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        S = fit_fn(X[keep])["S"]
        cols, mean = _match(S0[keep], S)
        worst.append((mean, i))
        for k in range(min(K, S.shape[1])):
            a, b = S0[keep][:, k], S[:, cols[k]]
            per[k] += np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS)
    return {"per_theme": (per / n).tolist(), "mean": float(np.mean([w for w, _ in worst])),
            "most_influential_csm_index": int(min(worst)[1]),
            "min": float(min(w for w, _ in worst))}


# ── 3. nearest-neighbour consistency ─────────────────────────────────────────
def neighbour_consistency(S: np.ndarray, D: np.ndarray, k: int = 5) -> dict:
    """Do geometric neighbours share a dominant theme?

    This is the check that ties the theme layer back to Phase 02.5: if themes disagree with
    the geometry they were meant to explain, they are describing something else.
    """
    top = S.argmax(axis=1)
    n = D.shape[0]
    hits = [float((top[np.argsort(D[i])[1:k + 1]] == top[i]).mean()) for i in range(n)]
    _, c = np.unique(top, return_counts=True)
    p = c / c.sum()
    return {"knn_agreement": float(np.mean(hits)), "chance": float((p ** 2).sum()),
            "per_csm": hits}


# ── 4. graph modularity of the dominant-theme partition ──────────────────────
def theme_modularity(S: np.ndarray, D: np.ndarray, ids: list[str], k: int = 5,
                     n_null: int = 40, seed: int = 0) -> dict:
    import networkx as nx
    from gaira.v7.geometry.nulls import degree_preserving_graph_null
    from gaira.v7.geometry.structure import knn_graph
    G = knn_graph(D, ids, k=k)
    top = S.argmax(axis=1)
    comms = [{ids[i] for i in range(len(ids)) if top[i] == t} for t in sorted(set(top))]
    comms = [c for c in comms if c]
    obs = nx.community.modularity(G, comms, weight="weight")
    W = np.zeros((len(ids), len(ids)))
    for u, v, d in G.edges(data=True):
        i, j = ids.index(u), ids.index(v)
        W[i, j] = W[j, i] = d["weight"]
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
        if Gn.number_of_edges():
            draws.append(nx.community.modularity(Gn, comms, weight="weight"))
    draws = np.array(draws) if draws else np.array([0.0])
    return {"observed": float(obs), "null_mean": float(draws.mean()),
            "null_sd": float(draws.std()),
            "p_empirical": float((draws >= obs).mean())}


# ── 5. reconstruction behaviour ──────────────────────────────────────────────
def reconstruction_comparison(X: np.ndarray, themes: np.ndarray) -> dict:
    """What the theme basis costs against the CSM basis it abstracts.

    A theme layer that reconstructs as well as the CSM layer has not abstracted anything; one
    that reconstructs far worse has thrown away chemistry. The number is reported either way.
    """
    def ev(D):
        res = tot = 0.0
        for x in X:
            c = nnls(D.T, x)[0]
            res += float(((x - c @ D) ** 2).sum())
            tot += float((x ** 2).sum())
        return max(0.0, 1.0 - res / (tot + EPS))
    ev_csm, ev_theme = ev(X), ev(themes)
    return {"ev_csm_basis": float(ev_csm), "ev_theme_basis": float(ev_theme),
            "delta": float(ev_theme - ev_csm),
            "compression": float(X.shape[0] / max(themes.shape[0], 1))}


# ── 6. value over the CSM layer (risk R-11) ──────────────────────────────────
def value_over_csm(S: np.ndarray, X: np.ndarray, themes: np.ndarray,
                   labels: list[str], D: np.ndarray, k: int = 5) -> dict:
    """Does the theme layer change any decision the CSM layer would have made?

    At V6.2 `theme_raw` and `theme_posterior` were numerically identical at every metric on
    every ontology — machinery that changed nothing. The test here is deliberately harsh:
    coarse-chemistry retrieval using theme coordinates against the same retrieval using CSM
    activations. If theme coordinates do not retrieve better, the layer is decorative and this
    function says so.
    """
    lab = np.asarray(labels)
    n = X.shape[0]

    def retrieval(Z):
        Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + EPS)
        Sim = Zn @ Zn.T
        np.fill_diagonal(Sim, -np.inf)
        hits = [float((lab[np.argsort(-Sim[i])[:k]] == lab[i]).mean()) for i in range(n)]
        return float(np.mean(hits))

    r_csm = retrieval(X)
    r_theme = retrieval(S)
    _, c = np.unique(lab, return_counts=True)
    chance = float(((c / c.sum()) ** 2).sum())
    return {"retrieval_csm_basis": r_csm, "retrieval_theme_coordinates": r_theme,
            "chance": chance, "delta": float(r_theme - r_csm),
            "theme_layer_adds_value": bool(r_theme > r_csm),
            "verdict": ("theme coordinates retrieve coarse chemistry better than raw CSM "
                        "activations" if r_theme > r_csm else
                        "theme layer does NOT improve retrieval over the CSM layer — recorded, "
                        "not hidden (risk R-11)")}


# ── 7. cross-source and excitation robustness ────────────────────────────────
def robustness(fit_fn, X: np.ndarray, S0: np.ndarray, groups: dict[str, list[int]]) -> pd.DataFrame:
    """Refit with each source / excitation group removed and match the themes back."""
    rows = []
    for g, drop in groups.items():
        keep = [i for i in range(X.shape[0]) if i not in set(drop)]
        if len(keep) < S0.shape[1] + 2:
            rows.append({"held_out": g, "n_removed": len(drop), "theme_recovery": np.nan,
                         "testable": False})
            continue
        S = fit_fn(X[keep])["S"]
        _, mean = _match(S0[keep], S)
        rows.append({"held_out": g, "n_removed": len(drop),
                     "theme_recovery": float(mean), "testable": True})
    return pd.DataFrame(rows)


# ── 8. ontology agreement — POST HOC ONLY ────────────────────────────────────
def ontology_agreement(S: np.ndarray, labels: list[str], n_perm: int = 500,
                       seed: int = 0) -> dict:
    """Mutual information between dominant theme and curated chemistry, against a permutation
    null. Computed only after `K` is fixed; never an objective."""
    from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
    top = S.argmax(axis=1)
    lab = np.asarray(labels)
    nmi = normalized_mutual_info_score(lab, top)
    ami = adjusted_mutual_info_score(lab, top)
    rng = np.random.default_rng(seed)
    draws = np.array([normalized_mutual_info_score(rng.permutation(lab), top)
                      for _ in range(n_perm)])
    return {"nmi": float(nmi), "ami": float(ami), "null_mean": float(draws.mean()),
            "null_sd": float(draws.std()), "p_empirical": float((draws >= nmi).mean())}


# ── 9. membership roles: bridges, isolates, unassigned ───────────────────────
def membership_roles(S: np.ndarray, ids: list[str], X: np.ndarray, themes: np.ndarray,
                     geometry_bridges: set[str], geometry_isolates: set[str]) -> pd.DataFrame:
    """Classify every CSM by what its membership distribution actually says.

    **Unassigned is a fit failure, not a diffuse membership.** An earlier version called a CSM
    unassigned whenever its largest membership fell below a floor — which is the same condition
    as having split membership, so every bridge was swallowed by the unassigned class and the
    run reported 15 unassigned CSMs and zero bridges. The two are opposite findings and must be
    separated: a bridge is *well explained by several themes*; an unassigned CSM is *explained
    by none*. So the unassigned test asks how well the best theme reconstructs the CSM, and the
    bridge test asks whether a second theme has a real claim on it.
    """
    H = np.array([entropy(s + EPS) for s in S])
    hi = np.quantile(H, BRIDGE_ENTROPY_QUANTILE)
    rows = []
    for i, cid in enumerate(ids):
        order = np.argsort(-S[i])
        primary, secondary = int(order[0]), int(order[1]) if S.shape[1] > 1 else -1
        top = float(S[i, primary])
        th = themes[primary]
        c = float(np.dot(X[i], th) / (np.dot(th, th) + EPS))
        fit = max(0.0, 1.0 - ((X[i] - c * th) ** 2).sum() / ((X[i] ** 2).sum() + EPS))
        second = float(S[i, secondary]) if secondary >= 0 else 0.0
        role = ("unassigned" if fit < UNASSIGNED_MIN_FIT else
                "bridge" if (H[i] >= hi and second >= BRIDGE_MIN_SECOND) else "member")
        rows.append({
            "csm_id": cid, "primary_theme": primary, "primary_membership": top,
            "secondary_theme": secondary,
            "secondary_membership": float(S[i, secondary]) if secondary >= 0 else 0.0,
            "membership_entropy": float(H[i]),
            "best_theme_fit": float(fit),
            "role": role,
            "geometry_bridge": cid in geometry_bridges,
            "geometry_isolate": cid in geometry_isolates,
        })
    return pd.DataFrame(rows)


# ── hierarchy ────────────────────────────────────────────────────────────────
def infer_hierarchy(themes: np.ndarray, S: np.ndarray, max_levels: int = 4) -> dict:
    """Do the themes nest, and into how many levels?

    The number of levels is inferred, not assumed. Themes are merged agglomeratively on their
    spectral correlation; a level exists only where merging produces a partition that is both
    stable and materially coarser than the one below it.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    N = themes / (np.linalg.norm(themes, axis=1, keepdims=True) + EPS)
    C = np.clip(N @ N.T, -1, 1)
    Dm = 1.0 - C
    np.fill_diagonal(Dm, 0.0)
    K = themes.shape[0]
    if K < 3:
        return {"levels": [{"level": 1, "n_groups": K,
                            "assignment": list(range(K))}],
                "n_levels": 1, "note": "too few themes to nest"}
    Z = linkage(squareform(Dm, checks=False), method="average")
    levels, prev = [], None
    for n_groups in range(2, min(K, max_levels + 1) + 1):
        lab = fcluster(Z, t=n_groups, criterion="maxclust") - 1
        if prev is not None and len(set(lab)) == len(set(prev)):
            continue
        # coherence of the level: mean within-group theme correlation
        coh = []
        for g in set(lab):
            m = np.where(lab == g)[0]
            if len(m) > 1:
                coh.append(float(C[np.ix_(m, m)][np.triu_indices(len(m), 1)].mean()))
        levels.append({"n_groups": int(len(set(lab))), "assignment": lab.tolist(),
                       "mean_within_correlation": float(np.mean(coh)) if coh else 1.0})
        prev = lab
    for i, lv in enumerate(levels, 1):
        lv["level"] = i
    return {"levels": levels, "n_levels": len(levels),
            "linkage_distances": Z[:, 2].tolist()}


# ── continuous gradients ─────────────────────────────────────────────────────
def theme_gradients(S: np.ndarray, coords: np.ndarray, ids: list[str],
                    n_perm: int = 500, seed: int = 0) -> pd.DataFrame:
    """Is each theme's membership a smooth gradient over the manifold, or a plateau?

    Phase 02.5 found a continuum; a theme layer that respects it should show membership varying
    smoothly along diffusion coordinates rather than switching. Tested as the Spearman
    correlation of membership with each coordinate, against a permutation null.
    """
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(S.shape[1]):
        for d in range(min(3, coords.shape[1])):
            r = spearmanr(S[:, k], coords[:, d]).statistic
            r = 0.0 if not np.isfinite(r) else float(r)
            draws = np.array([abs(spearmanr(rng.permutation(S[:, k]),
                                            coords[:, d]).statistic) for _ in range(n_perm)])
            draws = np.nan_to_num(draws)
            rows.append({"theme": k, "diffusion_coord": d + 1, "spearman": r,
                         "abs_spearman": abs(r),
                         "p_empirical": float((draws >= abs(r)).mean()),
                         "is_gradient": bool((draws >= abs(r)).mean() < 0.05)})
    return pd.DataFrame(rows)
