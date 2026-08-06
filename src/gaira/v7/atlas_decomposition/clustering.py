"""GAIRA V7 — deterministic clustering and cut selection for motif discovery.

Everything here is deterministic: hierarchical linkage on a fixed distance matrix, a fixed
candidate range, and a fixed tie-breaking rule. No k-means, no random initialisation, no
RNG anywhere on the discovery path. The only randomness in Phase 01 lives in the
*evaluation* permutation tests, which are seeded and never feed back into discovery.

PRE-REGISTERED SELECTION RULES
------------------------------
These were fixed before the final pipeline was run and are published with their comparison
tables whichever way they come out (`plan/VALIDATION_AND_DECISION_RULES.md` P-12).

1. **Linkage.** Compare `average`, `ward`, `complete`. Select the linkage whose selected cut
   is **most balanced (lowest size Gini) among those within 0.05 mean silhouette of the
   best**. Rationale: silhouette differences of a few hundredths are not meaningful, whereas
   a motif set in which one motif absorbs most participating analytes has not decomposed the
   component — it has peeled off outliers. Balance is what "preserves interpretability"
   means operationally here.

2. **Cut.** Within a component, choose `n_motifs ∈ [2, min(9, n_participants)]` maximising
   the silhouette on the cosine distance matrix. **`n_motifs = 1` (irreducible) is an
   admissible answer** and is reached whenever fewer than two motifs survive quality
   rejection.

3. **Chemistry is the EVALUATION, never the selection criterion.** No chemical class label
   enters band definition, profile construction, linkage choice or cut choice. If class
   labels chose the cut, the finding "motifs align with chemistry" would be circular.
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score

LINKAGES = ("average", "ward", "complete")
DEFAULT_LINKAGE = "ward"
MAX_MOTIFS = 9
SILHOUETTE_TOLERANCE = 0.05


def profile_distance(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cosine distance over band profiles. Returns (condensed, square)."""
    D = pdist(np.asarray(Q, float), metric="cosine")
    D = np.nan_to_num(D, nan=1.0, posinf=1.0, neginf=0.0)
    return D, squareform(D)


def build_linkage(Q: np.ndarray, method: str) -> np.ndarray:
    """Ward needs the observation matrix; the others take the condensed distance."""
    Q = np.asarray(Q, float)
    if method == "ward":
        return linkage(Q, method="ward")
    D, _ = profile_distance(Q)
    return linkage(D, method=method)


def size_gini(sizes) -> float:
    """0 = perfectly balanced motif sizes, → 1 = one motif absorbs everything."""
    x = np.sort(np.asarray(sizes, float))
    n = len(x)
    if n == 0 or x.sum() <= 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def select_cut(Q: np.ndarray, method: str = DEFAULT_LINKAGE,
               max_motifs: int = MAX_MOTIFS) -> dict:
    """Deterministically choose the number of motifs for one component.

    Returns the selected labelling plus the full sweep, so the choice is auditable.
    """
    Q = np.asarray(Q, float)
    n = Q.shape[0]
    _, Dm = profile_distance(Q)
    Z = build_linkage(Q, method)

    sweep, best = [], None
    # n_motifs must stay strictly below n: a cut that gives every analyte its own motif is
    # not a decomposition, and the silhouette is undefined there.
    for nc in range(2, min(max_motifs, n - 1) + 1):
        labels = fcluster(Z, nc, criterion="maxclust")
        if len(set(labels)) < 2 or len(set(labels)) >= n:
            continue
        sil = float(silhouette_score(Dm, labels, metric="precomputed"))
        sizes = np.bincount(labels)[1:]
        sizes = sizes[sizes > 0]
        row = {"n_motifs": int(nc), "silhouette": round(sil, 6),
               "size_gini": round(size_gini(sizes), 6),
               "max_motif_share": round(float(sizes.max() / sizes.sum()), 6),
               "sizes": sorted(sizes.tolist(), reverse=True)}
        sweep.append(row)
        # strict > with a fixed epsilon: the smallest n_motifs wins ties, deterministically
        if best is None or sil > best["silhouette"] + 1e-12:
            best = row | {"labels": labels}

    if best is None:
        return {"n_motifs": 1, "labels": np.ones(n, dtype=int), "silhouette": float("nan"),
                "size_gini": 0.0, "max_motif_share": 1.0, "sweep": sweep,
                "linkage": method, "irreducible_reason": "no admissible cut"}
    out = dict(best)
    out["sweep"] = sweep
    out["linkage"] = method
    return out


def compare_linkages(profiles: dict[int, np.ndarray],
                     max_motifs: int = MAX_MOTIFS) -> list[dict]:
    """Run the pre-registered linkage comparison across all analysable components."""
    rows = []
    for method in LINKAGES:
        sil, gini, share, nmot = [], [], [], 0
        for _, Q in sorted(profiles.items()):
            sel = select_cut(Q, method=method, max_motifs=max_motifs)
            if not np.isfinite(sel["silhouette"]):
                continue
            sil.append(sel["silhouette"])
            gini.append(sel["size_gini"])
            share.append(sel["max_motif_share"])
            nmot += sel["n_motifs"]
        rows.append({
            "linkage": method,
            "n_components": len(sil),
            "mean_silhouette": round(float(np.mean(sil)), 4),
            "mean_size_gini": round(float(np.mean(gini)), 4),
            "mean_max_motif_share": round(float(np.mean(share)), 4),
            "total_motifs_before_rejection": int(nmot),
        })
    return rows


def apply_linkage_rule(rows: list[dict], tolerance: float = SILHOUETTE_TOLERANCE) -> dict:
    """The pre-registered rule: lowest Gini among linkages within `tolerance` of best silhouette."""
    best_sil = max(r["mean_silhouette"] for r in rows)
    admissible = [r for r in rows if r["mean_silhouette"] >= best_sil - tolerance]
    chosen = min(admissible, key=lambda r: (r["mean_size_gini"], r["linkage"]))
    return {
        "selected_linkage": chosen["linkage"],
        "rule": (f"lowest mean size Gini among linkages within {tolerance} mean silhouette "
                 f"of the best ({best_sil:.4f})"),
        "admissible": [r["linkage"] for r in admissible],
        "selected_mean_silhouette": chosen["mean_silhouette"],
        "selected_mean_size_gini": chosen["mean_size_gini"],
    }


def jackknife_stability(Q: np.ndarray, labels: np.ndarray, method: str = DEFAULT_LINKAGE,
                        n_motifs: int | None = None) -> np.ndarray:
    """Per-motif stability by deterministic leave-one-analyte-out re-clustering.

    For each motif, the fraction of its member pairs that remain co-assigned when any single
    participating analyte is removed and the component is re-clustered at the same cut.
    Deterministic by construction — a jackknife, not a bootstrap, precisely so no RNG enters
    discovery.
    """
    Q = np.asarray(Q, float)
    n = Q.shape[0]
    labels = np.asarray(labels)
    k = int(n_motifs or len(set(labels)))
    uniq = sorted(set(labels.tolist()))
    if k < 2 or n <= 3:
        return np.ones(len(uniq), float)

    agree = {u: [0, 0] for u in uniq}                       # [kept_together, total_pairs]
    for drop in range(n):
        keep = np.setdiff1d(np.arange(n), [drop])
        Qk = Q[keep]
        if Qk.shape[0] < k:
            continue
        Zk = build_linkage(Qk, method)
        lk = fcluster(Zk, k, criterion="maxclust")
        pos = {orig: i for i, orig in enumerate(keep)}
        for u in uniq:
            members = [i for i in range(n) if labels[i] == u and i != drop]
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    agree[u][1] += 1
                    if lk[pos[members[a]]] == lk[pos[members[b]]]:
                        agree[u][0] += 1
    return np.array([(agree[u][0] / agree[u][1]) if agree[u][1] else 1.0 for u in uniq], float)
