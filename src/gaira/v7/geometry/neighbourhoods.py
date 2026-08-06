"""GAIRA V7 — Phase 02.5: neighbourhood discovery, confounding, and Phase 03 priors.

A neighbourhood is not a merge. Phase 02 asked which motifs are interchangeable and found one
pair; this module asks which motifs are *related*, which is a much larger and much softer
question. Every neighbourhood is classified by what kind of relationship it is — exact
equivalence, shared substructure, superfamily similarity, generic Raman overlap, or artefact —
because those five demand very different treatment in Phase 03.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12
RELATIONSHIP_TIERS = ("exact_equivalence", "shared_substructure", "broad_superfamily",
                      "generic_raman_overlap", "artefact_or_unresolved")
GEOMETRY_TYPES = ("discrete", "continuous", "branching", "overlapping", "unresolved")


def nearest_neighbour_cards(Dm: np.ndarray, ids: list[str], meta: list[dict],
                            bands: list[list[float]], k: int = 5) -> pd.DataFrame:
    """Each motif's k nearest neighbours with everything needed to judge the relationship."""
    rows = []
    for i, m in enumerate(ids):
        nb = np.argsort(Dm[i])[1:k + 1]
        for rank, j in enumerate(nb, 1):
            shared = sorted(set(meta[i]["analytes"]) & set(meta[j]["analytes"]))
            bi, bj = np.asarray(bands[i]), np.asarray(bands[j])
            matched = ([float(b) for b in bi if np.min(np.abs(bj - b)) <= 10.0]
                       if bi.size and bj.size else [])
            rows.append({
                "motif": m, "rank": rank, "neighbour": ids[j],
                "distance": float(Dm[i, j]),
                "similarity": float(1.0 - Dm[i, j]),
                "motif_class": meta[i]["chemical_class"],
                "neighbour_class": meta[j]["chemical_class"],
                "cross_class": meta[i]["chemical_class"] != meta[j]["chemical_class"],
                "shared_bands_cm1": ";".join(f"{b:.0f}" for b in matched),
                "n_shared_bands": len(matched),
                "shared_molecules": ";".join(shared),
                "n_shared_molecules": len(shared),
            })
    return pd.DataFrame(rows)


def classify_relationship(row, merged_pairs: set, edge_weight: float,
                          null_p: float, band_overlap: float) -> str:
    """Assign one of the five tiers, on stated evidence rather than impression."""
    pair = frozenset((row["motif"], row["neighbour"]))
    if pair in merged_pairs:
        return "exact_equivalence"
    if null_p > 0.05:
        return "generic_raman_overlap"
    if row["n_shared_bands"] >= 3 and band_overlap >= 0.5:
        return "shared_substructure"
    if edge_weight >= 0.4:
        return "broad_superfamily"
    return "artefact_or_unresolved"


# ── source and excitation confounding ────────────────────────────────────────
def permanova(Dm: np.ndarray, labels: list[str], n_perm: int = 999,
              seed: int = 0) -> dict:
    """PERMANOVA (Anderson 2001) — is between-group distance larger than within?

    Non-parametric and distance-based, so it applies to any of the metrics here without
    assuming multivariate normality, which none of these representations satisfy.
    """
    lab = np.asarray(labels)
    n = len(lab)
    groups = np.unique(lab)
    if len(groups) < 2:
        return {"F": float("nan"), "p": float("nan"), "n_groups": len(groups),
                "note": "single group — not testable"}
    D2 = np.asarray(Dm, float) ** 2

    def ss(l):
        sst = D2[np.triu_indices(n, 1)].sum() / n
        ssw = 0.0
        for g in np.unique(l):
            idx = np.where(l == g)[0]
            if idx.size < 2:
                continue
            ssw += D2[np.ix_(idx, idx)][np.triu_indices(idx.size, 1)].sum() / idx.size
        return sst - ssw, ssw

    ssa, ssw = ss(lab)
    a, b = len(groups) - 1, n - len(groups)
    F = (ssa / a) / (ssw / b + EPS)
    rng = np.random.default_rng(seed)
    draws = np.array([(lambda s: (s[0] / a) / (s[1] / b + EPS))(ss(rng.permutation(lab)))
                      for _ in range(n_perm)])
    return {"F": float(F), "p": float(((draws >= F).sum() + 1) / (n_perm + 1)),
            "n_groups": int(len(groups)),
            "R2": float(ssa / (ssa + ssw + EPS))}


def knn_label_predictability(Dm: np.ndarray, labels: list[str], k: int = 5) -> dict:
    """Can a motif's label be predicted from its neighbours? Compared against the chance rate.

    This is the operational question for confounding: if source can be read off the geometry,
    the geometry is partly a map of instruments.
    """
    lab = np.asarray(labels)
    n = len(lab)
    hits = [float((lab[np.argsort(Dm[i])[1:k + 1]] == lab[i]).mean()) for i in range(n)]
    _, c = np.unique(lab, return_counts=True)
    p = c / c.sum()
    chance = float((p ** 2).sum())
    return {"knn_accuracy": float(np.mean(hits)), "chance": chance,
            "excess_over_chance": float(np.mean(hits) - chance)}


def leave_one_out_geometry(build_fn, groups: dict[str, list[int]], ids: list[str],
                           base_D: np.ndarray, k: int = 5) -> pd.DataFrame:
    """Rebuild the geometry with each source/excitation group removed and compare neighbours."""
    rows = []
    n = len(ids)
    base_nb = [set(np.argsort(base_D[i])[1:k + 1]) for i in range(n)]
    for g, drop in groups.items():
        keep = [i for i in range(n) if i not in set(drop)]
        if len(keep) < k + 2:
            rows.append({"held_out": g, "n_motifs_removed": len(drop),
                         "mean_knn_jaccard": float("nan"), "testable": False})
            continue
        Dk = build_fn(keep)
        js = []
        for a, i in enumerate(keep):
            nb = {keep[x] for x in np.argsort(Dk[a])[1:k + 1]}
            ref = base_nb[i] & set(keep)
            if ref:
                js.append(len(nb & ref) / len(nb | ref))
        rows.append({"held_out": g, "n_motifs_removed": len(drop),
                     "mean_knn_jaccard": float(np.mean(js)) if js else float("nan"),
                     "testable": True})
    return pd.DataFrame(rows)


def single_source_motifs(lsm_meta: list[dict], sources_of: dict) -> list[str]:
    """Motifs whose every supporting molecule comes from one dataset — untestable for source."""
    out = []
    for m in lsm_meta:
        srcs = {s for a in m["analytes"] if a for s in sources_of.get(a, [])}
        if len(srcs) <= 1:
            out.append(m["motif_id"])
    return out


# ── Phase 03 priors ──────────────────────────────────────────────────────────
def build_prior(prior_id: str, name: str, member_idx: list[int], ids: list[str],
                csm_of: dict, meta: list[dict], H: np.ndarray, grid: np.ndarray,
                geometry_type: str, evidence: dict, must_not_merge: list[list[str]],
                notes: str) -> dict:
    """One provisional Phase 03 prior.

    A prior constrains, it does not decide. `must_not_hard_merge` is the important field: it
    carries forward every distinction Phase 02 spent its falsification budget establishing, so
    Phase 03 cannot quietly undo them by assigning two motifs to one theme with membership 1.0.
    """
    from gaira.v7.csm.csm import dominant_bands
    members = [ids[i] for i in member_idx]
    shared = _shared_bands(H[member_idx], grid)
    return {
        "prior_id": prior_id,
        "provisional_name": name,
        "supporting_lsms": members,
        "supporting_csms": sorted({csm_of[m] for m in members if m in csm_of}),
        "geometry_type": geometry_type,
        "shared_bands_cm1": shared,
        "contributing_classes": sorted({meta[i]["chemical_class"] for i in member_idx}),
        "n_molecules": len({a for i in member_idx for a in meta[i]["analytes"] if a}),
        "evidence_strength": evidence.get("tier", "moderate"),
        "evidence": evidence,
        "source_confounding": evidence.get("source_confounding", "not assessed"),
        "confidence": float(evidence.get("confidence", 0.5)),
        "must_not_hard_merge": must_not_merge,
        "notes": notes,
        "status": "PROVISIONAL — a prior for Phase 03, not a theme",
    }


def _shared_bands(Hm: np.ndarray, grid: np.ndarray, tol: float = 12.0) -> list[float]:
    """Bands present in a majority of a group's motifs."""
    from gaira.v7.csm.csm import dominant_bands
    all_b = [dominant_bands(h, grid) for h in Hm]
    flat = sorted(b for bl in all_b for b in bl)
    out, used = [], set()
    for b in flat:
        if any(abs(b - u) <= tol for u in used):
            continue
        support = sum(any(abs(b - x) <= tol for x in bl) for bl in all_b)
        if support > len(all_b) / 2:
            out.append(round(float(b), 0))
            used.add(b)
    return out
