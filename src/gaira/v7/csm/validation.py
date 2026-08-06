"""GAIRA V7 — Phase 02: the seven required validations.

Each function is written to look for a reason the CSM layer is wrong, not for confirmation
that it is right. A validation that can only pass is not a validation.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls
from sklearn.metrics import adjusted_rand_score

EPS = 1e-12
EV_DEGRADE_MAX = 0.05
NULL_PERMUTATIONS = 200


def _ev(X: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Per-row explained variance of a non-negative projection onto dictionary D."""
    out = np.zeros(X.shape[0])
    for i, x in enumerate(X):
        c = nnls(D.T, x)[0]
        out[i] = max(0.0, 1.0 - ((x - c @ D) ** 2).sum() / ((x ** 2).sum() + EPS))
    return out


# ── 1. reconstruction: CSMs vs the LSMs they replace ─────────────────────────
def reconstruction_comparison(X: np.ndarray, canonical_id: np.ndarray,
                              mol_class: dict[str, str],
                              H_lsm: np.ndarray, D_csm: np.ndarray) -> "pd.DataFrame":
    """Per-molecule explained variance under both dictionaries.

    Per molecule, not averaged: a class-average concealed urea at 0.123 and thymine at 0.130
    in Phase 01, and the same averaging would conceal the same thing here.
    """
    import pandas as pd
    ev_l, ev_c = _ev(X, H_lsm), _ev(X, D_csm)
    rows = []
    for cid in sorted(set(canonical_id)):
        m = canonical_id == cid
        rows.append({"canonical_id": cid, "chemistry_class": mol_class.get(cid, ""),
                     "n_spectra": int(m.sum()),
                     "ev_lsm": float(ev_l[m].mean()), "ev_csm": float(ev_c[m].mean()),
                     "delta": float(ev_c[m].mean() - ev_l[m].mean())})
    df = pd.DataFrame(rows).sort_values("delta")
    df["degraded_beyond_tolerance"] = df["delta"] < -EV_DEGRADE_MAX
    return df


def isolated_merge_cost(X: np.ndarray, canonical_id: np.ndarray, H_lsm: np.ndarray,
                        members: list[int], molecules: list[str],
                        consensus: np.ndarray) -> float:
    """Explained-variance cost of ONE merge, with every other motif left alone.

    The dictionary is the full LSM set with this group's members replaced by their single
    consensus spectrum — nothing else changes. Measuring each merge against the fully merged
    basis instead would charge every group for its neighbours' losses, and a good merge sitting
    next to a bad one would be rejected for the bad one's damage.
    """
    if len(members) < 2:
        return 0.0
    keep = [i for i in range(H_lsm.shape[0]) if i not in members]
    D_merged = np.vstack([H_lsm[keep], consensus[None, :]])
    rows = np.isin(canonical_id, list(molecules))
    if not rows.any():
        return 0.0
    return float(_ev(X[rows], D_merged).mean() - _ev(X[rows], H_lsm).mean())


# ── 2. bootstrap stability of the grouping itself ────────────────────────────
def bootstrap_partition_stability(W: np.ndarray, motif_ids: list[str], classes: list[str],
                                  types: list[str], tau: float, base_groups: list[list[int]],
                                  n_boot: int = 50, seed: int = 0) -> dict:
    """Resample the *edges* the graph was built from and re-derive communities.

    Reports both the global agreement (adjusted Rand index) and, per CSM, the fraction of
    resamples in which its members stayed together — which is the number that belongs on an
    individual CSM, since a good global ARI can hide one unstable group.
    """
    from .graph import build_graph, consensus_partition

    n = W.shape[0]
    rng = np.random.default_rng(seed)
    base = np.zeros(n, int)
    for k, g in enumerate(base_groups):
        base[g] = k

    # Perturb only edges that exist. An earlier version added Gaussian noise to the whole
    # matrix, which on a sparsified graph (53 non-zero pairs of 1225) invents ~1200 spurious
    # edges every repeat and measures how well a partition survives being replaced by a random
    # graph — it reported ARI 0.100 for a structure that leave-one-class-out reproduced at
    # 1.000. Perturbing existing evidence is the question; inventing evidence is not.
    mask = W > 0
    aris, together = [], np.zeros((n, n))
    for _ in range(n_boot):
        noise = rng.normal(0.0, 0.05, size=(n, n))
        noise = (noise + noise.T) / 2
        Wb = np.where(mask, np.clip(W + noise, 0.0, 1.0), 0.0)
        np.fill_diagonal(Wb, 0.0)
        G = build_graph(Wb, motif_ids, classes, types, tau)
        part = consensus_partition(G, seeds=4)
        lab = np.array([part[m] for m in motif_ids])
        aris.append(adjusted_rand_score(base, lab))
        together += (lab[:, None] == lab[None, :])
    together /= n_boot

    per_csm = []
    for k, g in enumerate(base_groups):
        if len(g) == 1:
            per_csm.append(1.0)
        else:
            per_csm.append(float(np.mean([together[a, b]
                                          for i, a in enumerate(g) for b in g[i + 1:]])))
    return {"mean_ari": float(np.mean(aris)), "min_ari": float(np.min(aris)),
            "per_csm_confidence": per_csm, "cooccurrence": together}


# ── 3. leave-one-class-out ───────────────────────────────────────────────────
def leave_one_class_out(W: np.ndarray, motif_ids: list[str], classes: list[str],
                        types: list[str], tau: float,
                        base_groups: list[list[int]]) -> "pd.DataFrame":
    """Drop each chemical class in turn and re-derive the communities of what remains.

    A CSM that only exists while one particular class is present is a description of that
    class, not a consensus across classes.
    """
    import pandas as pd
    from .graph import build_graph, consensus_partition

    n = W.shape[0]
    base = np.zeros(n, int)
    for k, g in enumerate(base_groups):
        base[g] = k

    rows = []
    for c in sorted(set(classes)):
        keep = [i for i in range(n) if classes[i] != c]
        if len(keep) < 3:
            continue
        Wk = W[np.ix_(keep, keep)]
        ids = [motif_ids[i] for i in keep]
        G = build_graph(Wk, ids, [classes[i] for i in keep], [types[i] for i in keep], tau)
        part = consensus_partition(G, seeds=4)
        lab = np.array([part[m] for m in ids])
        rows.append({"held_out_class": c, "n_lsms_removed": n - len(keep),
                     "ari_vs_base": float(adjusted_rand_score(base[keep], lab)),
                     "n_communities": int(len(set(lab)))})
    return pd.DataFrame(rows)


# ── 4. source robustness ─────────────────────────────────────────────────────
def source_robustness(groups: list[list[int]], lsm_meta: list[dict],
                      lsm_sources: list[dict[str, float]]) -> "pd.DataFrame":
    """Is a merge explained by chemistry, or by two motifs sharing one measurement source?

    A cross-class group whose members are all dominated by one dataset is a candidate
    instrument artefact, and it is flagged whether or not every other channel agrees.
    """
    import pandas as pd
    rows = []
    for k, g in enumerate(groups):
        srcs = [max(lsm_sources[i], key=lsm_sources[i].get) if lsm_sources[i] else ""
                for i in g]
        frac = [max(lsm_sources[i].values()) if lsm_sources[i] else 0.0 for i in g]
        shared = len(set(srcs)) == 1 and len(g) > 1
        classes = {lsm_meta[i]["chemical_class"] for i in g}
        rows.append({
            "csm_index": k, "n_lsms": len(g), "n_classes": len(classes),
            "dominant_sources": ";".join(sorted(set(srcs))),
            "single_shared_source": shared,
            "mean_source_dominance": float(np.mean(frac)) if frac else 0.0,
            "source_confound_risk": bool(shared and len(classes) > 1
                                         and np.mean(frac) > 0.8),
        })
    return pd.DataFrame(rows)


# ── 5. spectroscopic interpretability — class-conditioned assignment ─────────
GENERIC_BANDS = [
    (480, 560, "S–S / skeletal deformation"),
    (620, 650, "C–S stretch / phenyl ring"),
    (700, 730, "ring breathing"),
    (750, 770, "ring breathing (indole / pyrrole)"),
    (820, 860, "tyrosine Fermi doublet / C–C skeletal"),
    (870, 900, "C–C–N / C–O–C skeletal"),
    (930, 960, "C–C backbone stretch"),
    (1000, 1010, "phenylalanine ring breathing"),
    (1030, 1060, "C–N / C–O stretch"),
    (1060, 1100, "C–C skeletal (trans) / PO2- symmetric"),
    (1120, 1145, "C–C skeletal (gauche) / C–N stretch"),
    (1150, 1180, "C–C / C=C conjugated stretch"),
    (1200, 1240, "amide III / C–O–C asymmetric"),
    (1240, 1280, "amide III / =C–H in-plane bend"),
    (1290, 1320, "CH2 twist"),
    (1330, 1360, "CH deformation / tryptophan"),
    (1390, 1420, "COO- symmetric stretch"),
    (1430, 1465, "CH2 / CH3 scissoring"),
    (1500, 1530, "conjugated C=C (carotenoid)"),
    (1540, 1580, "amide II / COO- asymmetric"),
    (1600, 1620, "aromatic ring C=C"),
    (1630, 1690, "amide I / C=C stretch"),
    (1700, 1760, "C=O ester / carboxylic acid"),
]
CLASS_SPECIFIC = {
    "purine": [(700, 740, "purine ring breathing"), (1320, 1350, "purine N7–C5 stretch")],
    "pyrimidine": [(770, 800, "pyrimidine ring breathing"), (1650, 1680, "C=O / C=N")],
    "sterol_steroid": [(690, 720, "sterol ring skeletal"), (1660, 1680, "C=C sterol")],
    "fatty_acid": [(1060, 1080, "C–C trans chain"), (1650, 1670, "cis C=C"),
                   (1700, 1720, "carboxylic acid C=O")],
    "acylglycerol": [(1730, 1750, "ester C=O"), (860, 880, "glycerol C–C")],
    "phospholipid_sphingolipid": [(1080, 1100, "PO2- symmetric stretch"),
                                  (1730, 1745, "ester C=O")],
    "peptide_protein": [(1650, 1680, "amide I"), (1230, 1280, "amide III"),
                        (1000, 1010, "phenylalanine")],
    "free_amino_acid": [(1390, 1420, "COO- symmetric"), (2900, 3000, "CH stretch")],
    "mono_oligosaccharide": [(1080, 1130, "C–O–C glycosidic"), (840, 900, "anomeric C–H")],
    "polysaccharide": [(1080, 1130, "C–O–C glycosidic"), (930, 960, "C–O–C ring")],
    "nucleic_acid_polymer": [(780, 800, "backbone O–P–O"), (1080, 1100, "PO2- symmetric")],
    "chromophore_pigment": [(1150, 1170, "C–C conjugated"), (1510, 1530, "C=C conjugated")],
    "sulfur_thiol_cofactor": [(490, 550, "S–S stretch"), (630, 680, "C–S stretch")],
    "carboxylic_acid_metabolite": [(1390, 1420, "COO- symmetric"),
                                   (1700, 1730, "carboxylic acid C=O")],
    "phosphate_metabolite": [(970, 990, "PO4 symmetric stretch"),
                             (1080, 1100, "PO2- symmetric")],
    "small_nitrogenous": [(1000, 1020, "C–N symmetric"), (1590, 1620, "NH2 scissor")],
    "polyol": [(830, 890, "C–C–O"), (1050, 1090, "C–O stretch")],
}


def assign_band(cm: float, classes: list[str]) -> str:
    """Assign a band conditioned on the chemistry that produced it.

    A context-free table calls 702 cm-1 "purine ring breathing" inside a sterol motif, where
    it is the cholesterol ring mode — the Phase 01 investigation caught exactly that error, and
    a CSM spanning several classes makes it easier to make, not harder.
    """
    hits = []
    for c in classes:
        for lo, hi, label in CLASS_SPECIFIC.get(c, []):
            if lo <= cm <= hi:
                hits.append(f"{label} [{c}]")
    if hits:
        return " / ".join(dict.fromkeys(hits))
    for lo, hi, label in GENERIC_BANDS:
        if lo <= cm <= hi:
            return label
    return "unassigned"


# ── 6. cross-CSM redundancy ──────────────────────────────────────────────────
def redundancy_audit(D: np.ndarray, ids: list[str], threshold: float = 0.90) -> "pd.DataFrame":
    """Pairs of CSMs that are near-duplicates of one another.

    Precedent from V5: the motif layer shipped porphyrin<->flavin at 0.699 and
    carboxylate<->colloid_matrix at 0.687 support cosine. Both should have been caught here.
    """
    import pandas as pd
    N = D / (np.linalg.norm(D, axis=1, keepdims=True) + EPS)
    C = N @ N.T
    rows = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            rows.append({"csm_a": ids[i], "csm_b": ids[j], "cosine": float(C[i, j]),
                         "redundant": bool(C[i, j] >= threshold)})
    return pd.DataFrame(rows).sort_values("cosine", ascending=False)


# ── 7. false-merge investigation ─────────────────────────────────────────────
def band_permutation_null(H: np.ndarray, grid: np.ndarray, feat_fn,
                          n_perm: int = NULL_PERMUTATIONS, seed: int = 0) -> np.ndarray:
    """Edge weights under band-position permutation, preserving band count and intensity.

    Without this the observed weights have no scale. An edge of 0.6 means nothing until we
    know what 0.6 looks like when the chemistry has been destroyed but the spectral statistics
    have not.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_perm):
        Hp = np.array([np.roll(h, int(rng.integers(30, H.shape[1] - 30))) for h in H])
        out.append(feat_fn(Hp))
    return np.array(out)


def pair_dossier(i: int, j: int, feat: dict[str, np.ndarray], W: np.ndarray,
                 lsm_meta: list[dict]) -> dict:
    """Everything known about one candidate merge, for the named-suspect investigation."""
    d = {"lsm_a": lsm_meta[i]["motif_id"], "lsm_b": lsm_meta[j]["motif_id"],
         "class_a": lsm_meta[i]["chemical_class"], "class_b": lsm_meta[j]["chemical_class"],
         "type_a": lsm_meta[i]["lsm_type"], "type_b": lsm_meta[j]["lsm_type"],
         "edge_weight": float(W[i, j])}
    for f, Mx in feat.items():
        d[f] = float(Mx[i, j])
    return d
