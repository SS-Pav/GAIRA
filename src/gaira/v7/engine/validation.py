"""GAIRA V7 — Phase 04, Parts G–I: held-out validation at six abstraction levels.

The central question is not whether the engine reconstructs spectra. It is whether the frozen
hierarchy supports **inference about a spectrum it has not seen**, and whether abstraction
helps or hurts as you climb it.

**Two kinds of leakage, and only one is avoidable by grouping.**

1. *Spectrum-level* — evaluating a spectrum against itself. Removed by grouped CV at the
   canonical-molecule level: every spectrum of a held-out molecule is withheld together.
2. *Dictionary-level* — the LSMs were fitted on all 154 molecules, so the dictionary has seen
   the held-out molecule even when the retrieval set has not. Grouping cannot remove this, and
   the phase is forbidden from refitting the frozen atlas.

The second is measured rather than assumed away, by `leakage_control`: the same evaluation run
against a dictionary refit **without** the held-out fold, in a scratch control that never
touches the frozen tree. The gap between the two is the inflation in every in-sample number
this project has produced, and it is the single most important measurement in Phase 04.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

EPS = 1e-12


# ── the two splits, and why one is not enough ────────────────────────────────
def leave_one_spectrum_out(y: np.ndarray) -> np.ndarray:
    """Split A — each spectrum is held out alone, its molecule still represented by replicates.

    Answers: *can the engine identify a KNOWN molecule from a new measurement?* Defined only
    for molecules with at least two spectra; singletons are excluded because a singleton's
    identity cannot be retrieved when its only spectrum is the query.
    """
    counts = {v: int((y == v).sum()) for v in set(y)}
    return np.array([counts[v] >= 2 for v in y])


def grouped_folds_note() -> str:
    """Split B — molecule-grouped folds, every spectrum of a molecule withheld together.

    Answers: *can the engine place an UNSEEN molecule in the right chemistry?* Molecule-level
    top-k is **undefined** under this split — the true molecule is absent from the reference
    set by construction, so top-1 is exactly zero however good the engine is. Reporting it
    would be reporting the split, not the engine. Class-level retrieval is the metric here.
    """
    return ("molecule-grouped: molecule top-k is undefined by construction; class retrieval "
            "is the metric")


# ── Level 1: canonical molecule retrieval ────────────────────────────────────
def molecule_retrieval(A_query: np.ndarray, A_ref: np.ndarray, y_query: np.ndarray,
                       y_ref: np.ndarray, ks=(1, 3, 5)) -> dict:
    """Nearest-neighbour retrieval in an activation space, held-out against reference.

    Cosine similarity, because activation magnitude tracks acquisition conditions while its
    direction tracks chemistry.
    """
    Q = A_query / (np.linalg.norm(A_query, axis=1, keepdims=True) + EPS)
    R = A_ref / (np.linalg.norm(A_ref, axis=1, keepdims=True) + EPS)
    Sim = Q @ R.T
    order = np.argsort(-Sim, axis=1)
    hits = {f"top{k}": [] for k in ks}
    rr = []
    pred = []
    for i in range(Q.shape[0]):
        ranked = y_ref[order[i]]
        pred.append(ranked[0])
        for k in ks:
            hits[f"top{k}"].append(float(y_query[i] in set(ranked[:k])))
        w = np.where(ranked == y_query[i])[0]
        rr.append(1.0 / (w[0] + 1) if w.size else 0.0)
    return {**{k: float(np.mean(v)) for k, v in hits.items()},
            "mrr": float(np.mean(rr)), "n": int(Q.shape[0]),
            "predictions": np.array(pred)}


def per_class_retrieval(A_query, A_ref, y_query, y_ref, cls_query, ks=(1, 3, 5)) -> pd.DataFrame:
    rows = []
    for c in sorted(set(cls_query)):
        m = np.array([x == c for x in cls_query])
        if m.sum() == 0:
            continue
        r = molecule_retrieval(A_query[m], A_ref, y_query[m], y_ref, ks)
        rows.append({"chemistry_class": c, "n": int(m.sum()),
                     **{k: r[k] for k in r if k != "predictions"}})
    return pd.DataFrame(rows)


# ── Levels 2–3: activation recovery ──────────────────────────────────────────
def activation_recovery(A_held: np.ndarray, A_true: np.ndarray) -> dict:
    """How well a held-out spectrum's activations match its own molecule's reference profile.

    Three views, because they fail differently: cosine cares about the whole vector, Spearman
    about the ranking, and top-k overlap about whether the right components are switched on
    at all.
    """
    cos, sp, ov = [], [], []
    for a, b in zip(A_held, A_true):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        cos.append(float(a @ b / (na * nb + EPS)))
        if a.std() > EPS and b.std() > EPS:
            v = spearmanr(a, b).statistic
            sp.append(0.0 if not np.isfinite(v) else float(v))
        ta = set(np.argsort(-a)[:3])
        tb = set(np.argsort(-b)[:3])
        ov.append(len(ta & tb) / 3)
    return {"mean_cosine": float(np.mean(cos)), "median_cosine": float(np.median(cos)),
            "mean_spearman": float(np.mean(sp)) if sp else float("nan"),
            "top3_overlap": float(np.mean(ov)), "n": len(cos)}


# ── Level 5: BSV behaviour ───────────────────────────────────────────────────
def bsv_reproducibility(BSV: np.ndarray, groups: np.ndarray) -> dict:
    """Replicate consistency, and the ratio of between-molecule to within-molecule spread.

    A coordinate system in which replicates of one substance sit further apart than different
    substances is not a coordinate system. The ratio is the number that says which it is.
    """
    N = BSV / (np.linalg.norm(BSV, axis=1, keepdims=True) + EPS)
    within, cent = [], []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if idx.size >= 2:
            C = N[idx] @ N[idx].T
            within.append(float(C[np.triu_indices(idx.size, 1)].mean()))
        cent.append(BSV[idx].mean(axis=0))
    cent = np.array(cent)
    Cn = cent / (np.linalg.norm(cent, axis=1, keepdims=True) + EPS)
    B = Cn @ Cn.T
    between = float(B[np.triu_indices(len(cent), 1)].mean())
    wd, bd = [], []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if idx.size >= 2:
            wd += [float(np.linalg.norm(BSV[a] - BSV[b]))
                   for i, a in enumerate(idx) for b in idx[i + 1:]]
    for i in range(len(cent)):
        for j in range(i + 1, len(cent)):
            bd.append(float(np.linalg.norm(cent[i] - cent[j])))
    return {"within_molecule_cosine": float(np.mean(within)) if within else float("nan"),
            "between_molecule_cosine": between,
            "separation_ratio": float(np.mean(bd) / (np.mean(wd) + EPS)) if wd else float("nan"),
            "n_replicated_molecules": len(within)}


def noise_robustness(project_fn, X: np.ndarray, sigmas=(0.01, 0.02, 0.05, 0.10),
                     n: int = 5, seed: int = 0) -> pd.DataFrame:
    """How far the BSV moves under additive noise. Deterministic given the seed."""
    rng = np.random.default_rng(seed)
    base = project_fn(X)
    Bn = base / (np.linalg.norm(base, axis=1, keepdims=True) + EPS)
    rows = []
    for s in sigmas:
        cos = []
        for _ in range(n):
            Xn = np.clip(X + rng.normal(0, s * np.abs(X).max(), X.shape), 0, None)
            V = project_fn(Xn)
            Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + EPS)
            cos += list((Bn * Vn).sum(axis=1))
        rows.append({"sigma": s, "mean_cosine": float(np.mean(cos)),
                     "min_cosine": float(np.min(cos))})
    return pd.DataFrame(rows)


def distance_preservation(A_low: np.ndarray, A_high: np.ndarray, k: int = 5) -> dict:
    """Does the abstraction preserve which spectra are near which?"""
    def dm(Z):
        N = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + EPS)
        return 1.0 - N @ N.T
    Dl, Dh = dm(A_low), dm(A_high)
    iu = np.triu_indices(Dl.shape[0], 1)
    r = spearmanr(Dl[iu], Dh[iu]).statistic
    n = Dl.shape[0]
    keep = []
    for i in range(n):
        a = set(np.argsort(Dl[i])[1:k + 1])
        b = set(np.argsort(Dh[i])[1:k + 1])
        keep.append(len(a & b) / k)
    return {"distance_spearman": float(np.nan_to_num(r)),
            "knn_preservation": float(np.mean(keep))}


# ── Level 6: geometry recovery ───────────────────────────────────────────────
def geometry_recovery(coords_new: np.ndarray, coords_ref: np.ndarray,
                      cls_new: np.ndarray, cls_ref: np.ndarray, k: int = 5) -> dict:
    """Does a held-out spectrum land in a neighbourhood of its own chemistry?

    Measured as the share of its k nearest reference coordinates that share its class, against
    the base rate that class occupies in the reference set. An earlier version compared against
    "the first five same-class training spectra by array index", which is an arbitrary target —
    the question is whether the neighbourhood is right, not whether five specific spectra were
    hit.
    """
    hits, base = [], []
    for yv, c in zip(coords_new, cls_new):
        d = np.linalg.norm(coords_ref - yv, axis=1)
        nb = np.argsort(d)[:k]
        hits.append(float((cls_ref[nb] == c).mean()))
        base.append(float((cls_ref == c).mean()))
    return {"neighbourhood_purity": float(np.mean(hits)),
            "chance_purity": float(np.mean(base)),
            "lift_over_chance": float(np.mean(hits) / (np.mean(base) + EPS)),
            "n": len(hits)}


def ood_separation(ood_in: np.ndarray, ood_out: np.ndarray) -> dict:
    """AUROC of the OOD score at separating in-domain from out-of-domain spectra."""
    y = np.r_[np.zeros(len(ood_in)), np.ones(len(ood_out))]
    s = np.r_[ood_in, ood_out]
    order = np.argsort(s)
    y = y[order]
    n_pos, n_neg = y.sum(), (1 - y).sum()
    if n_pos == 0 or n_neg == 0:
        return {"auroc": float("nan"), "n_in": len(ood_in), "n_out": len(ood_out)}
    ranks = np.arange(1, len(y) + 1)
    auc = float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return {"auroc": auc, "n_in": len(ood_in), "n_out": len(ood_out),
            "mean_in": float(np.mean(ood_in)), "mean_out": float(np.mean(ood_out))}


# ── calibration ──────────────────────────────────────────────────────────────
def calibration(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 8) -> dict:
    """Expected calibration error, plus the reliability curve.

    A confidence that does not track correctness is worse than no confidence at all, because
    downstream code will act on it.
    """
    conf = np.asarray(confidence, float)
    acc = np.asarray(correct, float)
    edges = np.linspace(conf.min(), conf.max() + 1e-9, n_bins + 1)
    ece, curve = 0.0, []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf >= a) & (conf < b)
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(conf[m].mean() - acc[m].mean())
        curve.append({"bin_lo": float(a), "bin_hi": float(b), "n": int(m.sum()),
                      "mean_confidence": float(conf[m].mean()),
                      "accuracy": float(acc[m].mean())})
    return {"ece": float(ece), "curve": curve}


# ── the leakage control ──────────────────────────────────────────────────────
def leakage_control(X: np.ndarray, y: np.ndarray, cls_of: dict, folds: np.ndarray,
                    H_frozen: np.ndarray, lsm_classes: list[str], k_of_class: dict,
                    weights: np.ndarray, seed: int = 0) -> pd.DataFrame:
    """How much of the retrieval performance comes from the dictionary having seen the answer.

    For each fold: refit the class-local NMF **without** that fold's molecules, at the same
    `k_c`, and rerun the retrieval against the fold-honest dictionary. Compare with the same
    retrieval against the frozen dictionary.

    This does not modify anything frozen — it is a control experiment, in memory, exactly like
    the Phase 01 control that was preserved and relabelled rather than deleted. Refitting is
    forbidden as a *replacement* for the frozen atlas; it is the only way to *measure* what the
    frozen atlas's numbers are worth.
    """
    from gaira.v7.engine.projection import project
    from gaira.v7.lsm.classlocal import fit_nmf

    rows = []
    cls_arr = np.array([cls_of.get(v, "") for v in y])
    for f in sorted(set(folds)):
        te = folds == f
        tr = ~te
        if te.sum() == 0 or tr.sum() < 5:
            continue
        # honest dictionary: refit each class on training molecules only
        blocks, keep_cols = [], []
        for c, kc in sorted(k_of_class.items()):
            m = tr & (cls_arr == c)
            if m.sum() <= kc or kc < 1:
                continue
            try:
                _, Hc, _ = fit_nmf(X[m] * weights[m][:, None], int(kc), seed=seed)
            except Exception:                                    # pragma: no cover
                continue
            blocks.append(Hc)
            keep_cols += [c] * int(kc)
        if not blocks:
            continue
        H_honest = np.vstack(blocks)

        # Retrieval is scored on SPLIT A within the held-out fold: each test spectrum is
        # queried against the other spectra of the same fold, so its own molecule is present
        # and molecule top-k is defined — while the DICTIONARY has never seen the fold. That
        # isolates dictionary-level leakage from the split-level artefact.
        te_idx = np.where(te)[0]
        counts = {v: int((y[te_idx] == v).sum()) for v in set(y[te_idx])}
        q = np.array([i for i in te_idx if counts[y[i]] >= 2])
        if q.size < 3:
            continue
        for tag, D in (("frozen_dictionary", H_frozen), ("fold_honest_dictionary", H_honest)):
            A = project(X, D, "nnls")
            hits = {k: [] for k in ("top1", "top3", "top5")}
            rr = []
            for i in q:
                ref = np.array([j for j in te_idx if j != i])
                r = molecule_retrieval(A[[i]], A[ref], y[[i]], y[ref])
                for k in hits:
                    hits[k].append(r[k])
                rr.append(r["mrr"])
            rows.append({"fold": int(f), "dictionary": tag, "n_test": int(q.size),
                         "n_components": int(D.shape[0]),
                         **{k: float(np.mean(v)) for k, v in hits.items()},
                         "mrr": float(np.mean(rr))})
    return pd.DataFrame(rows)
