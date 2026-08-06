"""GAIRA V7 — Phase 04.5: the twelve validation axes and the robustness study.

Four representations are compared on identical frozen splits: raw spectrum, LSM activations,
CSM activations, Meta Component activations. No representation gets a different split, a
different metric or a different query set.

The robustness study is the phase's primary contribution, and it is written to be capable of
returning a negative result: if Meta Components degrade in lockstep with CSMs, the area-under-
robustness-curve difference will be zero and the report has to say so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


# ── retrieval ────────────────────────────────────────────────────────────────
def _sim(Q, R):
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + EPS)
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + EPS)
    return Qn @ Rn.T


def retrieval_split_a(A: np.ndarray, y: np.ndarray, repl: np.ndarray,
                      A_query: np.ndarray | None = None) -> dict:
    """Leave-one-spectrum-out molecule retrieval over replicated molecules.

    `A_query` lets a perturbed spectrum be queried against the CLEAN reference set, which is
    the realistic setting: the atlas is clean, the new measurement is not.
    """
    Q = A if A_query is None else A_query
    qi = np.where(repl)[0]
    hits = {1: [], 3: [], 5: []}
    rr = []
    for i in qi:
        ref = np.array([j for j in range(A.shape[0]) if j != i])
        s = _sim(Q[[i]], A[ref])[0]
        ranked = y[ref][np.argsort(-s)]
        for k in hits:
            hits[k].append(float(y[i] in set(ranked[:k])))
        w = np.where(ranked == y[i])[0]
        rr.append(1.0 / (w[0] + 1) if w.size else 0.0)
    return {"top1": float(np.mean(hits[1])), "top3": float(np.mean(hits[3])),
            "top5": float(np.mean(hits[5])), "mrr": float(np.mean(rr)), "n": len(qi)}


def retrieval_split_b(A: np.ndarray, cls: np.ndarray, folds: np.ndarray,
                      A_query: np.ndarray | None = None) -> dict:
    """Molecule-grouped chemistry retrieval, with balanced accuracy and macro F1.

    Molecule top-k is undefined under this split (the molecule is withheld with all its
    spectra), so only class-level metrics are reported — the Phase 04 finding, carried forward.
    """
    Q = A if A_query is None else A_query
    hits = {1: [], 3: [], 5: []}
    y_true, y_pred = [], []
    for f in sorted(set(folds)):
        te, tr = folds == f, folds != f
        if te.sum() == 0 or tr.sum() < 5:
            continue
        s = _sim(Q[te], A[tr])
        order = np.argsort(-s, axis=1)
        for a, i in enumerate(np.where(te)[0]):
            ranked = cls[tr][order[a]]
            for k in hits:
                hits[k].append(float(cls[i] in set(ranked[:k])))
            y_true.append(cls[i])
            y_pred.append(ranked[0])
    from sklearn.metrics import balanced_accuracy_score, f1_score
    return {"top1": float(np.mean(hits[1])), "top3": float(np.mean(hits[3])),
            "top5": float(np.mean(hits[5])),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "n": len(y_true)}


def replicate_consistency(A: np.ndarray, y: np.ndarray) -> float:
    N = A / (np.linalg.norm(A, axis=1, keepdims=True) + EPS)
    out = []
    for g in np.unique(y):
        idx = np.where(y == g)[0]
        if idx.size < 2:
            continue
        C = N[idx] @ N[idx].T
        out.append(float(C[np.triu_indices(idx.size, 1)].mean()))
    return float(np.mean(out)) if out else float("nan")


def cross_fold_reproducibility(A: np.ndarray, cls: np.ndarray, folds: np.ndarray) -> float:
    """Spread of split-B top-1 across folds — low spread means the number is not fold luck."""
    per = []
    for f in sorted(set(folds)):
        te, tr = folds == f, folds != f
        if te.sum() == 0 or tr.sum() < 5:
            continue
        s = _sim(A[te], A[tr])
        pred = cls[tr][np.argmax(s, axis=1)]
        per.append(float((pred == cls[te]).mean()))
    return float(1.0 - np.std(per) / (np.mean(per) + EPS)) if per else float("nan")


def activation_sparsity(A: np.ndarray) -> float:
    n = A.shape[1]
    out = []
    for a in A:
        l1, l2 = np.abs(a).sum(), np.linalg.norm(a)
        out.append((np.sqrt(n) - l1 / (l2 + EPS)) / (np.sqrt(n) - 1 + EPS))
    return float(np.mean(out))


def biochemical_coherence(A: np.ndarray, cls: np.ndarray, k: int = 5) -> float:
    """Fraction of each spectrum's k nearest neighbours sharing its chemistry class,
    expressed as a lift over the base rate."""
    S = _sim(A, A)
    np.fill_diagonal(S, -np.inf)
    hits = [float((cls[np.argsort(-S[i])[:k]] == cls[i]).mean()) for i in range(A.shape[0])]
    _, c = np.unique(cls, return_counts=True)
    base = float(((c / c.sum()) ** 2).sum())
    return float(np.mean(hits) / (base + EPS))


def information_retained(A_ref: np.ndarray, A_new: np.ndarray) -> dict:
    """How much of the CSM layer survives, measured two ways.

    Linear reconstruction says whether the information is still *there*; neighbourhood
    preservation says whether it is still *usable* — a representation can retain variance and
    destroy the neighbour structure retrieval depends on.
    """
    from numpy.linalg import lstsq
    B, *_ = lstsq(A_new, A_ref, rcond=None)
    ev = float(max(0.0, 1.0 - ((A_ref - A_new @ B) ** 2).sum() / ((A_ref ** 2).sum() + EPS)))
    Sr, Sn = _sim(A_ref, A_ref), _sim(A_new, A_new)
    np.fill_diagonal(Sr, -np.inf)
    np.fill_diagonal(Sn, -np.inf)
    keep = [len(set(np.argsort(-Sr[i])[:5]) & set(np.argsort(-Sn[i])[:5])) / 5
            for i in range(A_ref.shape[0])]
    return {"linear_ev": ev, "knn_preservation": float(np.mean(keep))}


def calibration(scores: np.ndarray, correct: np.ndarray, n_bins: int = 8) -> dict:
    s = np.asarray(scores, float)
    a = np.asarray(correct, float)
    edges = np.linspace(s.min(), s.max() + 1e-9, n_bins + 1)
    ece, curve = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & (s < hi)
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(s[m].mean() - a[m].mean())
        curve.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                      "mean_score": float(s[m].mean()), "accuracy": float(a[m].mean())})
    return {"ece": float(ece), "curve": curve}


# ── robustness ───────────────────────────────────────────────────────────────
def activation_stability(A_clean: np.ndarray, A_pert: np.ndarray) -> float:
    N1 = A_clean / (np.linalg.norm(A_clean, axis=1, keepdims=True) + EPS)
    N2 = A_pert / (np.linalg.norm(A_pert, axis=1, keepdims=True) + EPS)
    return float((N1 * N2).sum(axis=1).mean())


def area_under_robustness(levels, values, baseline: float) -> float:
    """Normalised AUC of a degradation curve — 1.0 means no degradation at any level.

    Normalised by the clean baseline so representations starting at different accuracies are
    compared on *retained fraction*, which is the question. A representation that starts lower
    and stays flat can beat one that starts high and collapses.
    """
    lv = np.asarray(levels, float)
    v = np.asarray(values, float) / (baseline + EPS)
    if lv.size < 2:
        return float(v.mean())
    x = (lv - lv.min()) / (lv.max() - lv.min() + EPS)
    return float(np.trapezoid(v, x))
