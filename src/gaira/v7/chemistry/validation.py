"""GAIRA V7 — Phase 06: nested molecule-grouped validation and soft-evidence metrics.

The held-out molecule must be absent from the reference bank, the prototypes, the calibrator,
the model-selection loop and every hyperparameter choice. Nested CV is what makes that true for
*selection* as well as for evaluation, and Phase 05 showed why it matters: a metric chosen on
the fold it is reported on is not a held-out number.
"""
from __future__ import annotations

import numpy as np

from .registry import ADJACENT, CLASS_ORDER

EPS = 1e-12
NC = len(CLASS_ORDER)
_ADJ = {c: set() for c in CLASS_ORDER}
for _a, _b in ADJACENT:
    _ADJ[_a].add(_b)
    _ADJ[_b].add(_a)


# ── hard metrics ─────────────────────────────────────────────────────────────
def rank_of_true(E: np.ndarray, cls: np.ndarray) -> np.ndarray:
    """1-based rank of the true class in each evidence vector. Ties broken pessimistically."""
    out = np.zeros(len(E), int)
    for i, c in enumerate(cls):
        j = CLASS_ORDER.index(c)
        out[i] = int((E[i] > E[i, j]).sum() + 1)
    return out


def topk(E, cls, k=1) -> float:
    return float(np.mean(rank_of_true(E, cls) <= k))


def mrr(E, cls) -> float:
    return float(np.mean(1.0 / rank_of_true(E, cls)))


def per_class(E, cls) -> "pd.DataFrame":
    import pandas as pd
    pred = np.array([CLASS_ORDER[int(np.argmax(e))] for e in E])
    rows = []
    for c in CLASS_ORDER:
        tp = int(((pred == c) & (cls == c)).sum())
        fp = int(((pred == c) & (cls != c)).sum())
        fn = int(((pred != c) & (cls == c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        rows.append({"class_id": c, "n": int((cls == c).sum()), "tp": tp, "fp": fp, "fn": fn,
                     "precision": pr, "recall": rc,
                     "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0})
    return pd.DataFrame(rows)


def macro_f1(E, cls) -> float:
    t = per_class(E, cls)
    return float(t[t.n > 0].f1.mean())


def balanced_accuracy(E, cls) -> float:
    pred = np.array([CLASS_ORDER[int(np.argmax(e))] for e in E])
    return float(np.mean([((pred == c) & (cls == c)).sum() / max((cls == c).sum(), 1)
                          for c in CLASS_ORDER if (cls == c).sum() > 0]))


def confusion(E, cls) -> "pd.DataFrame":
    import pandas as pd
    pred = np.array([CLASS_ORDER[int(np.argmax(e))] for e in E])
    M = pd.DataFrame(0, index=list(CLASS_ORDER), columns=list(CLASS_ORDER))
    for t, p in zip(cls, pred):
        M.loc[t, p] += 1
    return M


def adjacency_of_errors(E, cls) -> dict:
    """Are the mistakes chemically reasonable? Reported separately, never scored as correct."""
    pred = np.array([CLASS_ORDER[int(np.argmax(e))] for e in E])
    wrong = pred != cls
    if not wrong.any():
        return {"n_errors": 0, "adjacent_fraction": float("nan"), "chance_adjacent": float("nan")}
    adj = np.mean([p in _ADJ[t] for t, p in zip(cls[wrong], pred[wrong])])
    # Chance rate: if an error picked a wrong class uniformly at random, how often would it be
    # adjacent? Without this the adjacency number means nothing.
    chance = float(np.mean([len(_ADJ[t]) / (NC - 1) for t in cls[wrong]]))
    return {"n_errors": int(wrong.sum()), "adjacent_fraction": float(adj),
            "chance_adjacent": chance, "lift": float(adj / (chance + EPS))}


# ── soft-evidence metrics ────────────────────────────────────────────────────
def true_class_evidence(E, cls) -> np.ndarray:
    return np.array([E[i, CLASS_ORDER.index(c)] for i, c in enumerate(cls)])


def margin(E) -> np.ndarray:
    """Top-1 minus top-2 evidence — how decisive the vector is."""
    S = np.sort(E, axis=1)
    return S[:, -1] - S[:, -2]


def entropy(E) -> np.ndarray:
    P = E / (E.sum(axis=1, keepdims=True) + EPS)
    return -(np.where(P > 0, P * np.log(P + EPS), 0.0)).sum(axis=1) / np.log(NC)


def effective_rank(E) -> float:
    s = np.linalg.svd(np.asarray(E, float) - np.asarray(E, float).mean(axis=0),
                      compute_uv=False)
    p = s ** 2 / ((s ** 2).sum() + EPS)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def within_between(E, cls) -> dict:
    """Do same-class evidence vectors agree, and do different-class vectors differ?"""
    N = E / (np.linalg.norm(E, axis=1, keepdims=True) + EPS)
    C = N @ N.T
    same = cls[:, None] == cls[None, :]
    iu = np.triu_indices(len(E), 1)
    w = C[iu][same[iu]]
    b = C[iu][~same[iu]]
    return {"within_class_cosine": float(w.mean()) if len(w) else float("nan"),
            "between_class_cosine": float(b.mean()) if len(b) else float("nan"),
            "separation": float(w.mean() - b.mean()) if len(w) and len(b) else float("nan")}


def replicate_consistency(E, y) -> float:
    """Mean pairwise cosine between the evidence vectors of one molecule's replicates."""
    N = E / (np.linalg.norm(E, axis=1, keepdims=True) + EPS)
    vals = []
    for m in set(y.tolist()):
        idx = np.where(y == m)[0]
        if len(idx) < 2:
            continue
        C = N[idx] @ N[idx].T
        iu = np.triu_indices(len(idx), 1)
        vals.append(float(C[iu].mean()))
    return float(np.mean(vals)) if vals else float("nan")


# ── nested cross-validation ──────────────────────────────────────────────────
def nested_cv(A, y, cls, folds, candidates, fit_fn, predict_fn, select_metric=macro_f1,
              log=None) -> dict:
    """Outer folds evaluate; inner folds select. The two never share a molecule.

    `select_metric` decides the inner-loop winner. Macro-F1 rather than top-1, because the
    corpus is imbalanced 80:3 between its largest and smallest classes and top-1 would let the
    selection ignore the small ones entirely.
    """
    outer = sorted(set(folds))
    E_out = np.zeros((len(A), NC))
    chosen, per_fold, inner_scores = {}, [], {}
    for f in outer:
        te, tr = folds == f, folds != f
        inner = sorted(set(folds[tr]))
        best, best_s, scores = None, -np.inf, {}
        for name, cfg in candidates.items():
            sc = []
            for g in inner:
                itr, ite = tr & (folds != g), tr & (folds == g)
                if ite.sum() == 0 or itr.sum() < 20:
                    continue
                try:
                    m = fit_fn(A[itr], y[itr], cls[itr], cfg)
                    sc.append(select_metric(predict_fn(m, A[ite]), cls[ite]))
                except Exception:
                    sc = []
                    break
            if sc:
                scores[name] = float(np.mean(sc))
                if scores[name] > best_s:
                    best_s, best = scores[name], name
        if best is None:
            # Every candidate failed, or no inner fold was large enough to score one. Failing
            # loudly beats a KeyError three frames down, and beats silently falling back to an
            # arbitrary candidate.
            raise RuntimeError(
                f"outer fold {f}: no candidate could be selected — "
                f"{len(candidates)} offered, {len(set(folds[tr]))} inner folds, "
                f"{int(tr.sum())} training spectra")
        chosen[int(f)] = best
        inner_scores[int(f)] = scores
        m = fit_fn(A[tr], y[tr], cls[tr], candidates[best])
        E_out[te] = predict_fn(m, A[te])
        per_fold.append({"fold": int(f), "selected": best, "inner_score": best_s,
                         "n_test": int(te.sum()),
                         "outer_top1": topk(E_out[te], cls[te], 1),
                         "outer_macro_f1": macro_f1(E_out[te], cls[te])})
        if log:
            log(f"    fold {f}: selected {best} (inner {select_metric.__name__} {best_s:.3f}) "
                f"→ outer top-1 {per_fold[-1]['outer_top1']:.3f}")
    return {"E": E_out, "chosen_per_fold": chosen, "per_fold": per_fold,
            "inner_scores": inner_scores,
            "modal_choice": max(set(chosen.values()),
                                key=lambda c: (sum(v == c for v in chosen.values()), str(c)))}


def bootstrap_ci(E, y, cls, fn, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Molecule-level bootstrap. Resampling spectra would treat replicates as independent."""
    rng = np.random.default_rng(seed)
    mols = np.array(sorted(set(y.tolist())))
    idx_of = {m: np.where(y == m)[0] for m in mols}
    point = fn(E, cls)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(mols), len(mols), replace=True)
        idx = np.concatenate([idx_of[mols[p]] for p in pick])
        try:
            vals.append(fn(E[idx], cls[idx]))
        except Exception:
            continue
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    return float(point), float(lo), float(hi)
