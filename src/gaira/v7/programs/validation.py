"""GAIRA V7 — Phase 07: validation of a biochemical programme layer.

**This phase is not judged by molecule retrieval.** The questions are whether the programmes
reconstruct the chemistry they compress, whether they are stable, whether they generalise to
molecules the factorisation never saw, and whether they are anything more than a rotation of the
input.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def _rownorm(M):
    return np.asarray(M, float) / (np.linalg.norm(M, axis=1, keepdims=True) + EPS)


# ── stability ────────────────────────────────────────────────────────────────
def bootstrap_recovery(Ev, family, K, n_boot: int = 25, frac: float = 0.85,
                       seed: int = 0, **kw) -> dict:
    """Hungarian-matched programme recovery under resampling of the spectra."""
    from scipy.optimize import linear_sum_assignment
    from .factorization import fit
    base = fit(family, Ev, K, seed, **kw)["P"]
    rng = np.random.default_rng(seed)
    n = len(Ev)
    scores, per = [], np.zeros(K)
    for _ in range(n_boot):
        idx = np.sort(rng.choice(n, int(frac * n), replace=False))
        try:
            P = fit(family, Ev[idx], K, seed, **kw)["P"]
        except Exception:                                          # pragma: no cover
            continue
        C = _rownorm(np.abs(base)) @ _rownorm(np.abs(P)).T
        r, c = linear_sum_assignment(-C)
        scores.append(float(C[r, c].mean()))
        per[r] += C[r, c]
    return {"mean": float(np.mean(scores)) if scores else np.nan,
            "min": float(np.min(scores)) if scores else np.nan,
            "sd": float(np.std(scores)) if scores else np.nan,
            "per_programme": (per / max(len(scores), 1)).tolist(),
            "n_programmes_below_0.7": int(sum(v / max(len(scores), 1) < 0.7 for v in per))}


def seed_stability(Ev, family, K, seeds=(0, 1, 2, 3, 4)) -> float:
    """Programme recovery across random seeds — a different question from resampling."""
    from scipy.optimize import linear_sum_assignment
    from .factorization import fit
    base = fit(family, Ev, K, seeds[0])["P"]
    out = []
    for s in seeds[1:]:
        try:
            P = fit(family, Ev, K, s)["P"]
        except Exception:                                          # pragma: no cover
            continue
        C = _rownorm(np.abs(base)) @ _rownorm(np.abs(P)).T
        r, c = linear_sum_assignment(-C)
        out.append(float(C[r, c].mean()))
    return float(np.mean(out)) if out else np.nan


def fold_stability(Ev, folds, family, K, seed: int = 0) -> float:
    """Programme recovery when a whole molecule-grouped fold is withheld from the fit."""
    from scipy.optimize import linear_sum_assignment
    from .factorization import fit
    base = fit(family, Ev, K, seed)["P"]
    out = []
    for f in sorted(set(np.asarray(folds).tolist())):
        tr = np.asarray(folds) != f
        if tr.sum() < 3 * K:
            continue
        try:
            P = fit(family, Ev[tr], K, seed)["P"]
        except Exception:                                          # pragma: no cover
            continue
        C = _rownorm(np.abs(base)) @ _rownorm(np.abs(P)).T
        r, c = linear_sum_assignment(-C)
        out.append(float(C[r, c].mean()))
    return float(np.mean(out)) if out else np.nan


# ── information ──────────────────────────────────────────────────────────────
def information_retained(Ev, model, W=None) -> float:
    """Fraction of the Chemistry Evidence variance the programmes reconstruct."""
    from .factorization import reconstruction
    return float(max(0.0, reconstruction(Ev, model, W)["explained_variance"]))


def mutual_information_with_chemistry(W, cls, n_bins: int = 8) -> float:
    """MI between the programme activations and the chemistry class, in nats.

    Each programme is discretised into equal-frequency bins and the MI summed over programmes,
    then normalised by the class entropy. It is a coarse estimator; it is used comparatively —
    BSV2 against Chemistry Evidence against PCA — never as an absolute quantity.
    """
    from sklearn.metrics import mutual_info_score
    from scipy.stats import entropy as sent
    cls = np.asarray(cls)
    _, cc = np.unique(cls, return_counts=True)
    H = float(sent(cc / cc.sum()))
    tot = 0.0
    for k in range(np.asarray(W).shape[1]):
        v = np.asarray(W)[:, k]
        try:
            b = np.digitize(v, np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1]))
        except Exception:                                          # pragma: no cover
            continue
        tot += float(mutual_info_score(cls, b))
    return float(tot / (H + EPS))


def heldout_chemistry(W, cls, folds, y) -> dict:
    """Can a held-out molecule's chemistry class be read off its programme activations?

    **Not molecule retrieval** — the brief forbids judging this phase on that. This is the
    informativeness floor: a compression that cannot say what chemistry a spectrum has, has not
    preserved the chemistry it compressed. Nearest-class-prototype in programme space, with the
    prototypes rebuilt inside every molecule-grouped fold.
    """
    W = np.asarray(W, float)
    cls, folds, y = np.asarray(cls), np.asarray(folds), np.asarray(y)
    classes = sorted(set(cls.tolist()))
    hit1 = hit3 = 0
    for f in sorted(set(folds.tolist())):
        te, tr = folds == f, folds != f
        proto = {}
        for c in classes:
            m = tr & (cls == c)
            if m.any():
                mols = sorted(set(y[m].tolist()))
                proto[c] = np.vstack([W[(y == mm) & m].mean(axis=0) for mm in mols]).mean(axis=0)
        if not proto:
            continue
        names = list(proto)
        Pm = _rownorm(np.vstack([proto[c] for c in names]))
        S = _rownorm(W[te]) @ Pm.T
        for i, row in enumerate(S):
            order = np.argsort(-row)
            hit1 += cls[te][i] == names[order[0]]
            hit3 += cls[te][i] in [names[j] for j in order[:3]]
    return {"top1": hit1 / len(W), "top3": hit3 / len(W)}


# ── coherence ────────────────────────────────────────────────────────────────
def programme_coherence(W, P, cls) -> "pd.DataFrame":
    """Per-programme: which spectra use it, how specific it is, how redundant."""
    import pandas as pd
    from .factorization import overlap
    W = np.clip(np.asarray(W, float), 0, None)
    O = overlap(P)
    Wn = _rownorm(W)
    rows = []
    for k in range(W.shape[1]):
        w = W[:, k]
        top = w > np.quantile(w, 0.80)
        if top.sum() < 2:
            rows.append({"programme": k, "n_top_spectra": int(top.sum()),
                         "within_similarity": np.nan, "between_similarity": np.nan,
                         "specificity": np.nan, "max_overlap": float(
                             np.max([O[k, j] for j in range(W.shape[1]) if j != k])),
                         "usage_share": float((np.argmax(W, axis=1) == k).mean())})
            continue
        Cw = Wn[top] @ Wn[top].T
        iu = np.triu_indices(int(top.sum()), 1)
        within = float(Cw[iu].mean())
        between = float((Wn[top] @ Wn[~top].T).mean()) if (~top).any() else np.nan
        _, cc = np.unique(cls[top], return_counts=True)
        p = cc / cc.sum()
        spec = float(1.0 - (-(p * np.log(p + EPS)).sum() / np.log(max(len(p), 2))))
        rows.append({"programme": k, "n_top_spectra": int(top.sum()),
                     "within_similarity": within, "between_similarity": between,
                     "separation": within - between,
                     "specificity": spec,
                     "dominant_class": str(np.unique(cls[top], return_counts=True)[0][
                         int(np.argmax(cc))]),
                     "max_overlap": float(np.max([O[k, j] for j in range(W.shape[1]) if j != k])),
                     "usage_share": float((np.argmax(W, axis=1) == k).mean())})
    return pd.DataFrame(rows)


def replicate_consistency(W, y) -> float:
    N = _rownorm(np.clip(np.asarray(W, float), 0, None))
    y = np.asarray(y)
    vals = []
    for m in set(y.tolist()):
        idx = np.where(y == m)[0]
        if len(idx) < 2:
            continue
        C = N[idx] @ N[idx].T
        iu = np.triu_indices(len(idx), 1)
        vals.append(float(C[iu].mean()))
    return float(np.mean(vals)) if vals else np.nan
