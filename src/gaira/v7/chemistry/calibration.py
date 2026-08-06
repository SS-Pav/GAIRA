"""GAIRA V7 — Phase 06: multiclass calibration of the Chemistry Evidence Vector.

Phase 05 established that **ECE alone selects a constant predictor**: the ECE-optimal calibrator
reported the base rate for every spectrum, with sharpness exactly 0.000 and the worst Brier in
the table. Selection here therefore uses a proper scoring rule with explicit non-degeneracy
requirements, and any calibrator returning near-constant probabilities is rejected however good
its ECE (principle P-18).
"""
from __future__ import annotations

import numpy as np

from .registry import CLASS_ORDER

EPS = 1e-12
NC = len(CLASS_ORDER)
METHODS = ("uncalibrated", "temperature", "vector_scaling", "dirichlet", "isotonic", "platt")


def _softmax(Z):
    Z = np.asarray(Z, float)
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / (E.sum(axis=1, keepdims=True) + EPS)


def _logits(E):
    """Evidence → logits. Log of the L1-normalised evidence, floored so zeros are finite."""
    P = np.clip(np.atleast_2d(E), 0.0, None)
    P = P / (P.sum(axis=1, keepdims=True) + EPS)
    return np.log(np.clip(P, 1e-8, None))


# ── metrics ──────────────────────────────────────────────────────────────────
def ece(P, cls, n_bins: int = 10) -> float:
    conf = P.max(axis=1)
    correct = np.array([CLASS_ORDER[int(np.argmax(p))] == c for p, c in zip(P, cls)], float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def classwise_ece(P, cls, n_bins: int = 10) -> float:
    """Mean over classes of the one-vs-rest ECE. Top-label ECE hides small-class failure."""
    out = []
    for k, c in enumerate(CLASS_ORDER):
        y = (cls == c).astype(float)
        p = P[:, k]
        edges = np.linspace(0, 1, n_bins + 1)
        e = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p > lo) & (p <= hi)
            if m.any():
                e += m.mean() * abs(y[m].mean() - p[m].mean())
        out.append(e)
    return float(np.mean(out))


def brier(P, cls) -> float:
    Y = np.zeros_like(P)
    for i, c in enumerate(cls):
        Y[i, CLASS_ORDER.index(c)] = 1.0
    return float(((P - Y) ** 2).sum(axis=1).mean())


def log_loss(P, cls) -> float:
    idx = np.array([CLASS_ORDER.index(c) for c in cls])
    return float(-np.log(np.clip(P[np.arange(len(P)), idx], 1e-12, None)).mean())


def sharpness(P) -> float:
    return float(np.std(P.max(axis=1)))


def discrimination(P, cls) -> float:
    """AUROC of the top-label confidence separating correct from incorrect predictions."""
    from scipy.stats import rankdata
    conf = P.max(axis=1)
    correct = np.array([CLASS_ORDER[int(np.argmax(p))] == c for p, c in zip(P, cls)], bool)
    if correct.all() or (~correct).all():
        return float("nan")
    r = rankdata(conf)
    n1, n0 = correct.sum(), (~correct).sum()
    return float((r[correct].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def reliability(P, cls, n_bins: int = 10):
    conf = P.max(axis=1)
    correct = np.array([CLASS_ORDER[int(np.argmax(p))] == c for p, c in zip(P, cls)], float)
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        xs.append((lo + hi) / 2)
        ys.append(float(correct[m].mean()) if m.any() else np.nan)
        ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def selective_accuracy(P, cls, n: int = 25):
    conf = P.max(axis=1)
    correct = np.array([CLASS_ORDER[int(np.argmax(p))] == c for p, c in zip(P, cls)], float)
    out = []
    for q in np.linspace(0.0, 0.95, n):
        t = np.quantile(conf, q)
        m = conf >= t
        if m.sum() < 5:
            break
        out.append({"threshold": float(t), "coverage": float(m.mean()),
                    "accuracy": float(correct[m].mean())})
    return out


# ── calibrators ──────────────────────────────────────────────────────────────
class Calibrator:
    """Fitted inside training folds only; applied frozen."""

    def __init__(self, method: str = "temperature"):
        if method not in METHODS:
            raise ValueError(f"unknown calibration method {method}")
        self.method = method
        self.params_: dict = {}

    def fit(self, E, cls):
        L = _logits(E)
        idx = np.array([CLASS_ORDER.index(c) for c in cls])
        if self.method == "uncalibrated":
            self.params_ = {}
        elif self.method == "temperature":
            grid = np.exp(np.linspace(np.log(1e-2), np.log(1e2), 240))
            best = min(grid, key=lambda t: log_loss(_softmax(L / t), cls))
            self.params_ = {"T": float(best)}
        elif self.method == "vector_scaling":
            # Per-class scale and bias on the logits — 32 parameters, fitted by multinomial
            # logistic regression on the diagonal features. More expressive than a single
            # temperature and still fully inspectable.
            from sklearn.linear_model import LogisticRegression
            m = LogisticRegression(max_iter=4000, C=1.0, random_state=0).fit(L, idx)
            self.params_ = {"model": m}
        elif self.method == "dirichlet":
            from sklearn.linear_model import LogisticRegression
            m = LogisticRegression(max_iter=4000, C=0.3, random_state=0).fit(
                np.hstack([L, L.mean(axis=1, keepdims=True), L.max(axis=1, keepdims=True)]), idx)
            self.params_ = {"model": m}
        elif self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            models = []
            P0 = _softmax(L)
            for k in range(NC):
                yk = (idx == k).astype(float)
                if len(set(yk.tolist())) < 2:
                    models.append(None)
                    continue
                models.append(IsotonicRegression(out_of_bounds="clip", y_min=0.0,
                                                 y_max=1.0).fit(P0[:, k], yk))
            self.params_ = {"models": models}
        elif self.method == "platt":
            from sklearn.linear_model import LogisticRegression
            models = []
            P0 = _softmax(L)
            for k in range(NC):
                yk = (idx == k).astype(float)
                if len(set(yk.tolist())) < 2:
                    models.append(None)
                    continue
                models.append(LogisticRegression(max_iter=2000).fit(P0[:, k, None], yk))
            self.params_ = {"models": models}
        return self

    def transform(self, E) -> np.ndarray:
        L = _logits(E)
        if self.method == "uncalibrated":
            return _softmax(L)
        if self.method == "temperature":
            return _softmax(L / self.params_["T"])
        if self.method in ("vector_scaling", "dirichlet"):
            m = self.params_["model"]
            F = L if self.method == "vector_scaling" else np.hstack(
                [L, L.mean(axis=1, keepdims=True), L.max(axis=1, keepdims=True)])
            P = np.zeros((len(L), NC))
            P[:, list(m.classes_)] = m.predict_proba(F)
            return P / (P.sum(axis=1, keepdims=True) + EPS)
        P0 = _softmax(L)
        P = np.zeros_like(P0)
        for k, mdl in enumerate(self.params_["models"]):
            if mdl is None:
                P[:, k] = P0[:, k]
            elif self.method == "isotonic":
                P[:, k] = np.clip(mdl.predict(P0[:, k]), 0.0, 1.0)
            else:
                P[:, k] = mdl.predict_proba(P0[:, k, None])[:, 1]
        return P / (P.sum(axis=1, keepdims=True) + EPS)


# Non-degeneracy floors, declared before the benchmark runs (P-18).
SHARPNESS_FLOOR = 0.05
DISCRIMINATION_FLOOR = 0.60


def benchmark(E_tr, cls_tr, E_te, cls_te) -> "pd.DataFrame":
    """Fit on the training split, score on the held-out split. Ranked by log loss."""
    import pandas as pd
    rows = []
    for m in METHODS:
        try:
            cal = Calibrator(m).fit(E_tr, cls_tr)
            P = cal.transform(E_te)
        except Exception as exc:                                   # pragma: no cover
            rows.append({"method": m, "usable": False, "error": str(exc)[:60]})
            continue
        rows.append({"method": m, "usable": True, "log_loss": log_loss(P, cls_te),
                     "brier": brier(P, cls_te), "ece": ece(P, cls_te),
                     "classwise_ece": classwise_ece(P, cls_te),
                     "sharpness": sharpness(P), "discrimination": discrimination(P, cls_te),
                     "top1": float(np.mean([CLASS_ORDER[int(np.argmax(p))] == c
                                            for p, c in zip(P, cls_te)]))})
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)


def select(summary: "pd.DataFrame") -> tuple[str, str]:
    """The selection rule, declared before the numbers arrive.

    Minimise log loss — strictly proper, so it rewards being calibrated *and* sharp — among
    methods that clear the non-degeneracy floors. Returns the choice and the reason, so a
    fallback is visible in the log rather than silent.
    """
    ok = summary[(summary.get("usable", True)) &
                 (summary.sharpness > SHARPNESS_FLOOR) &
                 (summary.discrimination > DISCRIMINATION_FLOOR)]
    if len(ok):
        return str(ok.sort_values("log_loss").iloc[0]["method"]), "log_loss among non-degenerate"
    return (str(summary.sort_values("log_loss").iloc[0]["method"]),
            "FALLBACK: no method cleared the non-degeneracy floors")
