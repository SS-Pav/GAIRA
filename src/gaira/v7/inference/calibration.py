"""GAIRA V7 — Phase 05, Step 3: turning a similarity into a confidence.

A cosine of 0.94 is not a probability, and reporting it as one is the single easiest way for an
inference engine to mislead. Four calibration families are fitted on grouped CV predictions and
scored by Expected Calibration Error, Brier score and reliability; the most calibrated wins.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12
METHODS = ("uncalibrated", "temperature", "isotonic", "platt", "dirichlet")


def _softmax(S, T=1.0):
    Z = np.asarray(S, float) / max(T, 1e-6)
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / (E.sum(axis=1, keepdims=True) + EPS)


def expected_calibration_error(conf, correct, n_bins: int = 10) -> float:
    """ECE — mean |confidence − accuracy| over equal-width confidence bins, weighted by count."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def maximum_calibration_error(conf, correct, n_bins: int = 10) -> float:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    worst = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= 5:
            worst = max(worst, abs(correct[m].mean() - conf[m].mean()))
    return float(worst)


def brier(conf, correct) -> float:
    """Top-label Brier score — proper, so it rewards being both calibrated and sharp."""
    return float(np.mean((np.asarray(conf, float) - np.asarray(correct, float)) ** 2))


def reliability_curve(conf, correct, n_bins: int = 10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        xs.append((lo + hi) / 2)
        ys.append(float(correct[m].mean()) if m.sum() else np.nan)
        ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


class Calibrator:
    """Fitted on held-out grouped-CV scores; applied frozen at inference.

    Every method maps a *score matrix* to a confidence in [0, 1] for the top label. Fitting uses
    only training-fold outcomes, so the reported calibration is not the calibration the mapping
    was optimised on.
    """

    def __init__(self, method: str = "temperature"):
        if method not in METHODS:
            raise ValueError(f"unknown calibration method {method}")
        self.method = method
        self.params_: dict = {}

    def fit(self, S: np.ndarray, correct: np.ndarray):
        S = np.atleast_2d(np.asarray(S, float))
        correct = np.asarray(correct, float)
        if self.method == "uncalibrated":
            self.params_ = {}
        elif self.method == "temperature":
            # One scalar, chosen to minimise the **Brier** score. Fitting it to ECE instead —
            # the first version did — rewards flattening the confidences toward the base rate,
            # which is the same degeneracy that made Platt scaling win the first benchmark with
            # a constant 0.605. Brier is proper, so it cannot be gamed that way.
            grid = np.exp(np.linspace(np.log(1e-3), np.log(1e2), 200))
            best = min(grid, key=lambda t: brier(_softmax(S, t).max(axis=1), correct))
            self.params_ = {"T": float(best)}
        elif self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            raw = _softmax(S, 1.0).max(axis=1)
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            ir.fit(raw, correct)
            self.params_ = {"model": ir}
        elif self.method == "platt":
            from sklearn.linear_model import LogisticRegression
            raw = _softmax(S, 1.0).max(axis=1).reshape(-1, 1)
            if len(set(correct.tolist())) < 2:
                self.params_ = {"model": None}
            else:
                lr = LogisticRegression(max_iter=1000).fit(raw, correct)
                self.params_ = {"model": lr}
        elif self.method == "dirichlet":
            # Full Dirichlet calibration is a linear map on log-probabilities. With 154 classes
            # and 375 spectra that map is not estimable, so the diagonal form is used: a
            # per-feature scale on the log-scores plus a bias. Documented, not silently reduced.
            from sklearn.linear_model import LogisticRegression
            L = np.log(_softmax(S, 1.0) + EPS)
            f = np.column_stack([L.max(axis=1), L.mean(axis=1), L.std(axis=1)])
            if len(set(correct.tolist())) < 2:
                self.params_ = {"model": None}
            else:
                self.params_ = {"model": LogisticRegression(max_iter=1000).fit(f, correct)}
        return self

    def transform(self, S: np.ndarray) -> np.ndarray:
        S = np.atleast_2d(np.asarray(S, float))
        if self.method == "uncalibrated":
            return _softmax(S, 1.0).max(axis=1)
        if self.method == "temperature":
            return _softmax(S, self.params_["T"]).max(axis=1)
        raw = _softmax(S, 1.0).max(axis=1)
        m = self.params_.get("model")
        if m is None:
            return raw
        if self.method == "isotonic":
            return np.clip(m.predict(raw), 0.0, 1.0)
        if self.method == "platt":
            return m.predict_proba(raw.reshape(-1, 1))[:, 1]
        L = np.log(_softmax(S, 1.0) + EPS)
        f = np.column_stack([L.max(axis=1), L.mean(axis=1), L.std(axis=1)])
        return m.predict_proba(f)[:, 1]


def sharpness(conf) -> float:
    """Spread of the reported confidences. A constant predictor scores zero.

    Reported because **ECE alone is minimised by predicting the base rate for everything**. On
    the first pass, Platt scaling won the ECE comparison (0.080) by mapping every spectrum to
    0.605 — precisely the Split A top-1 accuracy — while scoring the *worst* Brier of any method
    (0.242). A confidence that is the same for a correct and an incorrect answer is perfectly
    calibrated and completely useless.
    """
    return float(np.std(np.asarray(conf, float)))


def discrimination(conf, correct) -> float:
    """AUROC of the confidence separating correct from incorrect answers.

    The other half of what a confidence is for. Chance is 0.5 and a constant predictor sits
    exactly there.
    """
    from scipy.stats import rankdata
    conf, correct = np.asarray(conf, float), np.asarray(correct, bool)
    if correct.all() or (~correct).all():
        return float("nan")
    r = rankdata(conf)
    n1, n0 = correct.sum(), (~correct).sum()
    return float((r[correct].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def benchmark(S_tr, c_tr, S_te, c_te, n_bins: int = 10) -> "pd.DataFrame":
    """Fit each method on the training split, score it on the held-out split.

    Ranked by **Brier**, which is a strictly proper scoring rule and therefore decomposes into
    calibration *and* refinement. ECE, sharpness and discrimination are all reported alongside so
    the trade each method makes is visible rather than implied by the ranking.
    """
    import pandas as pd
    rows = []
    for m in METHODS:
        cal = Calibrator(m).fit(S_tr, c_tr)
        p = cal.transform(S_te)
        rows.append({"method": m, "ece": expected_calibration_error(p, c_te, n_bins),
                     "mce": maximum_calibration_error(p, c_te, n_bins),
                     "brier": brier(p, c_te), "sharpness": sharpness(p),
                     "discrimination": discrimination(p, c_te),
                     "mean_confidence": float(p.mean()),
                     "accuracy": float(np.mean(c_te)),
                     "overconfidence": float(p.mean() - np.mean(c_te))})
    return pd.DataFrame(rows).sort_values("brier").reset_index(drop=True)
