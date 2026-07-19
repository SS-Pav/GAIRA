"""Modality / source leakage probes with grouped CV (Phase 2 §12–13).

Can a simple classifier read modality or source off the representation? High
balanced accuracy vs a naive (majority / prior) baseline = nuisance leakage.
Splits are GROUPED by analyte so no analyte appears in both train and test, and
technical replicates never cross the split.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score


def grouped_leakage(X, target, groups, n_splits=5, seed=0):
    """target: labels to predict (modality/source); groups: analyte (split unit).
    Returns balanced accuracy vs majority + prior baselines."""
    target = np.asarray(target); groups = np.asarray(groups)
    classes, y = np.unique(target, return_inverse=True)
    if len(classes) < 2:
        return {"skipped": "single class", "n_classes": len(classes)}
    n_splits = min(n_splits, np.min(np.bincount(y)), len(np.unique(groups)))
    n_splits = max(2, int(n_splits))
    sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, te in sgk.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        accs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    prior = float(np.mean(np.bincount(y) / len(y)))  # balanced-acc of random = 1/n_classes
    return {"classes": classes.tolist(), "n_classes": len(classes),
            "balanced_accuracy_mean": float(np.mean(accs)) if accs else None,
            "balanced_accuracy_std": float(np.std(accs)) if accs else None,
            "chance_balanced_accuracy": 1.0 / len(classes),
            "majority_fraction": float(np.max(np.bincount(y) / len(y))),
            "n_splits_used": len(accs)}
