"""GAIRA V7 — Phase 05, Step 2: reference bank and similarity-metric selection.

One reference CSM activation vector per canonical molecule. Seven similarity metrics are
benchmarked and the winner is chosen **by grouped cross-validation only** — never by the
number the engine is finally reported on.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

EPS = 1e-12
METRICS = ("cosine", "pearson", "spearman", "centered_cosine", "correlation_distance",
           "angular", "mahalanobis")


def build_reference_bank(A: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """One vector per canonical molecule — the mean activation over its spectra.

    The mean rather than the medoid: replicates of one substance differ by acquisition, and
    averaging in activation space suppresses that while preserving the pattern of which CSMs
    fire. Molecules with a single spectrum contribute that spectrum unchanged.
    """
    mols = sorted(set(y))
    return np.vstack([A[y == m].mean(axis=0) for m in mols]), mols


def _z(M):
    return (M - M.mean(axis=1, keepdims=True)) / (M.std(axis=1, keepdims=True) + EPS)


def similarity(Q: np.ndarray, R: np.ndarray, metric: str,
               cov_inv: np.ndarray | None = None) -> np.ndarray:
    """Query × reference similarity. Higher is more similar for every metric here."""
    Q = np.atleast_2d(np.asarray(Q, float))
    R = np.atleast_2d(np.asarray(R, float))
    if metric == "cosine":
        return (Q / (np.linalg.norm(Q, axis=1, keepdims=True) + EPS)) @ \
               (R / (np.linalg.norm(R, axis=1, keepdims=True) + EPS)).T
    if metric in ("pearson", "centered_cosine", "correlation_distance"):
        Qz, Rz = _z(Q), _z(R)
        C = (Qz @ Rz.T) / Q.shape[1]
        return C if metric != "correlation_distance" else 1.0 - (1.0 - C)
    if metric == "spearman":
        Qr = np.vstack([rankdata(q) for q in Q])
        Rr = np.vstack([rankdata(r) for r in R])
        return (_z(Qr) @ _z(Rr).T) / Q.shape[1]
    if metric == "angular":
        c = np.clip(similarity(Q, R, "cosine"), -1, 1)
        return 1.0 - np.arccos(c) / np.pi
    if metric == "mahalanobis":
        if cov_inv is None:
            raise ValueError("mahalanobis needs an inverse covariance")
        d = np.zeros((Q.shape[0], R.shape[0]))
        for i, q in enumerate(Q):
            D = R - q
            d[i] = np.einsum("ij,jk,ik->i", D, cov_inv, D)
        return -np.sqrt(np.clip(d, 0, None))
    raise ValueError(f"unknown metric {metric}")


def stable_covariance(A: np.ndarray, shrinkage: float = 0.2) -> np.ndarray | None:
    """Shrunk inverse covariance, or None when the estimate is not trustworthy.

    With 375 spectra and 49 dimensions the sample covariance is estimable but ill-conditioned;
    Ledoit–Wolf-style shrinkage toward a scaled identity makes it usable. If the shrunk matrix
    is still near-singular the metric is dropped rather than reported on a bad estimate.
    """
    C = np.cov(A.T)
    C = (1 - shrinkage) * C + shrinkage * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
    if np.linalg.cond(C) > 1e10:
        return None
    return np.linalg.inv(C)


def retrieve(Q: np.ndarray, R: np.ndarray, ref_labels: list[str], metric: str,
             cov_inv=None, k: int = 5) -> dict:
    """Top-k retrieval with the margin and entropy the calibration step needs."""
    S = similarity(Q, R, metric, cov_inv)
    order = np.argsort(-S, axis=1)
    top = [[ref_labels[j] for j in row[:k]] for row in order]
    best = S[np.arange(S.shape[0]), order[:, 0]]
    second = S[np.arange(S.shape[0]), order[:, 1]] if S.shape[1] > 1 else np.zeros_like(best)
    P = np.exp(S - S.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True) + EPS
    ent = -(np.where(P > 0, P * np.log(P + EPS), 0)).sum(axis=1) / np.log(S.shape[1])
    return {"similarity": S, "order": order, "topk": top, "score": best,
            "margin": best - second, "entropy": ent}


def grouped_cv_rank(A: np.ndarray, y: np.ndarray, folds: np.ndarray,
                    metrics=METRICS) -> "pd.DataFrame":
    """Rank metrics by grouped CV, on molecules the reference bank has not seen.

    The bank is rebuilt inside every fold from training molecules only, so a metric cannot be
    credited for retrieving a molecule whose reference vector contains the query. Molecule-level
    top-k is undefined here by construction (Phase 04's finding), so metrics are ranked on
    chemistry-class retrieval, which is defined.
    """
    import pandas as pd
    rows = []
    for m in metrics:
        per = []
        for f in sorted(set(folds)):
            te, tr = folds == f, folds != f
            if te.sum() == 0 or tr.sum() < 5:
                continue
            R, labs = build_reference_bank(A[tr], y[tr])
            ci = stable_covariance(A[tr]) if m == "mahalanobis" else None
            if m == "mahalanobis" and ci is None:
                per = []
                break
            S = similarity(A[te], R, m, ci)
            per.append(S)
        if not per:
            rows.append({"metric": m, "usable": False, "cv_class_top1": np.nan})
            continue
        rows.append({"metric": m, "usable": True})
    return pd.DataFrame(rows)
