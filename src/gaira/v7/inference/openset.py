"""GAIRA V7 — Phase 05, Step 4: rejecting spectra the frozen atlas cannot support.

**Raman only.** No cross-modality experiment appears in this phase. Phase 04's SERS out-of-domain
probe is removed from the canonical engine, not merely unreported: a Raman atlas evaluated on
Ag-SERS was answering a question this project does not ask.

Negatives here are synthetic — spectra corrupted past the point where the atlas should still
claim to recognise them, plus structured non-Raman signals. That makes the ROC a statement about
*evidence degradation*, which is what the channels actually measure.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12
CHANNELS = ("residual_fraction", "explained_variance", "top1_top2_margin",
            "activation_sparsity", "activation_entropy", "nearest_reference_distance",
            "centroid_distance", "ood_mahalanobis")
# Sign convention: +1 when a larger value means *more* out-of-domain.
CHANNEL_SIGN = {"residual_fraction": +1, "explained_variance": -1, "top1_top2_margin": -1,
                "activation_sparsity": -1, "activation_entropy": +1,
                "nearest_reference_distance": +1, "centroid_distance": +1,
                "ood_mahalanobis": +1}


def auroc(scores, is_ood) -> float:
    """Rank-based AUROC, ties handled by average rank."""
    from scipy.stats import rankdata
    s, y = np.asarray(scores, float), np.asarray(is_ood, bool)
    if y.all() or (~y).all():
        return float("nan")
    r = rankdata(s)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def channel_scores(A, diag, R_bank, cov_inv=None, ref_mean=None) -> dict:
    """All eight evidence channels for a batch of spectra.

    `ref_mean` is the **in-domain** activation mean. It must be supplied for the Mahalanobis
    channel to mean anything: centring a batch of negatives on its own mean asks how unusual
    each negative is *among negatives*, which is a different question and scored an inverted
    AUROC of 0.176 until it was fixed.
    """
    from .retrieval import similarity
    A = np.atleast_2d(A)
    S = similarity(A, R_bank, "cosine")
    order = np.argsort(-S, axis=1)
    best = S[np.arange(len(S)), order[:, 0]]
    second = S[np.arange(len(S)), order[:, 1]]
    centroid = R_bank.mean(axis=0)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + EPS)
    out = {
        "residual_fraction": np.asarray(diag["residual_fraction"], float),
        "explained_variance": np.asarray(diag["explained_variance"], float),
        "top1_top2_margin": best - second,
        "activation_sparsity": np.asarray(diag["component_sparsity"], float),
        "activation_entropy": np.asarray(diag["activation_entropy"], float),
        "nearest_reference_distance": 1.0 - best,
        "centroid_distance": np.linalg.norm(
            An - centroid / (np.linalg.norm(centroid) + EPS), axis=1),
    }
    if cov_inv is not None:
        D = A - (A.mean(axis=0) if ref_mean is None else np.asarray(ref_mean, float))
        out["ood_mahalanobis"] = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", D, cov_inv, D), 0, None))
    return out


def joint_score(chan: dict, ref_chan: dict, weights: dict | None = None) -> np.ndarray:
    """Combine channels after z-scoring each against the *in-domain* reference distribution.

    Standardising against in-domain statistics rather than the pooled batch is what makes the
    combination usable at inference on a single spectrum, where there is no batch to normalise
    against.
    """
    tot, n = None, 0
    for k, v in chan.items():
        if k not in ref_chan:
            continue
        mu, sd = float(np.mean(ref_chan[k])), float(np.std(ref_chan[k])) + EPS
        z = CHANNEL_SIGN[k] * (np.asarray(v, float) - mu) / sd
        w = 1.0 if weights is None else float(weights.get(k, 1.0))
        tot = z * w if tot is None else tot + z * w
        n += w
    return tot / max(n, EPS)


def evaluate(chan_in: dict, chan_out: dict, weights=None) -> "pd.DataFrame":
    """Per-channel and joint AUROC, in-domain vs synthetic out-of-domain."""
    import pandas as pd
    rows = []
    for k in chan_in:
        if k not in chan_out:
            continue
        s = np.concatenate([chan_in[k], chan_out[k]])
        y = np.concatenate([np.zeros(len(chan_in[k])), np.ones(len(chan_out[k]))])
        rows.append({"channel": k, "auroc": auroc(CHANNEL_SIGN[k] * s, y),
                     "in_domain_mean": float(np.mean(chan_in[k])),
                     "ood_mean": float(np.mean(chan_out[k]))})
    ji = joint_score(chan_in, chan_in, weights)
    jo = joint_score(chan_out, chan_in, weights)
    rows.append({"channel": "JOINT", "auroc": auroc(np.concatenate([ji, jo]),
                 np.concatenate([np.zeros(len(ji)), np.ones(len(jo))])),
                 "in_domain_mean": float(np.mean(ji)), "ood_mean": float(np.mean(jo))})
    return pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True)


def operating_point(s_in, s_out, target_tpr: float = 0.95) -> dict:
    """Threshold at a stated in-domain acceptance rate, with the rejection rate it buys."""
    thr = float(np.quantile(s_in, target_tpr))
    return {"threshold": thr, "in_domain_accept": float(np.mean(s_in <= thr)),
            "ood_reject": float(np.mean(s_out > thr)), "target_in_domain_accept": target_tpr}
