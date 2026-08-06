"""GAIRA V7 — Phase 06: held-out chemistry novelty.

**Not** cross-modality OOD. Every spectrum here is pure Raman from the frozen corpus; what is
withheld is an entire *chemistry class*. The question is narrow and answerable: when the atlas
has never seen sulfur/thiol cofactors at all, can the engine say so — or does it confidently
report the nearest chemistry it does know?

Phase 05's rejection experiment used corrupted and structureless spectra, four of six drawn
from the same perturbation module as its robustness study. That measured rejection of *degraded*
spectra. This measures rejection of *unrepresented chemistry*, which is the harder and more
useful question.
"""
from __future__ import annotations

import numpy as np

from .registry import CLASS_ORDER

EPS = 1e-12


def novelty_channels(E: np.ndarray, A: np.ndarray | None = None,
                     diag: dict | None = None) -> dict:
    """Evidence-side signals that a query is not represented. Signs declared here, not later."""
    E = np.atleast_2d(np.clip(E, 0.0, None))
    S = np.sort(E, axis=1)
    P = E / (E.sum(axis=1, keepdims=True) + EPS)
    ent = -(np.where(P > 0, P * np.log(P + EPS), 0.0)).sum(axis=1) / np.log(E.shape[1])
    out = {
        "max_evidence": S[:, -1],                  # lower  → more novel
        "support_margin": S[:, -1] - S[:, -2],     # lower  → more novel
        "entropy": ent,                            # higher → more novel
        "total_evidence": E.sum(axis=1),           # lower  → more novel
    }
    if diag is not None:
        out["explained_variance"] = np.asarray(diag["explained_variance"], float)
        out["residual_fraction"] = np.asarray(diag["residual_fraction"], float)
    return out


SIGN = {"max_evidence": -1, "support_margin": -1, "entropy": +1, "total_evidence": -1,
        "explained_variance": -1, "residual_fraction": +1}


def rejection_score(chan: dict, ref_chan: dict) -> np.ndarray:
    """Z-scored against the in-domain reference distribution, then averaged. Higher = novel."""
    tot, n = None, 0
    for k, v in chan.items():
        if k not in ref_chan:
            continue
        mu, sd = float(np.mean(ref_chan[k])), float(np.std(ref_chan[k])) + EPS
        z = SIGN[k] * (np.asarray(v, float) - mu) / sd
        tot = z if tot is None else tot + z
        n += 1
    return tot / max(n, 1)


def auroc(scores, is_novel) -> float:
    from scipy.stats import rankdata
    s, yv = np.asarray(scores, float), np.asarray(is_novel, bool)
    if yv.all() or (~yv).all():
        return float("nan")
    r = rankdata(s)
    n1, n0 = yv.sum(), (~yv).sum()
    return float((r[yv].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def holdout_class(A, y, cls, folds, held: str, fit_fn, predict_fn, cfg) -> dict:
    """Remove one chemistry class entirely, then query its spectra.

    The class is absent from the reference bank, the prototypes, any model fitting and the
    calibrator. The remaining corpus is the in-domain control, evaluated on its own frozen folds
    so the two populations are scored the same way.
    """
    keep = cls != held
    if keep.sum() < 30 or (~keep).sum() == 0:
        return {"held_class": held, "usable": False}
    model = fit_fn(A[keep], y[keep], cls[keep], cfg)
    E_novel = predict_fn(model, A[~keep])
    # In-domain control: held-out-fold predictions among the retained classes only, so no
    # spectrum is scored against a bank containing its own molecule.
    E_in = np.zeros((int(keep.sum()), len(CLASS_ORDER)))
    kept_idx = np.where(keep)[0]
    for f in sorted(set(folds[keep])):
        te = folds[kept_idx] == f
        tr = ~te
        if te.sum() == 0 or tr.sum() < 20:
            continue
        m = fit_fn(A[kept_idx][tr], y[kept_idx][tr], cls[kept_idx][tr], cfg)
        E_in[te] = predict_fn(m, A[kept_idx][te])
    ch_in = novelty_channels(E_in)
    ch_no = novelty_channels(E_novel)
    s_in = rejection_score(ch_in, ch_in)
    s_no = rejection_score(ch_no, ch_in)
    order = np.argsort(-E_novel, axis=1)
    nearest = [CLASS_ORDER[int(j)] for j in order[:, 0]]
    thr = float(np.quantile(s_in, 0.95))
    per_ch = {k: auroc(np.concatenate([SIGN[k] * ch_in[k], SIGN[k] * ch_no[k]]),
                       np.concatenate([np.zeros(len(s_in)), np.ones(len(s_no))]))
              for k in ch_in}
    return {
        "held_class": held, "usable": True,
        "n_novel_spectra": int((~keep).sum()),
        "n_novel_molecules": int(len(set(y[~keep].tolist()))),
        "joint_auroc": auroc(np.concatenate([s_in, s_no]),
                             np.concatenate([np.zeros(len(s_in)), np.ones(len(s_no))])),
        "per_channel_auroc": per_ch,
        "mean_max_evidence_novel": float(ch_no["max_evidence"].mean()),
        "mean_max_evidence_in_domain": float(ch_in["max_evidence"].mean()),
        "mean_entropy_novel": float(ch_no["entropy"].mean()),
        "mean_entropy_in_domain": float(ch_in["entropy"].mean()),
        "mean_margin_novel": float(ch_no["support_margin"].mean()),
        "mean_margin_in_domain": float(ch_in["support_margin"].mean()),
        "threshold_at_95pct_in_domain": thr,
        "abstain_rate_on_novel": float(np.mean(s_no > thr)),
        "false_abstain_rate_in_domain": float(np.mean(s_in > thr)),
        "nearest_represented_classes": {k: int(v) for k, v in
                                        zip(*np.unique(nearest, return_counts=True))},
    }
