"""GAIRA V7 — Phase 08: retrieval metrics, calibration and significance.

Split A asks *where does the correct molecule rank*, and is defined only when the molecule is in
the bank. Split B asks *what chemistry is this*, and molecule top-k there is undefined rather
than zero — the distinction Phase 04 lacked and Phase 05 introduced.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def ranks(S: np.ndarray, labels: list[str], truth: np.ndarray) -> np.ndarray:
    """1-based rank of the true molecule. `len(labels)+1` when it is absent from the bank."""
    lab = np.asarray(labels)
    out = np.zeros(len(S), int)
    for i, row in enumerate(S):
        hit = np.where(lab[np.argsort(-row)] == truth[i])[0]
        out[i] = int(hit[0]) + 1 if len(hit) else len(lab) + 1
    return out


def split_a_metrics(rk: np.ndarray, n_bank: int) -> dict:
    rk = np.asarray(rk, float)
    return {"top1": float((rk <= 1).mean()), "top3": float((rk <= 3).mean()),
            "top5": float((rk <= 5).mean()), "top10": float((rk <= 10).mean()),
            "mrr": float((1.0 / rk).mean()),
            "ndcg5": float(np.mean(np.where(rk <= 5, 1.0 / np.log2(rk + 1), 0.0))),
            "mean_reciprocal_rank": float((1.0 / rk).mean()),
            "median_rank": float(np.median(rk)),
            "mean_rank": float(rk.mean()),
            "unretrievable": int((rk > n_bank).sum())}


def rank_distribution(rk, bins=(1, 2, 3, 5, 10, 25, 50, 100, 10 ** 6)) -> "pd.DataFrame":
    import pandas as pd
    rk = np.asarray(rk)
    rows, lo = [], 0
    for hi in bins:
        n = int(((rk > lo) & (rk <= hi)).sum())
        rows.append({"rank_upper": hi, "n": n, "share": n / len(rk)})
        lo = hi
    return pd.DataFrame(rows)


def chemistry_metrics(S, labels, class_of, truth_class) -> dict:
    """Split B: rank distinct chemistry classes by their best-scoring molecule."""
    rl = np.array([class_of[m] for m in labels])
    h1 = h3 = 0
    preds = []
    for i, row in enumerate(S):
        seen = []
        for j in np.argsort(-row):
            if rl[j] not in seen:
                seen.append(rl[j])
            if len(seen) >= 3:
                break
        preds.append(seen[0])
        h1 += truth_class[i] == seen[0]
        h3 += truth_class[i] in seen
    preds = np.array(preds)
    classes = sorted(set(np.asarray(truth_class).tolist()))
    f1 = []
    for c in classes:
        tp = int(((preds == c) & (truth_class == c)).sum())
        fp = int(((preds == c) & (truth_class != c)).sum())
        fn = int(((preds != c) & (truth_class == c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return {"chem_top1": h1 / len(S), "chem_top3": h3 / len(S),
            "chem_macro_f1": float(np.mean(f1)),
            "chem_balanced_accuracy": float(np.mean(
                [((preds == c) & (truth_class == c)).sum() / max((truth_class == c).sum(), 1)
                 for c in classes])),
            "predictions": preds}


def nearest_supported_analogue(S, labels, class_of, truth_class) -> dict:
    """Split B: when the molecule is absent, is the top hit at least the right chemistry?"""
    lab = np.asarray(labels)
    rl = np.array([class_of[m] for m in labels])
    top = lab[np.argmax(S, axis=1)]
    top_cls = rl[np.argmax(S, axis=1)]
    return {"analogue_class_correct": float((top_cls == truth_class).mean()),
            "top_analogue": top.tolist()}


# ── calibration ──────────────────────────────────────────────────────────────
def softmax_conf(S, T: float = 1.0) -> np.ndarray:
    Z = np.asarray(S, float) / max(T, 1e-6)
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return (E / (E.sum(axis=1, keepdims=True) + EPS)).max(axis=1)


def ece(conf, correct, n_bins: int = 10) -> float:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def brier(conf, correct) -> float:
    return float(np.mean((np.asarray(conf, float) - np.asarray(correct, float)) ** 2))


def log_loss_binary(conf, correct) -> float:
    p = np.clip(np.asarray(conf, float), 1e-9, 1 - 1e-9)
    c = np.asarray(correct, float)
    return float(-(c * np.log(p) + (1 - c) * np.log(1 - p)).mean())


def sharpness(conf) -> float:
    return float(np.std(conf))


def discrimination(conf, correct) -> float:
    from scipy.stats import rankdata
    conf, correct = np.asarray(conf, float), np.asarray(correct, bool)
    if correct.all() or (~correct).all():
        return float("nan")
    r = rankdata(conf)
    n1, n0 = correct.sum(), (~correct).sum()
    return float((r[correct].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def reliability(conf, correct, n_bins: int = 10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        xs.append((lo + hi) / 2)
        ys.append(float(correct[m].mean()) if m.any() else np.nan)
        ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def risk_coverage(conf, correct, n: int = 30) -> "pd.DataFrame":
    """Selective prediction: what accuracy is bought by answering less often."""
    import pandas as pd
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    rows = []
    for q in np.linspace(0.0, 0.95, n):
        t = np.quantile(conf, q)
        m = conf >= t
        if m.sum() < 5:
            break
        rows.append({"threshold": float(t), "coverage": float(m.mean()),
                     "accuracy": float(correct[m].mean()),
                     "risk": float(1.0 - correct[m].mean())})
    return pd.DataFrame(rows)


# ── significance ─────────────────────────────────────────────────────────────
def paired_test(a: np.ndarray, b: np.ndarray, y: np.ndarray, n_boot: int = 2000,
                seed: int = 0) -> dict:
    """McNemar plus a molecule-level bootstrap CI on the paired difference b − a.

    Bootstrapping molecules rather than spectra: replicates are not independent, and resampling
    them would narrow every interval for a reason that has nothing to do with the models.
    """
    from scipy.stats import binomtest
    a, b, y = np.asarray(a, bool), np.asarray(b, bool), np.asarray(y)
    b01, b10 = int((~a & b).sum()), int((a & ~b).sum())
    pv = float(binomtest(b01, b01 + b10, 0.5).pvalue) if (b01 + b10) else 1.0
    rng = np.random.default_rng(seed)
    mols = np.array(sorted(set(y.tolist())))
    idx_of = {m: np.where(y == m)[0] for m in mols}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(len(mols), len(mols), replace=True)
        ii = np.concatenate([idx_of[mols[p]] for p in pick])
        diffs.append(float(b[ii].mean() - a[ii].mean()))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"delta": float(b.mean() - a.mean()), "ci95": [float(lo), float(hi)],
            "mcnemar_b01": b01, "mcnemar_b10": b10, "p_value": pv,
            "significant": bool(pv < 0.05 and lo > 0)}


def paired_continuous(a: np.ndarray, b: np.ndarray, y: np.ndarray, n_boot: int = 2000,
                      seed: int = 0) -> dict:
    """Molecule-level bootstrap CI for a continuous paired difference such as MRR."""
    from scipy.stats import wilcoxon
    a, b, y = np.asarray(a, float), np.asarray(b, float), np.asarray(y)
    try:
        pv = float(wilcoxon(b, a).pvalue)
    except Exception:
        pv = 1.0
    rng = np.random.default_rng(seed)
    mols = np.array(sorted(set(y.tolist())))
    idx_of = {m: np.where(y == m)[0] for m in mols}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(len(mols), len(mols), replace=True)
        ii = np.concatenate([idx_of[mols[p]] for p in pick])
        diffs.append(float(b[ii].mean() - a[ii].mean()))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"delta": float(b.mean() - a.mean()), "ci95": [float(lo), float(hi)],
            "wilcoxon_p": pv, "significant": bool(pv < 0.05 and lo > 0)}
