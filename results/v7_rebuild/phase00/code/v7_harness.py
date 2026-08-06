"""GAIRA V7 Phase 00 — the FROZEN evaluation harness.

The yardstick is fixed here, before anything is built with it. Phase 07 will compare V7
against the V5 control using exactly these functions on exactly the Phase-00 splits.

The metric set and the statistical procedures are adopted from the V6.3 ontology
revalidation, which is the strongest methodology this project has produced. Adopting it
wholesale rather than reinventing it is what makes the Phase-07 comparison like-for-like
with published V5/V6 numbers.

Metrics       leave-one-out retrieval@1, precision@5, MRR, mean first rank,
              Cohen's kappa, MCC, chance-adjusted accuracy, macro-P/R/F1,
              balanced accuracy
Null          >= 12 size-matched RANDOM ontologies (V6.3 used 12) — this is what
              separates real chemistry from the mechanical gain of having fewer classes
CI            bootstrap over canonical analytes, 95%
Paired test   exact McNemar + permutation, with Cohen's g and odds ratio
Calibration   expected calibration error, fixed 10-bin scheme
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HARNESS_VERSION = "v7_harness_v1"
EPS = 1e-12
N_PERM = 1000
N_BOOT = 4000
N_RANDOM_ONTOLOGIES = 12
ECE_BINS = 10
SEED = 0


def unit(X) -> np.ndarray:
    X = np.nan_to_num(np.atleast_2d(np.asarray(X, float)))
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + EPS)


def nn_hits(X, labels, k: int = 5):
    """Leave-one-out nearest-neighbour retrieval in cosine geometry."""
    U = unit(X)
    S = U @ U.T
    np.fill_diagonal(S, -np.inf)
    lab = np.asarray(labels)
    n = len(lab)
    hit = np.zeros(n, bool)
    rank = np.full(n, np.nan)
    pk = np.zeros(n)
    nn = np.zeros(n, int)
    for i in range(n):
        o = np.argsort(-S[i])
        o = o[np.isfinite(S[i][o])]
        nn[i] = o[0]
        rel = (lab[o] == lab[i])
        hit[i] = bool(rel[0])
        if rel.any():
            rank[i] = int(np.argmax(rel)) + 1
        pk[i] = rel[:min(k, len(rel))].mean()
    return hit, rank, pk, nn


def perm_null(X, labels, n_perm: int = N_PERM, seed: int = SEED):
    """Chance level of THIS label distribution given THIS geometry (labels permuted)."""
    rng = np.random.default_rng(seed)
    U = unit(X)
    S = U @ U.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argmax(S, 1)
    lab = np.asarray(labels)
    vals = [float((lab[p][nn] == lab[p]).mean())
            for p in (rng.permutation(len(lab)) for _ in range(n_perm))]
    a = np.array(vals)
    return float(a.mean()), float(np.percentile(a, 95))


def random_ontologies(labels, n: int = N_RANDOM_ONTOLOGIES, seed: int = SEED):
    """Size-matched random ontologies: same class-size histogram, random membership.

    Without this control, a coarser ontology looks better purely because it has fewer
    classes. The gain BEYOND this control is the number that means something.
    """
    rng = np.random.default_rng(seed)
    lab = np.asarray(labels)
    sizes = pd.Series(lab).value_counts().tolist()
    out = []
    for _ in range(n):
        idx = rng.permutation(len(lab))
        rand = np.empty(len(lab), dtype=object)
        pos = 0
        for c, s in enumerate(sizes):
            rand[idx[pos:pos + s]] = f"rand_{c}"
            pos += s
        out.append(rand.astype(str))
    return out


def kappa_mcc(y_true, y_pred):
    labs = sorted(set(map(str, y_true)) | set(map(str, y_pred)))
    idx = {l: i for i, l in enumerate(labs)}
    yt = np.array([idx[str(v)] for v in y_true])
    yp = np.array([idx[str(v)] for v in y_pred])
    K = len(labs)
    C = np.zeros((K, K))
    for a, b in zip(yt, yp):
        C[a, b] += 1
    n = C.sum()
    po = np.trace(C) / n
    pe = float((C.sum(0) * C.sum(1)).sum()) / (n * n)
    kap = (po - pe) / (1 - pe + EPS)
    t, p, c = C.sum(1), C.sum(0), np.trace(C)
    num = c * n - float(t @ p)
    den = np.sqrt(max((n * n - float(p @ p)) * (n * n - float(t @ t)), 0.0))
    return float(kap), float(num / (den + EPS))


def prf(y_true, y_pred):
    labs = sorted(set(map(str, y_true)))
    P, R, F = [], [], []
    for l in labs:
        yt = np.array([str(v) == l for v in y_true])
        yp = np.array([str(v) == l for v in y_pred])
        tp, fp, fn = int((yp & yt).sum()), int((yp & ~yt).sum()), int((~yp & yt).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        P.append(p)
        R.append(r)
        F.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return np.array(P), np.array(R), np.array(F)


def boot_ci(v, n_boot: int = N_BOOT, seed: int = SEED):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    n = len(v)
    b = [v[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def mcnemar(hit_a, hit_b, n_perm: int = N_PERM, seed: int = SEED):
    """Exact McNemar + permutation on paired per-analyte correctness."""
    from math import comb
    a = np.asarray(hit_a, bool)
    b = np.asarray(hit_b, bool)
    b01 = int((a & ~b).sum())      # a only
    b10 = int((~a & b).sum())      # b only
    n = b01 + b10
    if n == 0:
        p_exact = 1.0
    else:
        k = min(b01, b10)
        tail = sum(comb(n, i) for i in range(0, k + 1))
        p_exact = float(min(1.0, 2.0 * tail / (2 ** n)))
    rng = np.random.default_rng(seed)
    obs = abs(b.mean() - a.mean())
    d = (b.astype(int) - a.astype(int))
    perm = [abs(float((d * rng.choice([-1, 1], size=len(d))).mean())) for _ in range(n_perm)]
    p_perm = float((np.array(perm) >= obs - 1e-15).mean())
    g = abs(b01 - b10) / (2.0 * n) if n else 0.0
    odds = (b10 / b01) if b01 else (float("inf") if b10 else 1.0)
    return {"n": int(len(a)), "a_correct": int(a.sum()), "b_correct": int(b.sum()),
            "b_only_a": b01, "b_only_b": b10,
            "delta_accuracy": round(float(b.mean() - a.mean()), 4),
            "mcnemar_p": round(p_exact, 6), "permutation_p": round(p_perm, 6),
            "cohens_g": round(float(g), 4),
            "odds_ratio": (round(float(odds), 4) if np.isfinite(odds) else None),
            "significant_at_0.05": bool(p_exact < 0.05)}


def ece(conf, correct, n_bins: int = ECE_BINS) -> float:
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        tot += (m.sum() / len(conf)) * abs(correct[m].mean() - conf[m].mean())
    return float(tot)


def score(X, labels, level: str, tag: str, n_perm: int = N_PERM) -> dict:
    """The frozen metric row for one representation × one label set."""
    hit, rank, pk, nn = nn_hits(X, labels)
    nm, n95 = perm_null(X, labels, n_perm=n_perm)
    y_true = np.asarray(labels)
    y_pred = y_true[nn]
    kap, mcc = kappa_mcc(y_true, y_pred)
    Pm, Rm, Fm = prf(y_true, y_pred)
    lo, hi = boot_ci(hit.astype(float))
    return {
        "harness_version": HARNESS_VERSION, "representation": tag, "level": level,
        "dim": int(np.atleast_2d(X).shape[1]), "n_items": int(len(y_true)),
        "n_classes": int(len(set(map(str, y_true)))),
        "retrieval_p1": round(float(hit.mean()), 4),
        "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
        "precision_at_5": round(float(pk.mean()), 4),
        "mrr": round(float(np.nanmean(1.0 / rank)), 4),
        "mean_first_rank": round(float(np.nanmean(rank)), 3),
        "null_p1": round(nm, 4), "null_p95": round(n95, 4),
        "kappa": round(kap, 4), "mcc": round(mcc, 4),
        "chance_adjusted": round((float(hit.mean()) - nm) / (1 - nm + EPS), 4),
        "macro_precision": round(float(Pm.mean()), 4),
        "macro_recall": round(float(Rm.mean()), 4),
        "macro_f1": round(float(Fm.mean()), 4),
        "balanced_accuracy": round(float(Rm.mean()), 4),
    }, hit


def random_control(X, labels, level: str, n: int = N_RANDOM_ONTOLOGIES) -> dict:
    """Mean metrics over n size-matched random ontologies."""
    rows = []
    for i, rand in enumerate(random_ontologies(labels, n=n, seed=SEED)):
        r, _ = score(X, rand, level, f"random_{i}", n_perm=200)
        rows.append(r)
    df = pd.DataFrame(rows)
    num = df.select_dtypes("number").mean().round(4).to_dict()
    num |= {"harness_version": HARNESS_VERSION, "representation": f"random_mean_of_{n}",
            "level": level, "n_random_ontologies": n}
    return num
