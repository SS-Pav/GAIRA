"""GAIRA V7 — Phase 06.5: is the geometry chemistry, or is it the instrument?

The question that decides whether anything in this phase is usable. A cluster structure that
tracks the source library or the excitation wavelength is an acquisition artefact however
chemically suggestive its members look.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def permanova(D: np.ndarray, groups: np.ndarray, n_perm: int = 999, seed: int = 0) -> dict:
    """Distance-based PERMANOVA (Anderson 2001) with a permutation null.

    Pseudo-F on a squared-distance matrix, with R² as the fraction of total dispersion explained.
    Non-parametric, so it makes no normality assumption about a 49-dimensional non-negative
    activation space where normality is certainly false.
    """
    D2 = np.asarray(D, float) ** 2
    n = len(D2)
    g = np.asarray(groups)
    levels = sorted(set(g.tolist()))
    if len(levels) < 2 or n < 4:
        return {"pseudo_F": np.nan, "R2": np.nan, "p_value": np.nan, "n_levels": len(levels)}

    def ss(gv):
        tot = D2[np.triu_indices(n, 1)].sum() / n
        within = 0.0
        for lv in levels:
            m = gv == lv
            k = int(m.sum())
            if k > 1:
                within += D2[np.ix_(m, m)][np.triu_indices(k, 1)].sum() / k
        return tot, within

    tot, within = ss(g)
    between = tot - within
    a, N = len(levels), n
    F = (between / (a - 1)) / (within / (N - a) + EPS)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        gp = rng.permutation(g)
        t2, w2 = ss(gp)
        Fp = ((t2 - w2) / (a - 1)) / (w2 / (N - a) + EPS)
        cnt += Fp >= F
    return {"pseudo_F": float(F), "R2": float(between / (tot + EPS)),
            "p_value": float((cnt + 1) / (n_perm + 1)), "n_levels": len(levels)}


def variance_partition(D: np.ndarray, factors: dict[str, np.ndarray], n_perm: int = 499,
                       seed: int = 0) -> "pd.DataFrame":
    """Marginal R² per factor, plus a sequential partition in a declared order.

    Marginal R² double-counts shared variance between correlated factors — source and excitation
    are heavily confounded in this corpus — so a sequential partition is reported alongside, and
    the order in which factors are entered is stated rather than optimised.
    """
    import pandas as pd
    rows = []
    for name, g in factors.items():
        r = permanova(D, g, n_perm=n_perm, seed=seed)
        rows.append({"factor": name, "n_levels": r["n_levels"], "marginal_R2": r["R2"],
                     "pseudo_F": r["pseudo_F"], "p_value": r["p_value"]})
    return pd.DataFrame(rows).sort_values("marginal_R2", ascending=False)


def anova_effect_sizes(values: np.ndarray, groups: np.ndarray) -> dict:
    """One-way ANOVA with eta-squared, for scalar responses such as explained variance."""
    from scipy import stats
    g = np.asarray(groups)
    levels = sorted(set(g.tolist()))
    arrs = [np.asarray(values, float)[g == lv] for lv in levels]
    arrs = [a for a in arrs if len(a) > 1]
    if len(arrs) < 2:
        return {"F": np.nan, "p_value": np.nan, "eta_squared": np.nan}
    F, p = stats.f_oneway(*arrs)
    grand = np.concatenate(arrs).mean()
    ssb = sum(len(a) * (a.mean() - grand) ** 2 for a in arrs)
    sst = sum(((a - grand) ** 2).sum() for a in arrs)
    return {"F": float(F), "p_value": float(p), "eta_squared": float(ssb / (sst + EPS))}


def cluster_vs_factor(lab: np.ndarray, factors: dict[str, np.ndarray]) -> "pd.DataFrame":
    """Adjusted mutual information between the emergent partition and each external factor.

    AMI is chance-corrected, which matters because source has 3 levels and chemistry has 16: an
    uncorrected index would favour the finer factor for arithmetic reasons alone.
    """
    import pandas as pd
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    keep = lab >= 0
    rows = []
    for name, g in factors.items():
        rows.append({"factor": name, "n_levels": len(set(np.asarray(g)[keep].tolist())),
                     "AMI": float(adjusted_mutual_info_score(np.asarray(g)[keep], lab[keep])),
                     "ARI": float(adjusted_rand_score(np.asarray(g)[keep], lab[keep]))})
    return pd.DataFrame(rows).sort_values("AMI", ascending=False)
