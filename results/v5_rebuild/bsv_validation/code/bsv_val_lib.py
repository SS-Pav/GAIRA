"""GAIRA BSV Validation — analysis library (READ-ONLY on the frozen V6 engine).

Characterizes the Biochemical State Vector as a scientific coordinate system.
Every BSV is produced by the ACTUAL V6 pipeline (gaira.engine.GAIRAEngine); no
theme scores are computed by hand. The engine, atlas, ontology and weights are
frozen and only measured here.

All calibration inputs are Ag/Au-SERS -> OUT OF DOMAIN for a Raman atlas by
construction; that is a property to characterize, not a bug to fix.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr, kendalltau, linregress

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
from gaira.engine import GAIRAEngine
SPIKE = REPO / "results/v5_rebuild/spike_validation/tables"
K = 24


class Harness:
    """Drives cached frozen-atlas projections through the full V6 engine."""

    def __init__(self):
        self.eng = GAIRAEngine()
        self.bio = self.eng.builder.onto.biochemical_theme_ids
        self.all_themes = self.eng.builder.onto.theme_ids

    def infer_coords(self, coords24, domain="buffer"):
        return self.eng.infer(coordinates=np.asarray(coords24, float), domain=domain)

    def bsv_row(self, coords24, domain="buffer"):
        """One inference -> a flat record of themes/confidence/ood/components."""
        out = self.infer_coords(coords24, domain)
        b = out.bsv
        rec = {f"theme_{t}": b.composition[t] for t in self.all_themes}
        rec.update({f"conf_{t}": b.confidence[t] for t in self.bio})
        rec.update({f"coord_c{j}": float(b.component_coord[j]) for j in range(K)})
        rec["ood"] = b.ood_score
        rec["overall_confidence"] = b.overall_confidence
        rec["background_share"] = b.non_biochemical.get("background_matrix", 0.0)
        rec["unknown_share"] = b.non_biochemical.get("unknown_mixed", 0.0)
        return rec

    def project_dataset(self, name):
        df = pd.read_csv(SPIKE / f"phase3_projection_{name}.csv")
        Z = df[[f"c{j}" for j in range(K)]].values
        meta = df.drop(columns=[f"c{j}" for j in range(K)])
        return Z, meta

    def bsv_frame(self, Z, meta, domain="buffer"):
        rows = [self.bsv_row(Z[i], domain) for i in range(len(Z))]
        return pd.concat([meta.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


# ── monotonicity (Part 3) ──
def monotonicity(levels, scores):
    lv = np.asarray(levels, float); sc = np.asarray(scores, float)
    ok = np.isfinite(lv) & np.isfinite(sc)
    lv, sc = lv[ok], sc[ok]
    if len(np.unique(lv)) < 3 or sc.std() < 1e-12:
        return {"insufficient": True}
    out = {"spearman": float(spearmanr(lv, sc)[0]), "spearman_p": float(spearmanr(lv, sc)[1]),
           "pearson": float(pearsonr(lv, sc)[0]), "kendall": float(kendalltau(lv, sc)[0]),
           "dynamic_range": float(sc.max() - sc.min()),
           "effect_size": float((sc.max() - sc.min()) / (sc.std() + 1e-9))}
    # saturating vs linear on aggregated per-level means
    keys = np.array(sorted(np.unique(lv)))
    m = np.array([sc[lv == k].mean() for k in keys])
    lr = linregress(keys, m); out["linear_r2"] = float(lr.rvalue ** 2)
    try:
        from scipy.optimize import curve_fit
        f = lambda x, a, kk, b: a * x / (kk + x) + b
        popt, _ = curve_fit(f, keys, m, p0=[np.ptp(m), np.median(keys[keys > 0]) if (keys > 0).any() else 1, m[0]],
                            maxfev=8000)
        ss = 1 - np.sum((m - f(keys, *popt)) ** 2) / (np.sum((m - m.mean()) ** 2) + 1e-12)
        out["saturating_r2"] = float(ss); out["saturating_K"] = float(popt[1])
        out["best_model"] = "saturating" if ss > out["linear_r2"] else "linear"
    except Exception:
        out["best_model"] = "linear"
    return out


def permutation_p(levels, scores, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    lv = np.asarray(levels, float); sc = np.asarray(scores, float)
    obs = abs(spearmanr(lv, sc)[0])
    null = [abs(spearmanr(rng.permutation(lv), sc)[0]) for _ in range(n)]
    return float((np.sum(np.array(null) >= obs) + 1) / (n + 1))


# ── cross-talk / specificity (Part 4) ──
def crosstalk_row(frame, level_col, themes):
    """Spearman of each theme vs the dose variable: target vs off-target movement."""
    return {t: float(spearmanr(frame[level_col].astype(float), frame[f"theme_{t}"])[0])
            if frame[f"theme_{t}"].std() > 1e-12 else 0.0 for t in themes}


# ── replicate stability (Part 8) ──
def icc(groups):
    """ICC(1) from a list of per-group replicate arrays (one-way random effects)."""
    groups = [np.asarray(g, float) for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return np.nan
    all_v = np.concatenate(groups); grand = all_v.mean(); n = len(groups)
    k = np.mean([len(g) for g in groups])
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    ms_b = ss_between / (n - 1); ms_w = ss_within / (len(all_v) - n + 1e-9)
    denom = ms_b + (k - 1) * ms_w
    return float((ms_b - ms_w) / denom) if denom > 1e-12 else np.nan


def cv(x):
    x = np.asarray(x, float); m = x.mean()
    return float(x.std() / (abs(m) + 1e-9))


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
