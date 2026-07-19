"""GAIRA Raman Reference Atlas v0.1 — Perturbation Response Audit library.

Closes the loop between the Component Audit (what each latent Raman motif is) and
the Spike Validation (that perturbations move through the atlas). Everything here
reads the FROZEN atlas and the already-computed projections; nothing refits,
reweights or reinterprets the atlas.

Vocabulary discipline (enforced in the report):
  * a COMPONENT is a mathematical latent Raman motif (an NMF basis vector);
  * its THEME is a tentative post-hoc interpretation from the Component Audit;
  * a RESPONSE is the measured change in component activation under perturbation.
A response that matches a theme corroborates it; a mismatch is reported, not hidden.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress
from scipy.optimize import curve_fit

REPO = Path("/Users/surajpg/projects/GAIRA")
FROZEN = REPO / "results/v5_rebuild/foundation/artifacts"
AUDIT = REPO / "results/v5_rebuild/reference_atlas_audit/tables"
SPIKE = REPO / "results/v5_rebuild/spike_validation"
K = 24


# ───────────────────────── frozen context ─────────────────────────
def load_atlas_context():
    man = json.loads((FROZEN / "manifold.json").read_text())
    inv = pd.read_csv(AUDIT / "p1_component_inventory.csv")
    themes = dict(zip(inv.component, inv.primary_interpretation))
    theme_conf = dict(zip(inv.component, inv.confidence))
    theme_class = dict(zip(inv.component, inv.dominant_class))
    stability = dict(zip(inv.component, inv.bootstrap_stability))
    axes = man.get("axes", [])
    comp_to_axis = {}
    for ax in axes:
        for c in ax["components"]:
            comp_to_axis[c] = (ax["axis"], ax["tentative_theme"])
    return {"fingerprint": man["fingerprint"], "themes": themes, "theme_conf": theme_conf,
            "theme_class": theme_class, "stability": stability, "axes": axes,
            "comp_to_axis": comp_to_axis, "inventory": inv}


def load_component_reference_loadings():
    """For each component, the reference analytes that load it most (from the
    Component Audit P2 table). This is the component's ACTUAL chemical identity,
    independent of its coarse theme label — which for low-purity components can be
    wrong even though the component is chemically real."""
    comp = pd.read_csv(AUDIT / "p2_full_analyte_composition.csv")
    top = {}
    for j in range(K):
        s = comp[comp.component == j].nlargest(8, "contribution_pct")
        top[j] = list(zip(s.analyte, s.contribution_pct))
    return top


def component_encodes(analyte, comp_loadings, comp, tol_syn=True):
    """True if `analyte` is among the top reference analytes that load `comp`
    (a direct, label-independent identity test)."""
    import sys
    sys.path.insert(0, str(REPO / "results/v5_rebuild/reference_atlas_audit/code"))
    from run_confusability import canon
    a = canon(analyte) if tol_syn else analyte.lower()
    for name, _ in comp_loadings.get(comp, []):
        if (canon(name) if tol_syn else name.lower()) == a:
            return True
    return False


def load_projection(dataset):
    """Cached 24-component projection of a perturbation dataset (from spike study)."""
    p = SPIKE / f"tables/phase3_projection_{dataset}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    Z = df[[f"c{j}" for j in range(K)]].values
    meta = df.drop(columns=[f"c{j}" for j in range(K)])
    return Z, meta, df


DATASETS = ["ils_adenine", "ergothioneine", "uricase", "pure_sers", "spiked_serum",
            "serum_baseline", "isotopic"]


# ───────────────────────── helpers ─────────────────────────
def _unit(X):
    X = np.nan_to_num(np.atleast_2d(X))
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def norm_entropy(p):
    p = np.abs(np.asarray(p, float)); p = p[p > 1e-12]
    if p.size <= 1:
        return 0.0
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)) / np.log(len(p)))


# ───────────────────────── Part 1: component dose-response ─────────────────────────
def component_dose_response(Z, concs, comp):
    """Activation of one component vs perturbation level, with dose-model fits."""
    concs = np.asarray(concs, float)
    keys = np.array(sorted(pd.unique(concs)))
    mean = np.array([np.nan_to_num(Z[concs == k, comp]).mean() for k in keys])
    sem = np.array([np.nan_to_num(Z[concs == k, comp]).std() /
                    max(1, np.sqrt((concs == k).sum())) for k in keys])
    base = mean[0]
    d = mean - base
    rho, p = (spearmanr(keys, mean) if len(keys) > 2 else (np.nan, np.nan))
    eff = float(d[-1] / (np.nanmean(sem) + 1e-9))
    out = {"component": comp, "levels": keys.tolist(), "mean": mean.tolist(),
           "sem": sem.tolist(), "delta_final": float(d[-1]), "effect_size": eff,
           "spearman_rho": float(rho), "spearman_p": float(p),
           "direction": "up" if d[-1] > 0 else ("down" if d[-1] < 0 else "flat")}
    # dose fits on |activation change|
    ok = np.isfinite(keys) & np.isfinite(mean)
    if ok.sum() >= 4 and np.ptp(mean[ok]) > 1e-9:
        c, y = keys[ok], mean[ok]
        lr = linregress(c, y); out["linear_r2"] = float(lr.rvalue ** 2)
        try:
            f = lambda x, a, kk, b: a * x / (kk + x) + b
            popt, _ = curve_fit(f, c, y, p0=[np.ptp(y), np.median(c[c > 0]) if (c > 0).any() else 1, y[0]],
                                maxfev=8000)
            ss = 1 - np.sum((y - f(c, *popt)) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12)
            out["saturating_r2"] = float(ss); out["saturating_K"] = float(popt[1])
        except Exception:
            pass
    return out


def responsive_components(Z, concs, rho_thr=0.7, eff_thr=1.0):
    """Which components genuinely track the perturbation."""
    rows = []
    for j in range(K):
        r = component_dose_response(Z, concs, j)
        r["responsive"] = bool(abs(r["spearman_rho"]) >= rho_thr and abs(r["effect_size"]) >= eff_thr)
        rows.append(r)
    return pd.DataFrame(rows)


# ───────────────────────── Part 2: response fingerprint ─────────────────────────
def response_fingerprint(Z_treat, Z_control):
    """Δ(component activation) of a treatment relative to its control, with a
    replicate-bootstrap CI and a per-component effect size."""
    a = np.nan_to_num(Z_treat); b = np.nan_to_num(Z_control).mean(axis=0)
    d = a.mean(axis=0) - b
    sd = a.std(axis=0) + 1e-9
    eff = d / sd
    rng = np.random.default_rng(0)
    boots = np.array([a[rng.integers(0, len(a), len(a))].mean(0) - b for _ in range(500)])
    lo, hi = np.percentile(boots, 2.5, 0), np.percentile(boots, 97.5, 0)
    sig = (lo > 0) | (hi < 0)
    return {"delta": d, "effect_size": eff, "ci_lo": lo, "ci_hi": hi,
            "significant": sig, "n": len(a),
            "top_up": [int(j) for j in np.argsort(-d)[:5]],
            "top_down": [int(j) for j in np.argsort(d)[:5]],
            "entropy": norm_entropy(np.abs(d)),
            "n_significant": int(sig.sum())}


# ───────────────────────── Part 4: component specificity ─────────────────────────
def component_specificity(fingerprints, themes, theme_class):
    """For each component, how many analytes / classes activate it, and how
    concentrated the activation is (Gini): specific vs generic."""
    F = np.vstack([fp["delta"] for fp in fingerprints.values()])
    analytes = list(fingerprints.keys())
    classes = np.array([_analyte_class(a) for a in analytes])
    rows = []
    for j in range(K):
        col = F[:, j]
        up = np.where(col > 0)[0]; down = np.where(col < 0)[0]
        pos = np.clip(col, 0, None)
        g = _gini(pos) if pos.sum() > 0 else 0.0
        n_cls_up = len(set(classes[up])) if len(up) else 0
        rows.append({"component": j, "theme": themes.get(j), "theme_class": theme_class.get(j),
                     "n_activators": int((col > 0).sum()), "n_suppressors": int((col < 0).sum()),
                     "n_classes_activating": n_cls_up,
                     "top_activators": [analytes[i] for i in up[np.argsort(-col[up])][:5]],
                     "activation_gini": float(g),
                     "specificity": "specific" if (g >= 0.6 and n_cls_up <= 2)
                                    else ("generic" if n_cls_up >= 4 else "intermediate")})
    return pd.DataFrame(rows).sort_values("activation_gini", ascending=False)


def _gini(v):
    v = np.sort(np.abs(np.asarray(v, float))); n = len(v); c = np.cumsum(v)
    return float((n + 1 - 2 * (c / (c[-1] + 1e-12)).sum()) / n) if c[-1] > 0 else 0.0


_CLASS_CACHE = {}


def _analyte_class(a):
    if a not in _CLASS_CACHE:
        import sys
        sys.path.insert(0, str(REPO / "results/v5_rebuild/reference_atlas_audit/code"))
        import atlas_audit as AA
        _CLASS_CACHE[a] = AA.molecular_class(a)
    return _CLASS_CACHE[a]


def analyte_family(a):
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from gaira.foundation.families_raman import family_of
    return family_of(a)


# ───────────────────────── Part 12: trajectory fingerprints ─────────────────────────
def trajectory_fingerprint(Z, concs):
    """Geometric summary of a dose trajectory for the trajectory library."""
    keys = np.array(sorted(pd.unique(np.asarray(concs, float))))
    M = np.vstack([np.nan_to_num(Z[np.asarray(concs, float) == k]).mean(0) for k in keys])
    steps = np.diff(M, axis=0)
    sn = np.linalg.norm(steps, axis=1)
    net = M[-1] - M[0]
    path = float(sn.sum())
    curv = []
    for i in range(len(steps) - 1):
        a, b = steps[i], steps[i + 1]
        curv.append(np.degrees(np.arccos(np.clip(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12), -1, 1))))
    # component turnover: how the ranking of active components changes end vs start
    r0 = set(np.argsort(-M[0])[:5]); r1 = set(np.argsort(-M[-1])[:5])
    turnover = 1 - len(r0 & r1) / 5
    return {"n_levels": len(keys), "path_length": path,
            "net_displacement": float(np.linalg.norm(net)),
            "straightness": float(np.linalg.norm(net) / (path + 1e-12)),
            "mean_curvature_deg": float(np.mean(curv)) if curv else np.nan,
            "component_turnover": float(turnover),
            "net_direction": net.tolist(),
            "response_entropy": norm_entropy(np.abs(net)),
            "dominant_components": [int(j) for j in np.argsort(-np.abs(net))[:5]]}


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
