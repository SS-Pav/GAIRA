"""Biological-studies analysis (Page 6).

Loads the committed V6 artifacts produced by tools/build_biological_v6.py (genuine
GAIRAEngine outputs — component coordinates -> BSV -> MSS -> OOD/confidence) and
computes group contrasts with the correct level of inference:

  - PATIENT-level datasets (diabetes): one unit per patient -> proper non-parametric
    tests, effect sizes, FDR and bootstrap CIs.
  - SPECTRUM-level datasets (covid, hcc): subject mapping undocumented -> descriptive
    contrast + a clearly-labelled spectrum-level permutation test, NOT patient-level
    inference (pseudoreplication is avoided by never claiming it).

Nothing is recomputed from raw spectra here; the artifacts already ARE the V6 output.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from scipy.stats import mannwhitneyu

from .engine_bridge import REPO

ART = Path(__file__).resolve().parents[1] / "biological_artifacts"

# study-specific contrast (group_a vs group_b), labelled verbatim
CONTRAST = {
    "covid_serum_raman": ("COVID", "Healthy"),
    "hcc_serum": ("HCC", "control"),
    "diabetes_plasma_ev_sers": ("Impact", "Strong-D"),
    "shine_ev_sers": ("D2", "D0"),               # later − earlier (timepoint); strata = dose
    "small2023_ev": ("c100", "c00"),             # probe-loading effect (characterization only)
}
# datasets whose contrast is a technical/probe effect, not biology
CHARACTERIZATION_ONLY = {"small2023_ev"}
# datasets with a paired secondary factor (strata) for longitudinal analysis
PAIRED = {"shine_ev_sers"}


def available():
    if not (ART / "manifest.json").exists():
        return {}
    return json.loads((ART / "manifest.json").read_text())["datasets"]


def load(dataset_id):
    p = ART / f"{dataset_id}.json"
    if not p.exists():
        return None
    a = json.loads(p.read_text())
    a["themes_mat"] = np.array([r["themes"] for r in a["records"]])
    a["motifs_mat"] = np.array([r["motifs"] for r in a["records"]])
    a["coord_mat"] = np.array([r["coord"] for r in a["records"]])
    a["group"] = np.array([r["group"] for r in a["records"]])
    a["ood"] = np.array([r["ood"] for r in a["records"]])
    a["conf"] = np.array([r["confidence"] for r in a["records"]])
    a["bg"] = np.array([r["background"] for r in a["records"]])
    return a


def _cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    return (gt - lt) / (len(x) * len(y))


def _bh_fdr(pvals):
    p = np.asarray(pvals); n = len(p)
    order = np.argsort(p); ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(q, 0, 1)
    return out


def group_contrast(art, seed=0):
    """Per biochemical-theme contrast a−b with MWU p, BH-FDR, Cliff's delta and a
    bootstrap CI on the mean difference. Unit = record (patient or spectrum)."""
    a, b = CONTRAST[art["dataset_id"]]
    themes = art["theme_ids"]
    bio = [i for i, t in enumerate(themes) if t not in ("background_matrix", "unknown_mixed")]
    Ta = art["themes_mat"][art["group"] == a]
    Tb = art["themes_mat"][art["group"] == b]
    rng = np.random.default_rng(seed)
    rows, pvals = [], []
    for i in bio:
        xa, xb = Ta[:, i], Tb[:, i]
        try:
            p = float(mannwhitneyu(xa, xb, alternative="two-sided")[1])
        except ValueError:
            p = 1.0
        boot = [rng.choice(xa, len(xa)).mean() - rng.choice(xb, len(xb)).mean() for _ in range(2000)]
        rows.append({"theme": themes[i], "delta": float(xa.mean() - xb.mean()),
                     "cliffs_delta": float(_cliffs_delta(xa, xb)), "p": p,
                     "ci_lo": float(np.percentile(boot, 2.5)),
                     "ci_hi": float(np.percentile(boot, 97.5))})
        pvals.append(p)
    q = _bh_fdr(pvals)
    for r, qi in zip(rows, q):
        r["q"] = float(qi); r["sig"] = qi < 0.05
    rows.sort(key=lambda r: -abs(r["delta"]))
    return {"a": a, "b": b, "na": int((art["group"] == a).sum()),
            "nb": int((art["group"] == b).sum()), "rows": rows}


def group_theme_means(art):
    """Absolute per-group theme means (for the atlas-position radar) — NOT cohort-mean
    normalised."""
    themes = art["theme_ids"]
    means = {}
    for g in art["groups"]:
        means[g] = art["themes_mat"][art["group"] == g].mean(0)
    return themes, means


def group_delta_axes(art):
    """Signed per-biochemical-theme composition delta (group_a − group_b) in radar-axis
    schema — the difference the absolute overlapping polygons cannot show."""
    a, b = CONTRAST[art["dataset_id"]]
    themes, means = group_theme_means(art)
    bio = [t for t in themes if t not in ("background_matrix", "unknown_mixed")]
    idx = {t: i for i, t in enumerate(themes)}
    axes = [{"theme": t, "delta": float(means[a][idx[t]] - means[b][idx[t]])} for t in bio]
    shared = max(abs(x["delta"]) for x in axes) if axes else 1.0
    return a, b, axes, shared


def group_radar_axes(art, group):
    """Radar-axis dicts (engine schema) for one group's absolute theme means."""
    themes, means = group_theme_means(art)
    bio = [t for t in themes if t not in ("background_matrix", "unknown_mixed")]
    m = means[group]; idx = {t: i for i, t in enumerate(themes)}
    return [{"theme": t, "score": float(m[idx[t]])} for t in bio]


def group_means_by(art, matrix_key):
    """Per-group mean of 'themes_mat' | 'motifs_mat' | 'coord_mat'."""
    return {g: art[matrix_key][art["group"] == g].mean(0) for g in art["groups"]}


def motif_contrast(art):
    """Per-MSS-motif signed difference (a−b) for biochemical motifs."""
    a, b = CONTRAST[art["dataset_id"]]
    means = group_means_by(art, "motifs_mat")
    return a, b, art["motif_ids"], means[a] - means[b]


def top_components_for(art, delta_vector, k=5):
    """Components whose per-group difference is largest (numerical provenance)."""
    a, b = CONTRAST[art["dataset_id"]]
    cm = group_means_by(art, "coord_mat")
    d = cm[a] - cm[b]
    order = np.argsort(-np.abs(d))[:k]
    return [(int(j), float(d[j])) for j in order]


def pca_2d(X):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    for i in range(2):
        if Vt[i][np.argmax(np.abs(Vt[i]))] < 0:
            Vt[i] = -Vt[i]
    return Xc @ Vt[:2].T, (S[:2] ** 2) / (S ** 2).sum()


def strata_values(art):
    return sorted({r.get("strata") for r in art["records"] if r.get("strata")})


def paired_timepoint_theme(art, theme_id):
    """For a longitudinal dataset (group = timepoint, strata = dose): per-stratum mean
    theme value at each timepoint. Returns {stratum: {group: mean}} — a paired,
    stratum-level view (NOT per-subject repeated measures; cohort-level)."""
    a, b = CONTRAST[art["dataset_id"]]
    ti = art["theme_ids"].index(theme_id)
    strata = np.array([r.get("strata") for r in art["records"]])
    out = {}
    for s in strata_values(art):
        row = {}
        for g in (b, a):                                    # baseline first
            m = (strata == s) & (art["group"] == g)
            if m.any():
                row[g] = float(art["themes_mat"][m][:, ti].mean())
        if len(row) == 2:
            out[s] = row
    return a, b, out


def heatmap_matrix(art, matrix_key="themes_mat", max_rows=80, seed=0):
    """Z-scored (per-theme) sample matrix for a VISUALIZATION heatmap only, with group
    labels. Subsamples rows for legibility. Z-scoring is for display, never inference."""
    rng = np.random.default_rng(seed)
    X = art[matrix_key]; g = art["group"]
    bio = [i for i, t in enumerate(art["theme_ids"] if matrix_key == "themes_mat" else art["motif_ids"])
           if t not in ("background_matrix", "unknown_mixed", "colloid_matrix_background")]
    labels = [(art["theme_ids"] if matrix_key == "themes_mat" else art["motif_ids"])[i] for i in bio]
    idx = np.arange(len(X))
    if len(X) > max_rows:                                   # stratified subsample
        keep = []
        for grp in np.unique(g):
            gi = idx[g == grp]
            keep.extend(rng.choice(gi, min(len(gi), max_rows // len(np.unique(g))), replace=False))
        idx = np.sort(keep)
    Z = X[np.ix_(idx, bio)].astype(float)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    strata = [art["records"][i].get("strata") for i in idx]
    has_strata = any(s is not None for s in strata)
    return Z, [g[i] for i in idx], labels, (strata if has_strata else None)


def distance_summary(art):
    """Within-group vs between-group BSV distance + effect-to-variability ratio.
    Answers: are group differences large relative to biological heterogeneity?"""
    a, b = CONTRAST[art["dataset_id"]]
    bio = [i for i, t in enumerate(art["theme_ids"])
           if t not in ("background_matrix", "unknown_mixed")]
    Xa = art["themes_mat"][art["group"] == a][:, bio]
    Xb = art["themes_mat"][art["group"] == b][:, bio]
    within = 0.5 * (np.linalg.norm(Xa - Xa.mean(0), axis=1).mean()
                    + np.linalg.norm(Xb - Xb.mean(0), axis=1).mean())
    between = float(np.linalg.norm(Xa.mean(0) - Xb.mean(0)))
    return {"within_group": float(within), "between_group": between,
            "ratio": float(between / (within + 1e-9))}


def balanced_view(art):
    """Diabetes-style EXPLORATORY balanced view: standardize each theme's deviation
    across samples so a single dominant axis (e.g. redox) does not visually obscure the
    rest. VISUALIZATION ONLY — the canonical BSV is unchanged."""
    a, b = CONTRAST[art["dataset_id"]]
    bio = [i for i, t in enumerate(art["theme_ids"])
           if t not in ("background_matrix", "unknown_mixed")]
    themes = [art["theme_ids"][i] for i in bio]
    X = art["themes_mat"][:, bio]
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    za = Z[art["group"] == a].mean(0); zb = Z[art["group"] == b].mean(0)
    return themes, za, zb


def study_centroids():
    """Cross-study biochemical centroids in a shared BSV space (generalization panel)."""
    rows, labels, oods = [], [], []
    themes = None
    for key in CONTRAST:
        art = load(key)
        if art is None:
            continue
        themes = art["theme_ids"]
        bio = [i for i, t in enumerate(themes) if t not in ("background_matrix", "unknown_mixed")]
        for g in art["groups"]:
            rows.append(art["themes_mat"][art["group"] == g][:, bio].mean(0))
            labels.append(f"{art['display_name'].split()[0]}·{g}")
            oods.append(float(art["ood"][art["group"] == g].mean()))
    if not rows:
        return None
    proj, var = pca_2d(np.array(rows))
    return {"proj": proj, "labels": labels, "ood": oods, "var": var}
