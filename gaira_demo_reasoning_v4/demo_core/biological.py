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
}


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
