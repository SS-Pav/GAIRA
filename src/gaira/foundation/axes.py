"""GAIRA V5 Foundation — Phase C3: emergent biochemical axes.

The previous curated radar axes are NOT hard-coded. Axes are discovered by
clustering the frozen latent components on how they CO-ACTIVATE across the pure
reference analytes, then interpreted post-hoc from (a) literature Raman peak
assignments and (b) the chemistry of the analytes that load on them.

Every theme label is explicitly tentative and carries an uncertainty record.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from ..evidence.families import family_of


def component_band_peaks(W, grid, n_peaks=6, prom_frac=0.15):
    """Dominant bands of each component loading."""
    from scipy.signal import find_peaks
    out = []
    for row in np.abs(np.nan_to_num(W)):
        rng = row.max() - row.min()
        if rng <= 0:
            out.append([]); continue
        idx, props = find_peaks(row, prominence=prom_frac * rng, distance=4)
        if len(idx) == 0:
            out.append([float(grid[int(np.argmax(row))])]); continue
        order = np.argsort(-props["prominences"])[:n_peaks]
        out.append([float(grid[i]) for i in np.sort(idx[order])])
    return out


def analyte_activation_matrix(Z, meta):
    """Mean activation per analyte (rows) x component (cols)."""
    df = pd.DataFrame(np.clip(np.nan_to_num(Z), 0, None))
    df["analyte"] = meta.analyte.values
    A = df.groupby("analyte").mean()
    return A


def cluster_components(A, max_axes=10, seed=0):
    """Cluster components by their co-activation profile across analytes."""
    C = np.corrcoef(A.values.T)
    C = np.nan_to_num(C, nan=0.0)
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    Zl = linkage(squareform(D, checks=False), method="average")
    best, best_k, best_sil = None, None, -np.inf
    from sklearn.metrics import silhouette_score
    for k in range(3, min(max_axes, A.shape[1] - 1) + 1):
        lab = fcluster(Zl, t=k, criterion="maxclust")
        if len(np.unique(lab)) < 2:
            continue
        try:
            sil = silhouette_score(D, lab, metric="precomputed")
        except Exception:
            continue
        if sil > best_sil:
            best, best_k, best_sil = lab, k, sil
    if best is None:
        best, best_k, best_sil = fcluster(Zl, t=3, criterion="maxclust"), 3, np.nan
    return best, {"n_axes": int(best_k), "silhouette": float(best_sil)}


def match_literature(bands, peaks_df, tol_default=12.0):
    """Map a component's dominant bands onto literature Raman peak assignments."""
    if peaks_df is None or len(peaks_df) == 0:
        return []
    hits = []
    for b in bands:
        d = peaks_df.copy()
        tol = d.tolerance_cm.fillna(tol_default) if "tolerance_cm" in d else tol_default
        m = (d.peak_cm - b).abs() <= tol
        for _, r in d[m].iterrows():
            hits.append({"band_cm": float(b), "lit_peak_cm": float(r.peak_cm),
                         "group": str(r.get("assigned_group", "")),
                         "molecule": str(r.get("assigned_molecule", "")),
                         "confidence": str(r.get("confidence_text", ""))})
    return hits


def component_theme(A, j, fam_fn, top_analytes=10):
    """Tentative biochemical theme of ONE component, from the chemistry of the
    analytes that actually load on it (weighted by activation). This is direct
    evidence from the reference corpus; literature is used only to corroborate."""
    act = A.values[:, j]
    order = np.argsort(-act)[:top_analytes]
    names = [A.index[i] for i in order]
    w = np.array([act[i] for i in order], float)
    w = w / (w.sum() + 1e-12)
    fams = [fam_fn(n) for n in names]
    score = {}
    for f, wi in zip(fams, w):
        if f == "unknown":
            continue
        score[f] = score.get(f, 0.0) + float(wi)
    if not score:
        return "unassigned", 0.0, names, {}
    theme = max(score, key=score.get)
    return theme, float(score[theme]), names, score


def build_axes(Z, meta, W, grid, peaks_df, max_axes=10, top_analytes=10, seed=0,
               fam_fn=None, min_purity=0.30):
    """Discover emergent biochemical axes.

    Components are the emergent parts. Each component is interpreted from the
    chemistry of its top-loading reference analytes (PRIMARY evidence), then
    components sharing a theme are grouped into an axis. Literature peak
    assignments corroborate but never define a theme (the available literature
    table is biofluid-oriented and protein-heavy, so using it as the primary label
    would bias every axis toward 'Proteins'). Co-activation clustering is reported
    separately as a diagnostic.
    """
    from .families_raman import family_of as raman_family, FAMILY_TO_LIT
    fam_fn = fam_fn or raman_family
    A = analyte_activation_matrix(Z, meta)
    bands = component_band_peaks(W, grid)

    # diagnostic only — does component co-activation form natural clusters?
    coact_labels, cl_stats = cluster_components(A, max_axes=max_axes, seed=seed)

    comp_rows = []
    for j in range(W.shape[0]):
        theme, purity, names, score = component_theme(A, j, fam_fn, top_analytes)
        comp_rows.append({"component": j, "theme": theme, "theme_purity": round(purity, 3),
                          "top_analytes": names[:6], "dominant_bands_cm": bands[j],
                          "coactivation_cluster": int(coact_labels[j]),
                          "mean_activation": float(A.values[:, j].mean()),
                          "max_activation": float(A.values[:, j].max())})
    comp_df = pd.DataFrame(comp_rows)
    # weak-evidence components are held out of the named axes rather than forced
    comp_df["axis_theme"] = np.where(comp_df.theme_purity >= min_purity,
                                     comp_df.theme, "unassigned")

    axes = []
    for ax_i, theme in enumerate(sorted(comp_df.axis_theme.unique()), start=1):
        members = comp_df[comp_df.axis_theme == theme].component.tolist()
        act = A.values[:, members].sum(axis=1)
        order = np.argsort(-act)[:top_analytes]
        top = [A.index[i] for i in order]
        fam_counts = pd.Series([fam_fn(t) for t in top]).value_counts().to_dict()
        allbands = sorted({b for j in members for b in bands[j]})
        lit = match_literature(allbands, peaks_df)
        lit_groups = pd.Series([h["group"] for h in lit]).value_counts().to_dict() if lit else {}

        expected_lit = FAMILY_TO_LIT.get(theme)
        lit_supports = bool(expected_lit and expected_lit in lit_groups)
        purity = float(np.mean(comp_df[comp_df.axis_theme == theme].theme_purity))
        conf = "low"
        if theme != "unassigned":
            if purity >= 0.6 and lit_supports:
                conf = "medium-high"
            elif purity >= 0.6 or lit_supports:
                conf = "medium"
        axes.append({
            "axis": ax_i, "tentative_theme": theme,
            "n_components": len(members), "components": members,
            "mean_component_purity": round(purity, 3),
            "dominant_bands_cm": allbands[:14],
            "top_analytes": top,
            "top_analyte_families": fam_counts,
            "literature_hits": len(lit), "literature_groups": lit_groups,
            "literature_supports_theme": lit_supports,
            "expected_literature_group": expected_lit,
            "theme_basis": ("chemistry of top-loading reference analytes "
                            f"(mean purity {purity:.2f})" +
                            (f"; corroborated by literature group '{expected_lit}'"
                             if lit_supports else "; no literature corroboration")),
            "theme_confidence": conf,
            "uncertainty_note": ("TENTATIVE post-hoc interpretation of an unsupervised "
                                 "decomposition. Themes were not used to fit the manifold and "
                                 "are not molecular assignments."),
            "variance_share": float(act.sum() / (A.values.sum() + 1e-12)),
        })
    axes.sort(key=lambda a: -a["variance_share"])
    for i, a in enumerate(axes, start=1):
        a["axis"] = i
    amap = {a["tentative_theme"]: a["axis"] for a in axes}
    comp_df["axis"] = comp_df.axis_theme.map(amap)
    return axes, comp_df, A, cl_stats
