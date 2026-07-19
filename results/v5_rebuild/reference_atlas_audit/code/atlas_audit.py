"""GAIRA Raman Reference Atlas v0.1 — component audit engine (READ-ONLY).

Audits the FROZEN NMF k=24 atlas. Nothing here refits, reweights or reselects the
representation; the atlas is loaded from its frozen artifact and only projected.

Throughout, a strict distinction is kept between the MATHEMATICAL object (a latent
component) and any BIOCHEMICAL interpretation of it, which is always tentative,
evidence-scored, and never a molecular assignment.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score, mutual_info_score

from gaira.foundation.families_raman import family_of as _base_raman_family

# ── audit-stage family refinement ────────────────────────────────────────────
# The foundation module is deliberately NOT modified (its report must stay
# reproducible). These are gaps found during this audit: monounsaturated fatty
# acids that fell through to "organic_acid", triacylglycerols whose names contain
# digits, and greek-lettered phospholipids.
_FA_EXTRA = ("elaidic", "vaccenic", "palmitoleic", "myristoleic", "erucic", "nervonic",
             "gadoleic", "eicosenoic", "eicosapentaenoic", "docosahexaenoic", "petroselinic",
             "ricinoleic", "lignoceric", "cerotic", "montanic")


def family_of(name):
    """Foundation assignment, with audit-stage gap fixes. Chemistry only."""
    base = _base_raman_family(name)
    s = str(name).strip().lower().lstrip("\u03b1\u03b2\u03b3-").strip()
    if base in ("organic_acid", "unknown"):
        if any(k in s for k in _FA_EXTRA) or re.search(r"(enoic|anoic)\s*acid$", s):
            return "fatty_acid"
        if re.match(r"^tri[-a-z0-9]+in$", s):
            return "triglyceride"
        if s.startswith("phosphatidyl") or "sphingomyelin" in s:
            return "phospholipid"
    if base == "unknown" and re.match(r"^tri[-a-z0-9]+in$", s):
        return "triglyceride"
    return base

# ── coarse molecular class above the chemical family ──
CLASS_OF_FAMILY = {
    "lipid": "lipid", "fatty_acid": "lipid", "triglyceride": "lipid",
    "phospholipid": "lipid", "sterol": "lipid", "carotenoid": "lipid",
    "saccharide": "carbohydrate", "polysaccharide": "carbohydrate", "polyol": "carbohydrate",
    "protein": "protein/peptide", "amino_acid": "protein/peptide",
    "nucleic_acid": "nucleic", "purine": "nucleic", "pyrimidine": "nucleic",
    "nucleoside": "nucleic", "nucleotide": "nucleic",
    "cofactor": "cofactor/small-molecule", "organic_acid": "cofactor/small-molecule",
    "small_nitrogenous": "cofactor/small-molecule",
}


def molecular_class(analyte):
    return CLASS_OF_FAMILY.get(family_of(analyte), "unassigned")


def subfamily(analyte):
    """Conservative finer label; 'unavailable' when it cannot be derived from the
    name with confidence. Never invented."""
    s = str(analyte).lower()
    f = family_of(analyte)
    if f == "saccharide":
        if any(d in s for d in ("sucrose", "lactose", "maltose", "trehalose", "cellobiose")):
            return "disaccharide"
        if "raffinose" in s:
            return "trisaccharide"
        return "monosaccharide"
    if f == "fatty_acid":
        return ("unsaturated fatty acid"
                if any(u in s for u in ("oleic", "linole", "arachidonic", "enoic", "palmitoleic"))
                else "saturated fatty acid")
    if f == "purine":
        return "purine base/derivative"
    if f == "pyrimidine":
        return "pyrimidine base"
    if f == "triglyceride":
        return "triacylglycerol"
    if f == "amino_acid":
        if any(a in s for a in ("phenylalanine", "tyrosine", "tryptophan", "histidine")):
            return "aromatic amino acid"
        return "aliphatic/polar amino acid"
    return "unavailable"


def molecular_weight(analyte):
    """No structure/formula metadata exists in this corpus."""
    return "unavailable"


def biochemical_role(analyte):
    """No curated per-analyte role table exists in this corpus."""
    return "unavailable"


# ───────────────────────── helpers ─────────────────────────
def _unit(X):
    X = np.nan_to_num(X)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def norm_entropy(p):
    p = np.asarray(p, float)
    p = p[p > 0]
    if p.size <= 1:
        return 0.0
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)) / np.log(len(p)))


def gini(v):
    v = np.sort(np.abs(np.asarray(v, float)))
    n = len(v); c = np.cumsum(v)
    return float((n + 1 - 2 * (c / (c[-1] + 1e-12)).sum()) / n)


def analyte_activation(Z, meta):
    """Mean activation per analyte x component (analyte-level atlas view)."""
    df = pd.DataFrame(np.clip(np.nan_to_num(Z), 0, None))
    df["analyte"] = meta.analyte.values
    return df.groupby("analyte").mean()


# ───────────────────────── P5 spectral interpretation ─────────────────────────
def component_bands(W, grid, prom_frac=0.12, n_max=10):
    out = []
    for row in W:
        r = np.nan_to_num(row); rng = r.max() - r.min()
        if rng <= 0:
            out.append(pd.DataFrame(columns=["position", "prominence", "width_cm", "importance"]))
            continue
        idx, props = find_peaks(r, prominence=prom_frac * rng, distance=4)
        if len(idx) == 0:
            idx = np.array([int(np.argmax(r))]); props = {"prominences": np.array([rng])}
        w = peak_widths(r, idx, rel_height=0.5)[0] * float(np.median(np.diff(grid)))
        order = np.argsort(-props["prominences"])[:n_max]
        tot = props["prominences"].sum() + 1e-12
        out.append(pd.DataFrame({
            "position": grid[idx[order]], "prominence": props["prominences"][order],
            "width_cm": w[order], "importance": props["prominences"][order] / tot}).sort_values("position"))
    return out


def band_uniqueness(bands, tol=10.0):
    """How many OTHER components carry a band at the same position."""
    allpos = [b.position.values for b in bands]
    uniq = []
    for j, pos in enumerate(allpos):
        u = []
        for p in pos:
            shared = sum(1 for i, o in enumerate(allpos) if i != j and np.any(np.abs(o - p) <= tol))
            u.append(1.0 - shared / max(1, len(allpos) - 1))
        uniq.append(float(np.mean(u)) if u else np.nan)
    return uniq


def match_literature(bands, peaks_df, tol_default=12.0):
    if peaks_df is None or len(peaks_df) == 0:
        return []
    hits = []
    for b in bands:
        tol = peaks_df.tolerance_cm.fillna(tol_default) if "tolerance_cm" in peaks_df else tol_default
        m = (peaks_df.peak_cm - b).abs() <= tol
        for _, r in peaks_df[m].iterrows():
            hits.append({"band_cm": float(b), "lit_cm": float(r.peak_cm),
                         "group": str(r.get("assigned_group", "")),
                         "molecule": str(r.get("assigned_molecule", "")),
                         "confidence": str(r.get("confidence_text", ""))})
    return hits


# ───────────────────────── P2 composition ─────────────────────────
def composition(A, j, min_share=1e-4):
    """Every analyte contributing to component j, sorted by loading."""
    v = A.values[:, j]
    tot = v.sum() + 1e-12
    rows = []
    for i, a in enumerate(A.index):
        if v[i] <= min_share * tot:
            continue
        rows.append({"component": j, "analyte": a, "loading": float(v[i]),
                     "normalized_loading": float(v[i] / (v.max() + 1e-12)),
                     "contribution_pct": float(100 * v[i] / tot),
                     "molecular_class": molecular_class(a), "chemical_family": family_of(a),
                     "subfamily": subfamily(a), "molecular_weight": molecular_weight(a),
                     "biochemical_role": biochemical_role(a)})
    return pd.DataFrame(rows).sort_values("loading", ascending=False)


# ───────────────────────── P4 chemical coherence ─────────────────────────
def coherence(A, j, Xa, top_n=15):
    """Chemical + spectral coherence of one component."""
    v = A.values[:, j]
    tot = v.sum() + 1e-12
    fam = np.array([family_of(a) for a in A.index])
    cls = np.array([molecular_class(a) for a in A.index])

    fam_share = pd.Series(v, index=fam).groupby(level=0).sum() / tot
    cls_share = pd.Series(v, index=cls).groupby(level=0).sum() / tot
    fam_share = fam_share.sort_values(ascending=False)
    cls_share = cls_share.sort_values(ascending=False)

    base = pd.Series(A.values.sum(axis=1), index=fam).groupby(level=0).sum()
    base = base / base.sum()
    top_fam = fam_share.index[0]
    enrich = float(fam_share.iloc[0] / (base.get(top_fam, 1e-12) + 1e-12))

    # mutual information between (discretised) activation and family label
    q = pd.qcut(pd.Series(v).rank(method="first"), 4, labels=False)
    mi = float(mutual_info_score(fam, q))

    order = np.argsort(-v)[:top_n]
    spec = _unit(Xa[order])
    S = spec @ spec.T
    iu = np.triu_indices(len(order), 1)
    spec_sim = float(S[iu].mean()) if len(order) > 1 else np.nan

    return {
        "component": j,
        "shannon_entropy_family": norm_entropy(fam_share.values),
        "shannon_entropy_analyte": norm_entropy(v),
        "dominant_class": cls_share.index[0], "dominant_class_fraction": float(cls_share.iloc[0]),
        "dominant_family": top_fam, "class_purity": float(fam_share.iloc[0]),
        "enrichment_vs_corpus": enrich, "mutual_information_family": mi,
        "n_chemical_families": int((fam_share > 0.01).sum()),
        "avg_molecular_similarity": "unavailable",     # no structures/fingerprints in corpus
        "avg_spectral_similarity_top": spec_sim,
        "avg_loading_concentration": float(np.sort(v)[::-1][:5].sum() / tot),
    }


# ───────────────────────── P6 relationships ─────────────────────────
def relationships(A, W, bands, tol=10.0, top_n=15):
    k = W.shape[0]
    act_corr = np.corrcoef(A.values.T)
    act_corr = np.nan_to_num(act_corr)
    spec_cos = _unit(W) @ _unit(W).T
    tops = [set(A.index[np.argsort(-A.values[:, j])[:top_n]]) for j in range(k)]
    shared_an = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            shared_an[i, j] = len(tops[i] & tops[j]) / len(tops[i] | tops[j])
    shared_bd = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            pi, pj = bands[i].position.values, bands[j].position.values
            if len(pi) == 0 or len(pj) == 0:
                continue
            m = sum(1 for p in pi if np.any(np.abs(pj - p) <= tol))
            shared_bd[i, j] = m / len(pi)
    return {"activation_corr": act_corr, "spectral_cosine": spec_cos,
            "shared_analytes": shared_an, "shared_bands": shared_bd}


# ───────────────────────── P7 grouping study ─────────────────────────
def grouping_distance(rel, w_spec=0.5, w_act=0.5):
    D = w_spec * (1 - rel["spectral_cosine"]) + w_act * (1 - rel["activation_corr"])
    D = np.clip((D + D.T) / 2, 0, None)
    np.fill_diagonal(D, 0.0)
    return D


def grouping_study(A, W, rel, Xa, ks=(6, 8, 10, 12, 14, 16), n_boot=40, seed=0):
    D = grouping_distance(rel)
    Zl = linkage(squareform(D, checks=False), method="average")
    rng = np.random.default_rng(seed)
    analytes = np.array(A.index)
    rows, assignments = [], {}
    for k in ks:
        lab = fcluster(Zl, t=k, criterion="maxclust")
        assignments[k] = lab
        try:
            sil = float(silhouette_score(D, lab, metric="precomputed"))
        except Exception:
            sil = np.nan
        # bootstrap reproducibility: resample ANALYTES, rebuild the activation
        # correlation (the atlas itself is never refitted), recluster, measure
        # co-assignment agreement.
        agree = []
        base_co = (lab[:, None] == lab[None, :])
        for _ in range(n_boot):
            idx = rng.integers(0, len(analytes), len(analytes))
            Ab = A.values[idx]
            ac = np.nan_to_num(np.corrcoef(Ab.T))
            Db = 0.5 * (1 - rel["spectral_cosine"]) + 0.5 * (1 - ac)
            Db = np.clip((Db + Db.T) / 2, 0, None); np.fill_diagonal(Db, 0)
            lb = fcluster(linkage(squareform(Db, checks=False), method="average"),
                          t=k, criterion="maxclust")
            co = (lb[:, None] == lb[None, :])
            iu = np.triu_indices(len(lab), 1)
            agree.append(float((co[iu] == base_co[iu]).mean()))
        # chemical coherence of the groups
        chem, spec = [], []
        for g in np.unique(lab):
            members = np.where(lab == g)[0]
            act = A.values[:, members].sum(axis=1)
            fam = pd.Series(act, index=[family_of(a) for a in A.index]).groupby(level=0).sum()
            fam = fam / (fam.sum() + 1e-12)
            chem.append(float(fam.max()))
            if len(members) > 1:
                S = _unit(W[members]) @ _unit(W[members]).T
                iu = np.triu_indices(len(members), 1)
                spec.append(float(S[iu].mean()))
        interp = float(np.mean([c >= 0.4 for c in chem]))   # groups with a clear dominant family
        rows.append({"n_groups": k, "silhouette": sil,
                     "bootstrap_reproducibility": float(np.mean(agree)),
                     "chemical_coherence": float(np.mean(chem)),
                     "spectral_coherence": float(np.mean(spec)) if spec else np.nan,
                     "interpretable_group_fraction": interp,
                     "singleton_groups": int(np.sum(np.bincount(lab)[1:] == 1))})
    df = pd.DataFrame(rows)
    # rank on the four scientific criteria, NOT silhouette alone
    for c, inv in (("silhouette", False), ("bootstrap_reproducibility", False),
                   ("chemical_coherence", False), ("interpretable_group_fraction", False)):
        v = df[c].values.astype(float)
        lo, hi = np.nanmin(v), np.nanmax(v)
        df["r_" + c] = 0.0 if hi - lo < 1e-12 else (v - lo) / (hi - lo)
    df["composite"] = (0.20 * df.r_silhouette + 0.30 * df.r_bootstrap_reproducibility +
                       0.30 * df.r_chemical_coherence + 0.20 * df.r_interpretable_group_fraction)
    return df.sort_values("composite", ascending=False), assignments, Zl, D


# ───────────────────────── P12 MSS readiness ─────────────────────────
def mss_readiness(A, energy=0.90):
    An = A.values / (A.values.sum(axis=1, keepdims=True) + 1e-12)
    U = _unit(An)
    S = U @ U.T
    np.fill_diagonal(S, -np.inf)
    rows = []
    for i, a in enumerate(A.index):
        w = An[i]
        order = np.argsort(-w)
        cum = np.cumsum(w[order])
        n_need = int(np.searchsorted(cum, energy) + 1)
        nn = np.argsort(-S[i])[:3]
        rows.append({
            "analyte": a, "concentration_top3": float(np.sort(w)[::-1][:3].sum()),
            "n_components_for_90pct": n_need,
            "dominant_components": [int(c) for c in order[:3]],
            "dominant_weights": [round(float(w[c]), 4) for c in order[:3]],
            "entropy": norm_entropy(w),
            "signature_uniqueness": float(1.0 - S[i].max()),
            "nearest_neighbours": [A.index[c] for c in nn],
            "nn_similarity": [round(float(S[i][c]), 4) for c in nn],
        })
    df = pd.DataFrame(rows)
    df["assignment_confidence"] = np.select(
        [df.signature_uniqueness >= 0.35, df.signature_uniqueness >= 0.15],
        ["high", "moderate"], default="low")
    return df
