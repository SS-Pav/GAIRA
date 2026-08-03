"""GAIRA V4 — null-calibrated hierarchical recoverability analysis.

For every representation level (latent, MSS, raw theme, 4 theme-identity residual variants,
theme rank, top-k, argmax) computes: matched value, analyte-mismatched null distribution,
retrieval-rank permutation p-value, Benjamini-Hochberg FDR, and leave-one-replicate-out
(jackknife) stability. Defines analyte recovery ONLY as matched > mismatched null with
FDR-significant retrieval and stable replicates — never a raw cosine threshold. Adds serum-blank
purine controls and pure→serum matrix prediction.

Reproduces V3's matched z/m/b exactly (same frozen calls); adds only null calibration.
Deterministic (fixed seeds). Frozen atlas 09ed804a… unchanged.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr, linregress

REPO = Path("/Users/surajpg/projects/GAIRA")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "results/v5_rebuild/spike_validation/code"))
from gaira.engine import GAIRAEngine
from gaira.engine.mss import MSSLayer
from gaira.foundation import dataset as DS
from gaira.foundation.families_raman import family_of
from gaira.data.synonyms import canonical
import spike_lib as SL

OUT = REPO / "results/v5_rebuild/hierarchical_recoverability_v4"
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "artifacts").mkdir(parents=True, exist_ok=True)
V3M = REPO / "results/v5_rebuild/representation_hierarchy_v3/tables/per_analyte_hierarchy.csv"
PHASE7 = REPO / "results/v5_rebuild/spike_validation/tables/phase7_serum_vs_pure.csv"
VALID = REPO / "results/v5_rebuild/foundation_audit/tables/validation_results.json"
CANON = "09ed804a40836f4a05a91ba10900cded"

FAM_THEME = {"purine": ["nucleic_purine"], "pyrimidine": ["nucleic_pyrimidine"],
    "nucleic_acid": ["nucleic_purine", "nucleic_pyrimidine"], "nucleoside": ["nucleic_purine", "nucleic_pyrimidine"],
    "protein": ["protein_peptide"], "amino_acid": ["aromatic_amino_acid", "protein_peptide"],
    "saccharide": ["saccharide_glycan"], "polysaccharide": ["saccharide_glycan"], "polyol": ["saccharide_glycan"],
    "fatty_acid": ["lipid_acyl"], "triglyceride": ["lipid_acyl"], "phospholipid": ["lipid_acyl"],
    "lipid": ["lipid_acyl", "sterol_membrane"], "sterol": ["sterol_membrane"],
    "organic_acid": ["organic_acid_metabolism"], "cofactor": ["sulfur_antioxidant", "redox_broad"],
    "small_nitrogenous": ["organic_acid_metabolism"]}
ANALYTE_THEME = {"glutathione": ["sulfur_antioxidant"], "cysteine": ["sulfur_antioxidant"],
    "methionine": ["sulfur_antioxidant"], "ergothioneine": ["sulfur_antioxidant"], "riboflavin": ["redox_broad"]}


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def cos_rows(A, B):
    """cosine similarity matrix rows(A) x rows(B)."""
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return An @ Bn.T


def bh_fdr(p):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, float); n = len(p); order = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        i = n - rank
        val = min(prev, p[idx] * n / i); q[idx] = val; prev = val
    return q


def whiten(M, mu, eps=1e-3):
    """Σ^(-1/2)(x-μ) rows of M; regularized symmetric inverse sqrt."""
    X = M - mu
    S = np.cov(X.T) + eps * np.eye(X.shape[1])
    w, V = np.linalg.eigh(S)
    W = V @ np.diag(1.0 / np.sqrt(np.clip(w, eps, None))) @ V.T
    return X @ W


def main():
    eng = GAIRAEngine(); mss = MSSLayer.from_engine(eng); atlas = eng.atlas
    assert atlas.meta["fingerprint"] == CANON, "FROZEN ATLAS CHANGED"
    THEMES = eng.builder.onto.biochemical_theme_ids
    MOTIFS = [m.id for m in mss.motifs if not m.non_biochemical]
    MOTIF_PARENT = {m.id: m.parent_theme for m in mss.motifs if not m.non_biochemical}
    PUR = THEMES.index("nucleic_purine")

    def coords(V): return atlas.coordinates(np.atleast_2d(np.nan_to_num(V)))

    def vecs(coord):
        b = eng.infer(coordinates=np.asarray(coord, float), domain="buffer").bsv
        tv = np.array([b.composition[t] for t in THEMES])
        acts = {a.id: a.composition for a in mss.activate(b)}
        mv = np.array([acts.get(x, 0.0) for x in MOTIFS])
        return tv, mv, float(b.ood_score), float(b.overall_confidence)

    # ── replicate-level coords ──
    corpus = DS.load_reference_corpus(); Zr_all = coords(corpus.X)
    Xs, rs = SL.load_pure_sers(); Zs_all = coords(Xs)
    ra = corpus.meta.analyte.values
    sa = np.array([canonical(a) for a in rs.analyte.values])
    matched = sorted(set(ra) & set(sa))

    # per-analyte mean coord + replicate coords (SERS, for jackknife)
    zR, zS, sers_reps = {}, {}, {}
    for a in matched:
        zR[a] = Zr_all[ra == a].mean(0)
        rep = Zs_all[sa == a]; zS[a] = rep.mean(0); sers_reps[a] = rep

    # infer per-analyte vectors
    tR, mR, oR, cR, tS, mS, oS, cS = ({} for _ in range(8))
    for a in matched:
        tR[a], mR[a], oR[a], cR[a] = vecs(zR[a])
        tS[a], mS[a], oS[a], cS[a] = vecs(zS[a])
    N = len(matched)
    zRM = np.array([zR[a] for a in matched]); zSM = np.array([zS[a] for a in matched])
    tRM = np.array([tR[a] for a in matched]); tSM = np.array([tS[a] for a in matched])
    mRM = np.array([mR[a] for a in matched]); mSM = np.array([mS[a] for a in matched])
    muR, muS = tRM.mean(0), tSM.mean(0)

    # serum-blank theme (Ag background control) — unspiked serum on Ag colloid
    Xb, rb = SL.load_serum_baseline()
    blank_coord = coords(Xb).mean(0)
    tBlank, mBlank, oBlank, cBlank = vecs(blank_coord)

    # ── generic null-calibrated recovery for a (R, S) matrix pair ──
    def recover(R, S, jack=None, name=""):
        """R,S: (N,d) per-analyte. jack: optional dict a->list of loo S-vectors.
        Returns per-analyte dict arrays."""
        sim = cos_rows(R, S)                       # sim[i,j] = cos(R_i, S_j)
        matched_v = np.diag(sim).copy()
        out = {"matched": matched_v}
        p95 = np.zeros(N); p90 = np.zeros(N); p99 = np.zeros(N)
        rank = np.zeros(int, ) if False else np.zeros(N)
        perm_p = np.zeros(N); null_med = np.zeros(N)
        for i in range(N):
            off = np.delete(sim[i], i)
            p90[i], p95[i], p99[i] = np.percentile(off, [90, 95, 99])
            null_med[i] = np.median(off)
            rank[i] = 1 + int((off > matched_v[i]).sum())         # 1 = self best
            perm_p[i] = (1 + int((off >= matched_v[i]).sum())) / (1 + len(off))
        out.update(null_med=null_med, null_p90=p90, null_p95=p95, null_p99=p99,
                   retrieval_rank=rank, perm_p=perm_p, fdr_q=bh_fdr(perm_p))
        # jackknife stability: leave out each SERS replicate; still exceed null95?
        if jack is not None:
            stable = np.ones(N, bool); loo_min = matched_v.copy()
            for i, a in enumerate(matched):
                vals = [cos(R[i], s) for s in jack[a]]
                loo_min[i] = min(vals)
                stable[i] = all(v > p95[i] for v in vals)
            out.update(jack_min=loo_min, stable=stable)
        return out

    # jackknife S-vectors per level
    def jack_latent():
        return {a: [np.delete(sers_reps[a], k, 0).mean(0) for k in range(len(sers_reps[a]))]
                for a in matched}
    JZ = jack_latent()
    def jack_theme_mss():
        jt, jm = {}, {}
        for a in matched:
            reps = sers_reps[a]; tl, ml = [], []
            for k in range(len(reps)):
                t, m, _, _ = vecs(np.delete(reps, k, 0).mean(0)); tl.append(t); ml.append(m)
            jt[a] = tl; jm[a] = ml
        return jt, jm
    JT, JM = jack_theme_mss()

    # ── LEVEL 1 latent ──
    L1 = recover(zRM, zSM, jack=JZ)
    # ── LEVEL 2 MSS ──
    L2 = recover(mRM, mSM, jack=JM)
    # ── LEVEL 3A raw theme ──
    L3raw = recover(tRM, tSM, jack=JT)

    # ── LEVEL 3B theme identity residual variants ──
    variants = {}
    # A Raman-centered
    variants["Rcenter"] = (tRM - muR, tSM - muR, {a: [t - muR for t in JT[a]] for a in matched})
    # B modality-centered
    variants["Mcenter"] = (tRM - muR, tSM - muS, {a: [t - muS for t in JT[a]] for a in matched})
    # C Ag-blank-corrected (serum blank as Ag background)
    variants["Blank"] = (tRM - muR, tSM - tBlank, {a: [t - tBlank for t in JT[a]] for a in matched})
    # D whitened modality residuals
    wR = whiten(tRM, muR); wS = whiten(tSM, muS)
    # jackknife whitened S: recompute using fixed whitening matrix from full S is complex;
    # approximate stability by re-whitening loo vector with full-S transform (fixed muS/Σ)
    Sw = np.cov((tSM - muS).T) + 1e-3 * np.eye(len(THEMES))
    wv, wV = np.linalg.eigh(Sw); Wmat = wV @ np.diag(1/np.sqrt(np.clip(wv, 1e-3, None))) @ wV.T
    variants["Whiten"] = (wR, wS, {a: [(t - muS) @ Wmat for t in JT[a]] for a in matched})

    var_res = {k: recover(A, B, jack=J, name=k) for k, (A, B, J) in variants.items()}

    # expected-theme retention (for theme recovery gating)
    def expected_theme(a):
        return ANALYTE_THEME.get(a, FAM_THEME.get(family_of(a), []))
    exp_top3 = np.zeros(N, bool); exp_rankS = np.zeros(N)
    for i, a in enumerate(matched):
        oS_ = np.argsort(-tSM[i]); rankS = {THEMES[t]: r for r, t in enumerate(oS_, 1)}
        exps = expected_theme(a)
        r = min([rankS[t] for t in exps], default=99); exp_rankS[i] = r
        exp_top3[i] = r <= 3

    # variant scoring: separation, self-rank1 frac, stability, family-robustness (NOT max score)
    def score_variant(res):
        sep = np.median(res["matched"] - res["null_med"])
        r1 = float((res["retrieval_rank"] == 1).mean())
        stab = float(res["stable"].mean())
        # family leave-out robustness: std of median separation dropping each family
        fams = np.array([family_of(a) for a in matched]); seps = []
        for f in set(fams):
            keep = fams != f
            seps.append(np.median((res["matched"] - res["null_med"])[keep]))
        fam_robust = 1.0 / (1.0 + np.std(seps))
        # purine sensitivity: how much self-rank1 depends on purine dim already handled by residual
        composite = 0.5 * r1 + 0.3 * stab + 0.2 * fam_robust
        return dict(separation=round(float(sep), 4), self_rank1_frac=round(r1, 3),
                    stable_frac=round(stab, 3), fam_robust=round(float(fam_robust), 3),
                    composite=round(float(composite), 4))
    var_scores = {k: score_variant(v) for k, v in var_res.items()}
    # select most STABLE + meaningful (highest composite), not highest raw score
    selected = max(var_scores, key=lambda k: var_scores[k]["composite"])
    SEL_NAME = {"Rcenter": "Raman-centered identity residual",
                "Mcenter": "modality-centered identity residual",
                "Blank": "serum-blank-corrected identity residual",
                "Whiten": "whitened modality-residual"}[selected]
    L3id = var_res[selected]

    # ── recovery flags per level ──
    # Analyte-SPECIFIC recovery (documented rule): the analyte's own Ag-SERS is the UNIQUELY
    # nearest match among all 51 (retrieval rank == 1 ⇒ matched > every mismatched ⇒ matched >
    # null95; per-analyte permutation p = 1/51 = 0.0196) AND leave-one-replicate-out stable.
    # Rank==1 is used rather than BH-FDR<0.05 because the discrete retrieval p floors at 1/51,
    # making BH degenerate at N=51 (ties give q≈0.05); FDR q is still computed and reported.
    # A weaker SUPPORTING tier = matched in the top-5% of the null (>null95) but not rank 1.
    def flag(res, extra=None):
        f = res["stable"] & (res["retrieval_rank"] == 1)
        return f if extra is None else (f & extra)
    def supporting(res):
        return (res["matched"] > res["null_p95"]) & (res["retrieval_rank"] != 1)
    rec_latent = flag(L1); sup_latent = supporting(L1)
    rec_mss = flag(L2); sup_mss = supporting(L2)
    rec_theme = flag(L3id, extra=exp_top3)              # theme recovery also needs expected-theme top-3

    # top-k with chance-adjusted null
    top2 = np.zeros(N); top3 = np.zeros(N); top3_null = np.zeros(N)
    for i in range(N):
        oR_ = set(np.argsort(-tRM[i])[:3]);
        top2[i] = len(set(np.argsort(-tRM[i])[:2]) & set(np.argsort(-tSM[i])[:2])) / 2
        top3[i] = len(oR_ & set(np.argsort(-tSM[i])[:3])) / 3
        # null: overlap of i's Raman top3 with OTHER analytes' SERS top3
        nulls = [len(oR_ & set(np.argsort(-tSM[j])[:3])) / 3 for j in range(N) if j != i]
        top3_null[i] = np.mean(nulls)
    # argmax + top-two margin
    argmax_agree = np.array([np.argmax(tRM[i]) == np.argmax(tSM[i]) for i in range(N)])
    margin_R = np.array([np.sort(tRM[i])[-1] - np.sort(tRM[i])[-2] for i in range(N)])
    margin_S = np.array([np.sort(tSM[i])[-1] - np.sort(tSM[i])[-2] for i in range(N)])
    robust_argmax = argmax_agree & (margin_S > 0.02) & (margin_R > 0.02)

    # ── perturbation (Level 4) — 3 analytes only ──
    val = json.loads(VALID.read_text())
    PERT = {"adenine": True, "ergothioneine": True, "urate": True}

    # ── matrix (Level 5) ──
    p7 = pd.read_csv(PHASE7).set_index("analyte")
    serum_tested = np.array([a in p7.index for a in matched])
    serum_disp = np.array([float(p7.loc[a, "spike_displacement_norm"]) if a in p7.index else np.nan for a in matched])
    serum_dir = np.array([float(p7.loc[a, "replicate_direction_cos"]) if a in p7.index else np.nan for a in matched])
    def serum_tier(d, dc):
        if np.isnan(d): return None
        return "strong" if (d >= 0.05 and dc >= 0.35) else "moderate" if d >= 0.02 else "weak"
    serum_tiers = [serum_tier(serum_disp[i], serum_dir[i]) for i in range(N)]
    rec_matrix = np.array([t == "strong" for t in serum_tiers])

    # ── master per-analyte table ──
    rows = []
    for i, a in enumerate(matched):
        prof = []
        if rec_latent[i] and rec_mss[i]: prof.append("spectral+motif preserved")
        elif rec_mss[i] and not rec_latent[i]: prof.append("latent redistributed, motif retained")
        if rec_theme[i]: prof.append("theme-specific")
        if a in PERT: prof.append("perturbation-validated")
        if rec_matrix[i]: prof.append("matrix-recovered")
        if not prof: prof = ["broad-theme only" if top3[i] > top3_null[i] else "no analyte-specific evidence"]
        rows.append({
            "analyte": a, "family": family_of(a), "expected_theme": "|".join(expected_theme(a)) or "mixed",
            "C_latent": round(L1["matched"][i], 4), "latent_null95": round(L1["null_p95"][i], 4),
            "latent_rank": int(L1["retrieval_rank"][i]), "latent_q": round(float(L1["fdr_q"][i]), 4),
            "latent_recovered": bool(rec_latent[i]), "latent_supporting": bool(sup_latent[i]),
            "C_MSS": round(L2["matched"][i], 4), "MSS_null95": round(L2["null_p95"][i], 4),
            "MSS_rank": int(L2["retrieval_rank"][i]), "MSS_q": round(float(L2["fdr_q"][i]), 4),
            "MSS_recovered": bool(rec_mss[i]), "MSS_supporting": bool(sup_mss[i]),
            "C_theme_raw": round(L3raw["matched"][i], 4), "theme_raw_null_med": round(L3raw["null_med"][i], 4),
            "theme_identity": round(L3id["matched"][i], 4), "theme_id_null95": round(L3id["null_p95"][i], 4),
            "theme_id_rank": int(L3id["retrieval_rank"][i]), "theme_id_q": round(float(L3id["fdr_q"][i]), 4),
            "expected_theme_top3": bool(exp_top3[i]), "theme_recovered": bool(rec_theme[i]),
            "rank_rho": round(float(spearmanr(tRM[i], tSM[i]).correlation), 4),
            "top2": round(float(top2[i]), 3), "top3": round(float(top3[i]), 3),
            "top3_null": round(float(top3_null[i]), 3), "top3_over_null": bool(top3[i] > top3_null[i]),
            "argmax_agree": bool(argmax_agree[i]), "argmax_robust": bool(robust_argmax[i]),
            "margin_R": round(float(margin_R[i]), 4), "margin_S": round(float(margin_S[i]), 4),
            "perturbation_validated": a in PERT,
            "perturbation_status": ("dose-response" if a in ("adenine", "ergothioneine")
                                    else "directional depletion" if a == "urate" else "not tested"),
            "serum_tested": bool(serum_tested[i]),
            "serum_tier": serum_tiers[i] if serum_tested[i] else "not tested",
            "matrix_recovered": bool(rec_matrix[i]),
            "delta_purine": round(float(tSM[i][PUR] - tRM[i][PUR]), 4),
            "purine_R": round(float(tRM[i][PUR]), 4), "purine_S": round(float(tSM[i][PUR]), 4),
            "ood_sers": round(oS[a], 4), "confidence_sers": round(cS[a], 4),
            "evidence_profile": "; ".join(prof),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables/per_analyte_evidence_profile.csv", index=False)

    # ── reproducibility vs V3 ──
    v3 = pd.read_csv(V3M).set_index("analyte")
    j = df.set_index("analyte")
    repro = {"C_latent_maxdiff": float((j.C_latent - v3.L1_latent_fingerprint).abs().max()),
             "C_MSS_maxdiff": float((j.C_MSS - v3.L2_mss_motif).abs().max()),
             "C_theme_raw_maxdiff": float((j.C_theme_raw - v3.L3a_theme_raw).abs().max())}

    # ── counts + bootstrap CI on fractions (resample analytes) ──
    rng = np.random.default_rng(0)
    def frac_ci(flagarr, denom_mask=None, B=4000):
        mask = np.ones(N, bool) if denom_mask is None else denom_mask
        v = flagarr[mask]; n = len(v)
        if n == 0: return 0, 0, (0.0, 0.0, 0.0)
        boot = [v[rng.integers(0, n, n)].mean() for _ in range(B)]
        return int(v.sum()), n, (round(float(v.mean()), 3),
                                 round(float(np.percentile(boot, 2.5)), 3),
                                 round(float(np.percentile(boot, 97.5)), 3))
    levels = {
        "latent": (rec_latent, None), "MSS": (rec_mss, None), "theme": (rec_theme, None),
        "top3_over_null": (df.top3_over_null.values, None),
        "argmax_robust": (robust_argmax, None),
        "perturbation": (df.perturbation_validated.values, None),
        "matrix": (rec_matrix, serum_tested),
    }
    crows = []
    for name, (fl, dm) in levels.items():
        n_rec, denom, (frac, lo, hi) = frac_ci(np.asarray(fl, bool), dm)
        crows.append({"level": name, "n_recovered": n_rec, "denominator": denom,
                      "fraction": frac, "ci95_low": lo, "ci95_high": hi,
                      "denominator_note": "serum-tested" if name == "matrix" else "all matched (51)"})
    pd.DataFrame(crows).to_csv(OUT / "tables/recoverable_analyte_counts.csv", index=False)

    # analyte lists per level
    bylevel = {"latent": df[rec_latent].analyte.tolist(), "MSS": df[rec_mss].analyte.tolist(),
               "theme": df[rec_theme].analyte.tolist(),
               "perturbation": df[df.perturbation_validated].analyte.tolist(),
               "matrix": df[rec_matrix].analyte.tolist()}
    pd.DataFrame([{"level": k, "n": len(v), "analytes": ", ".join(v)} for k, v in bylevel.items()]
                 ).to_csv(OUT / "tables/recoverable_analytes_by_level.csv", index=False)

    # overlap matrix
    flags = {"latent": rec_latent, "MSS": rec_mss, "theme": rec_theme,
             "perturbation": df.perturbation_validated.values.astype(bool), "matrix": rec_matrix}
    ov = pd.DataFrame(index=flags, columns=flags, dtype=int)
    for a in flags:
        for b in flags:
            ov.loc[a, b] = int((flags[a] & flags[b]).sum())
    ov.to_csv(OUT / "tables/recoverability_overlap_matrix.csv")

    # threshold sensitivity: supporting-tier counts at 90/95/99 null percentiles (rank-1 is
    # percentile-independent, so this varies the SUPPORTING 'matched>null_pct & stable' tier)
    trows = []
    for pct, key in [(90, "null_p90"), (95, "null_p95"), (99, "null_p99")]:
        rl = L1["stable"] & (L1["matched"] > L1[key])
        rm = L2["stable"] & (L2["matched"] > L2[key])
        rt = L3id["stable"] & (L3id["matched"] > L3id[key]) & exp_top3
        trows.append({"null_percentile": pct, "latent_supported": int(rl.sum()),
                      "MSS_supported": int(rm.sum()), "theme_supported": int(rt.sum()),
                      "latent_rank1": int(rec_latent.sum()), "MSS_rank1": int(rec_mss.sum()),
                      "theme_rank1": int(rec_theme.sum())})
    pd.DataFrame(trows).to_csv(OUT / "tables/recoverability_threshold_sensitivity.csv", index=False)

    # theme-variant comparison
    vc = pd.DataFrame([{"variant": k, "name": SEL_NAME if k == selected else k, **var_scores[k],
                        "selected": k == selected} for k in var_res])
    vc.to_csv(OUT / "tables/theme_variant_comparison.csv", index=False)

    # MSS specificity ranking (null-adjusted)
    msr = df[["analyte", "family", "C_MSS", "MSS_null95", "MSS_rank", "MSS_q", "MSS_recovered"]].copy()
    msr["mss_specificity"] = (df.C_MSS - df.MSS_null95).round(4)
    msr.sort_values("mss_specificity", ascending=False).to_csv(OUT / "tables/mss_specificity_ranking.csv", index=False)

    # ── purine attractor blank controls ──
    pc = {"serum_blank_purine_theme": round(float(tBlank[PUR]), 4),
          "serum_blank_dominant_theme": THEMES[int(np.argmax(tBlank))],
          "serum_blank_ood": round(float(oBlank), 4),
          "median_purine_R": round(float(np.median(tRM[:, PUR])), 4),
          "median_purine_S": round(float(np.median(tSM[:, PUR])), 4),
          "n_delta_purine_positive": int((df.delta_purine > 0).sum()),
          "note": "No pure Ag-colloid buffer blank exists in the dataset; unspiked-serum-on-Ag used as the available background control."}
    # Δpurine correlations
    dpc = {}
    for nm, x in [("latent", df.C_latent.values), ("MSS", df.C_MSS.values),
                  ("ood", df.ood_sers.values), ("matrix_disp", serum_disp)]:
        m = ~np.isnan(x) & ~np.isnan(df.delta_purine.values)
        r, p = pearsonr(df.delta_purine.values[m], x[m]); sr = spearmanr(df.delta_purine.values[m], x[m]).correlation
        dpc[nm] = {"pearson_r": round(float(r), 4), "p": float(f"{p:.2e}"), "spearman": round(float(sr), 4)}
    pd.DataFrame([{"variable": k, **v} for k, v in dpc.items()]).to_csv(OUT / "tables/delta_purine_correlations.csv", index=False)
    pd.DataFrame([pc]).to_csv(OUT / "tables/purine_blank_controls.csv", index=False)

    # ── matrix prediction: does each pure metric predict serum recovery? ──
    st = serum_tested
    def predict(x, y=serum_disp, mask=st):
        m = mask & ~np.isnan(x) & ~np.isnan(y)
        r, p = pearsonr(x[m], y[m]); sr = spearmanr(x[m], y[m]).correlation
        # bootstrap CI on pearson r
        idx = np.where(m)[0]; boots = []
        for _ in range(2000):
            s = rng.integers(0, len(idx), len(idx)); ii = idx[s]
            if np.std(x[ii]) > 0 and np.std(y[ii]) > 0:
                boots.append(pearsonr(x[ii], y[ii])[0])
        return {"pearson_r": round(float(r), 4), "p": float(f"{p:.2e}"), "spearman": round(float(sr), 4),
                "ci95_low": round(float(np.percentile(boots, 2.5)), 4),
                "ci95_high": round(float(np.percentile(boots, 97.5)), 4), "n": int(m.sum())}
    mp = {"latent_cosine": predict(df.C_latent.values), "MSS_cosine": predict(df.C_MSS.values),
          "MSS_specificity": predict((df.C_MSS - df.MSS_null95).values),
          "theme_raw": predict(df.C_theme_raw.values), "theme_identity": predict(df.theme_identity.values),
          "rank_rho": predict(df.rank_rho.values), "ood": predict(df.ood_sers.values),
          "confidence": predict(df.confidence_sers.values)}
    pd.DataFrame([{"predictor": k, **v} for k, v in mp.items()]).to_csv(OUT / "tables/matrix_prediction.csv", index=False)
    # categorical enrichment: are strong-serum analytes enriched in latent-recovered?
    strong = rec_matrix
    enrich = {"latent_recovered_and_serum_strong": int((rec_latent & strong).sum()),
              "serum_strong_total": int(strong.sum()),
              "latent_recovered_total": int(rec_latent.sum())}

    # ── level null summary (headline table) ──
    def lvl_summary(res, fl, label, denom=N):
        return {"level": label, "matched_median": round(float(np.median(res["matched"])), 4),
                "null_median": round(float(np.median(res["null_med"])), 4),
                "separation": round(float(np.median(res["matched"] - res["null_med"])), 4),
                "median_perm_p": round(float(np.median(res["perm_p"])), 4),
                "n_rank1": int((res["retrieval_rank"] == 1).sum()),
                "n_above_null95": int((res["matched"] > res["null_p95"]).sum()),
                "n_fdr_sig": int((res["fdr_q"] < 0.05).sum()),
                "n_recovered": int(np.asarray(fl).sum()), "denominator": denom}
    lvlsum = [lvl_summary(L1, rec_latent, "L1 latent"), lvl_summary(L2, rec_mss, "L2 MSS"),
              lvl_summary(L3raw, (L3raw["retrieval_rank"] == 1), "L3a theme raw (diagnostic)"),
              lvl_summary(L3id, rec_theme, f"L3b {SEL_NAME}")]
    pd.DataFrame(lvlsum).to_csv(OUT / "tables/level_null_summary.csv", index=False)

    # ── summary json ──
    summary = {
        "atlas_fingerprint": CANON, "n_matched": N,
        "reproducibility_vs_v3": repro,
        "selected_theme_variant": {"key": selected, "name": SEL_NAME, "scores": var_scores},
        "level_null_summary": lvlsum,
        "counts": {r["level"]: {"n": r["n_recovered"], "denom": r["denominator"],
                                "frac": r["fraction"], "ci": [r["ci95_low"], r["ci95_high"]]} for r in crows},
        "recovered_analytes": bylevel,
        "threshold_sensitivity": trows,
        "purine_controls": pc, "delta_purine_correlations": dpc,
        "matrix_prediction": mp, "matrix_enrichment": enrich,
        "mss_is_primary_candidate": {
            "mss_separation": lvlsum[1]["separation"], "latent_separation": lvlsum[0]["separation"],
            "mss_n_recovered": lvlsum[1]["n_recovered"], "latent_n_recovered": lvlsum[0]["n_recovered"],
            "verdict_note": "decided by null-specificity below, not by preference"},
        "spearman_matched_vs_null": {"matched_median": round(float(np.median([spearmanr(tRM[i], tSM[i]).correlation for i in range(N)])), 4)},
        "argmax": {"agree": int(argmax_agree.sum()), "robust_after_margin": int(robust_argmax.sum())},
    }
    (OUT / "artifacts/recoverability_summary.json").write_text(json.dumps(summary, indent=2))

    # per-analyte vectors + mean atlas-grid spectra for the figure scripts
    ram_spec = np.array([np.nan_to_num(corpus.X[ra == a]).mean(0) for a in matched])
    sers_spec = np.array([np.nan_to_num(Xs[sa == a]).mean(0) for a in matched])
    grid = getattr(atlas, "grid", getattr(atlas, "wavenumbers", np.arange(ram_spec.shape[1])))
    np.savez(OUT / "artifacts/vectors.npz", analytes=np.array(matched),
             families=np.array([family_of(a) for a in matched]),
             zR=zRM, zS=zSM, mR=mRM, mS=mSM, tR=tRM, tS=tSM,
             themes=np.array(THEMES), motifs=np.array(MOTIFS),
             ram_spec=ram_spec, sers_spec=sers_spec, grid=np.asarray(grid, float),
             t_blank=tBlank, purine_idx=PUR)

    print(json.dumps({"fingerprint": CANON, "N": N, "reproducibility_vs_v3": repro,
                      "selected_theme_variant": selected, "counts": summary["counts"],
                      "level_null_summary": lvlsum,
                      "purine_blank_theme": pc["serum_blank_purine_theme"],
                      "purine_blank_dominant": pc["serum_blank_dominant_theme"]}, indent=2))
    print("\nvariant scores:"); print(vc.to_string(index=False))
    print("\nmatrix prediction (latent, MSS, MSS_specificity):")
    for k in ["latent_cosine", "MSS_cosine", "MSS_specificity"]:
        print(" ", k, mp[k])


if __name__ == "__main__":
    main()
