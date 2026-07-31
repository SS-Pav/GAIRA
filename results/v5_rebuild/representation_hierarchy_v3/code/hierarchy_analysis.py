"""V3 — the Representation Hierarchy reanalysis of Raman → Ag-SERS transfer.

Completely reruns the cross-modal analysis on the FROZEN atlas (09ed804a…), keeps every V2
metric, and adds three first-class new ones — theme RANK preservation (Spearman), top-k theme
overlap as a stand-alone layer, and per-analyte ΔPurine — plus a matrix-robustness regression.
Cross-checks its shared numbers against the committed V2 table so previous analyses stay
reproducible. Additive; nothing frozen is modified.

Outputs: results/v5_rebuild/representation_hierarchy_v3/{tables,artifacts}.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, linregress, pearsonr

REPO = Path("/Users/surajpg/projects/GAIRA")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "results/v5_rebuild/spike_validation/code"))
from gaira.engine import GAIRAEngine
from gaira.engine.mss import MSSLayer
from gaira.foundation import dataset as DS
from gaira.foundation.families_raman import family_of
from gaira.data.synonyms import canonical
import spike_lib as SL

OUT = REPO / "results/v5_rebuild/representation_hierarchy_v3"
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "artifacts").mkdir(parents=True, exist_ok=True)
V2_METRICS = REPO / "results/v5_rebuild/pure_ag_sers_theme_preservation/tables/per_analyte_transfer_metrics.csv"
PHASE7 = REPO / "results/v5_rebuild/spike_validation/tables/phase7_serum_vs_pure.csv"
CANON_FP = "09ed804a40836f4a05a91ba10900cded"

FAM_THEME = {
    "purine": ["nucleic_purine"], "pyrimidine": ["nucleic_pyrimidine"],
    "nucleic_acid": ["nucleic_purine", "nucleic_pyrimidine"],
    "nucleoside": ["nucleic_purine", "nucleic_pyrimidine"],
    "protein": ["protein_peptide"],
    "amino_acid": ["aromatic_amino_acid", "protein_peptide"],
    "saccharide": ["saccharide_glycan"], "polysaccharide": ["saccharide_glycan"],
    "polyol": ["saccharide_glycan"], "fatty_acid": ["lipid_acyl"], "triglyceride": ["lipid_acyl"],
    "phospholipid": ["lipid_acyl"], "lipid": ["lipid_acyl", "sterol_membrane"],
    "sterol": ["sterol_membrane"], "organic_acid": ["organic_acid_metabolism"],
    "cofactor": ["sulfur_antioxidant", "redox_broad"], "small_nitrogenous": ["organic_acid_metabolism"]}
ANALYTE_THEME = {"glutathione": ["sulfur_antioxidant"], "cysteine": ["sulfur_antioxidant"],
                 "methionine": ["sulfur_antioxidant"], "ergothioneine": ["sulfur_antioxidant"],
                 "riboflavin": ["redox_broad"]}


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    eng = GAIRAEngine(); mss = MSSLayer.from_engine(eng); atlas = eng.atlas
    assert atlas.meta["fingerprint"] == CANON_FP, "FROZEN ATLAS CHANGED — abort"
    THEMES = eng.builder.onto.biochemical_theme_ids
    MOTIFS = [m.id for m in mss.motifs if not m.non_biochemical]
    PURINE = THEMES.index("nucleic_purine")

    def coords(V): return atlas.coordinates(np.atleast_2d(np.nan_to_num(V)))

    def profile(coord):
        b = eng.infer(coordinates=np.asarray(coord, float), domain="buffer").bsv
        tv = np.array([b.composition[t] for t in THEMES])
        acts = {a.id: a.composition for a in mss.activate(b)}
        mv = np.array([acts.get(mid, 0.0) for mid in MOTIFS])
        return tv, mv, float(b.ood_score), float(b.overall_confidence)

    corpus = DS.load_reference_corpus(); Zr = coords(corpus.X)
    raman, rn = {}, {}
    for a in pd.unique(corpus.meta.analyte):
        m = corpus.meta.analyte.values == a; raman[a] = Zr[m].mean(0); rn[a] = int(m.sum())
    Xs, rs = SL.load_pure_sers(); Zs = coords(Xs)
    sers, sn = {}, {}
    for a in pd.unique(rs.analyte):
        m = rs.analyte.values == a; sers[canonical(a)] = Zs[m].mean(0); sn[canonical(a)] = int(m.sum())
    matched = sorted(set(raman) & set(sers))

    prof = {a: {"R": profile(raman[a]), "S": profile(sers[a])} for a in matched}
    tbar = np.mean([prof[a]["R"][0] for a in matched], axis=0)   # shared baseline theme vector

    rows = []
    for a in matched:
        fam = family_of(a); expected = ANALYTE_THEME.get(a, FAM_THEME.get(fam, []))
        zr, zs = raman[a], sers[a]
        tR, mR, oodR, confR = prof[a]["R"]; tS, mS, oodS, confS = prof[a]["S"]

        # Layer 1 — latent fingerprint preservation
        L1 = cos(zr, zs)
        # Layer 2 — MSS motif preservation
        L2 = cos(mR, mS)
        # Layer 3a — raw theme similarity (baseline-inflated)
        L3_raw = cos(tR, tS)
        # Layer 3b — identity-specific theme preservation (baseline-subtracted + null)
        dR, dS = tR - tbar, tS - tbar
        L3_identity = cos(dR, dS)
        id_null = float(np.mean([cos(dR, prof[b]["S"][0] - tbar) for b in matched if b != a]))
        # Layer 4 — theme RANK preservation (Spearman) + null
        rho = float(spearmanr(tR, tS).correlation)
        rank_null = float(np.mean([spearmanr(tR, prof[b]["S"][0]).correlation for b in matched if b != a]))
        # Layer 5 — top-k theme overlap
        oR = np.argsort(-tR); oS = np.argsort(-tS)
        top2 = len(set(oR[:2]) & set(oS[:2])) / 2
        top3 = len(set(oR[:3]) & set(oS[:3])) / 3
        # Layer 6 — dominant-theme agreement (argmax)
        domR, domS = THEMES[oR[0]], THEMES[oS[0]]
        argmax_agree = bool(domR == domS)
        # expected theme retention
        rankR = {t: r for r, t in enumerate(np.array(THEMES)[oR], 1)}
        rankS = {t: r for r, t in enumerate(np.array(THEMES)[oS], 1)}
        exp_rank_S = min([rankS[t] for t in expected], default=None)
        # ΔPurine
        dpurine = float(tS[PURINE] - tR[PURINE])

        rows.append({
            "analyte": a, "family": fam, "expected_theme": "|".join(expected) or "mixed",
            "n_raman": rn[a], "n_sers": sn[a],
            "L1_latent_fingerprint": round(L1, 4),
            "L2_mss_motif": round(L2, 4),
            "L3a_theme_raw": round(L3_raw, 4),
            "L3b_theme_identity": round(L3_identity, 4),
            "L3b_identity_null": round(id_null, 4),
            "L3b_identity_separation": round(L3_identity - id_null, 4),
            "L4_theme_rank_rho": round(rho, 4),
            "L4_rank_null": round(rank_null, 4),
            "L4_rank_separation": round(rho - rank_null, 4),
            "L5_top2_overlap": round(top2, 3), "L5_top3_overlap": round(top3, 3),
            "L6_argmax_agreement": argmax_agree,
            "raman_dominant": domR, "sers_dominant": domS,
            "expected_rank_sers": exp_rank_S,
            "expected_retained_top3": bool(exp_rank_S is not None and exp_rank_S <= 3),
            "purine_share_raman": round(float(tR[PURINE]), 4),
            "purine_share_sers": round(float(tS[PURINE]), 4),
            "delta_purine": round(dpurine, 4),
            "ood_sers": round(oodS, 4), "confidence_sers": round(confS, 4),
        })
    df = pd.DataFrame(rows).sort_values("L4_theme_rank_rho", ascending=False).reset_index(drop=True)
    df.to_csv(OUT / "tables/per_analyte_hierarchy.csv", index=False)

    # ── reproducibility cross-check vs committed V2 ──
    v2 = pd.read_csv(V2_METRICS).set_index("analyte")
    j = df.set_index("analyte")
    repro = {
        "component_cosine_max_abs_diff": float((j.L1_latent_fingerprint - v2.component_cosine).abs().max()),
        "theme_raw_max_abs_diff": float((j.L3a_theme_raw - v2.theme_cosine).abs().max()),
        "theme_identity_max_abs_diff": float((j.L3b_theme_identity - v2.theme_cosine_distinct).abs().max()),
        "mss_max_abs_diff": float((j.L2_mss_motif - v2.mss_cosine).abs().max()),
    }

    # ── Layer 4 focussed table ──
    df[["analyte", "family", "L4_theme_rank_rho", "L4_rank_null", "L4_rank_separation",
        "L3a_theme_raw", "L3b_theme_identity", "L6_argmax_agreement"]].to_csv(
        OUT / "tables/theme_rank_preservation.csv", index=False)
    df[["analyte", "family", "L5_top2_overlap", "L5_top3_overlap", "L6_argmax_agreement",
        "expected_retained_top3"]].to_csv(OUT / "tables/topk_overlap.csv", index=False)
    df[["analyte", "family", "purine_share_raman", "purine_share_sers", "delta_purine",
        "L1_latent_fingerprint", "sers_dominant"]].sort_values("delta_purine", ascending=False).to_csv(
        OUT / "tables/delta_purine.csv", index=False)

    # ── family breakdown (all layers) ──
    fam = df.groupby("family").agg(
        n=("analyte", "size"),
        L1_latent=("L1_latent_fingerprint", "median"),
        L2_mss=("L2_mss_motif", "median"),
        L3a_theme_raw=("L3a_theme_raw", "median"),
        L3b_theme_identity=("L3b_theme_identity", "median"),
        L4_rank_rho=("L4_theme_rank_rho", "median"),
        L4_rank_separation=("L4_rank_separation", "median"),
        L5_top3=("L5_top3_overlap", "median"),
        L6_argmax=("L6_argmax_agreement", "mean"),
        delta_purine=("delta_purine", "median")).round(3).reset_index()
    fam.sort_values("L4_rank_rho", ascending=False).to_csv(OUT / "tables/rank_by_family.csv", index=False)

    # ── Sankey edges: Raman dominant -> Ag dominant ──
    sankey = df.groupby(["raman_dominant", "sers_dominant"]).size().reset_index(name="n")
    sankey.to_csv(OUT / "tables/sankey_dominant_flow.csv", index=False)

    # ── Matrix robustness: does pure Ag transfer predict serum recoverability? ──
    p7 = pd.read_csv(PHASE7).set_index("analyte")
    mrows = []
    for a in df.analyte:
        if a in p7.index:
            r = p7.loc[a]
            mrows.append({"analyte": a, "L1_latent_fingerprint": float(j.loc[a, "L1_latent_fingerprint"]),
                          "L4_theme_rank_rho": float(j.loc[a, "L4_theme_rank_rho"]),
                          "serum_spike_displacement": float(r.spike_displacement_norm),
                          "serum_direction_cos": float(r.replicate_direction_cos),
                          "serum_vs_pureSERS_cos": float(r.cos_spike_vs_pureSERS)})
    md = pd.DataFrame(mrows)
    md.to_csv(OUT / "tables/matrix_robustness.csv", index=False)

    def reg(x, y):
        lr = linregress(x, y); pr = pearsonr(x, y); sr = spearmanr(x, y)
        n = len(x)
        # 95% CI on slope
        tcrit = 1.9799  # ~df large; adequate for n≈51
        ci = tcrit * lr.stderr
        return {"n": int(n), "slope": round(float(lr.slope), 4), "intercept": round(float(lr.intercept), 4),
                "slope_ci95": round(float(ci), 4), "r": round(float(lr.rvalue), 4),
                "r2": round(float(lr.rvalue ** 2), 4), "p_value": float(f"{lr.pvalue:.2e}"),
                "pearson_r": round(float(pr[0]), 4), "spearman_rho": round(float(sr.correlation), 4)}
    matrix_reg = {
        "predictor_latent_fingerprint": reg(md.L1_latent_fingerprint.values, md.serum_spike_displacement.values),
        "predictor_theme_rank": reg(md.L4_theme_rank_rho.values, md.serum_spike_displacement.values),
        "predictor_latent_vs_serum_direction": reg(md.L1_latent_fingerprint.values, md.serum_direction_cos.values),
    }

    # ── the Representation Hierarchy summary (5 levels) ──
    def stat(col):
        s = df[col]
        return {"median": round(float(s.median()), 4), "mean": round(float(s.mean()), 4),
                "std": round(float(s.std()), 4), "min": round(float(s.min()), 4),
                "max": round(float(s.max()), 4)}
    hierarchy = {
        "level1_latent_fingerprint": {**stat("L1_latent_fingerprint"),
            "metric": "component cosine over 24 NMF coordinates", "limitation": "adsorption-dominated; not interpretation"},
        "level2_mss_motif": {**stat("L2_mss_motif"),
            "metric": "cosine over 12 biochemical MSS motif activations", "limitation": "mid-level; still surface-sensitive"},
        "level3_theme": {"raw": stat("L3a_theme_raw"), "identity_specific": stat("L3b_theme_identity"),
            "rank_rho": stat("L4_theme_rank_rho"), "top3": stat("L5_top3_overlap"),
            "argmax_agreement_rate": round(float(df.L6_argmax_agreement.mean()), 3),
            "limitation": "raw is baseline-inflated; identity/rank/argmax progressively stricter"},
        "level4_perturbation": {"n_validated": 3, "analytes": ["adenine", "ergothioneine", "uricase"],
            "metric": "dose ρ / directional motif Δ", "limitation": "measured for 3 analytes only"},
        "level5_matrix": {"strong": int((md.serum_spike_displacement >= 0.05).sum()),
            "regression": matrix_reg["predictor_latent_fingerprint"],
            "metric": "serum spike displacement", "limitation": "serum competition; subset of Level 1"},
    }

    summary = {
        "atlas_fingerprint": atlas.meta["fingerprint"], "n_matched": int(len(df)),
        "reproducibility_vs_v2": repro,
        "layers": {
            "L1_latent_fingerprint": stat("L1_latent_fingerprint"),
            "L2_mss_motif": stat("L2_mss_motif"),
            "L3a_theme_raw": stat("L3a_theme_raw"),
            "L3b_theme_identity": stat("L3b_theme_identity"),
            "L4_theme_rank_rho": stat("L4_theme_rank_rho"),
            "L4_rank_separation": stat("L4_rank_separation"),
            "L5_top2_overlap": stat("L5_top2_overlap"),
            "L5_top3_overlap": stat("L5_top3_overlap"),
            "L6_argmax_agreement_rate": round(float(df.L6_argmax_agreement.mean()), 3),
        },
        "rank_positive_separation": int((df.L4_rank_separation > 0).sum()),
        "delta_purine": {"median": round(float(df.delta_purine.median()), 4),
                         "n_increase": int((df.delta_purine > 0).sum()),
                         "mean": round(float(df.delta_purine.mean()), 4)},
        "matrix_regression": matrix_reg,
        "representation_hierarchy": hierarchy,
    }
    (OUT / "artifacts/hierarchy_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame([
        {"level": "1 · Latent fingerprint", "metric": "component cosine", **stat("L1_latent_fingerprint")},
        {"level": "2 · MSS motif", "metric": "mss cosine", **stat("L2_mss_motif")},
        {"level": "3 · Theme (raw)", "metric": "theme cosine raw", **stat("L3a_theme_raw")},
        {"level": "3 · Theme (identity)", "metric": "baseline-subtracted cosine", **stat("L3b_theme_identity")},
        {"level": "3 · Theme (rank ρ)", "metric": "Spearman rank", **stat("L4_theme_rank_rho")},
        {"level": "3 · Theme (top-3)", "metric": "top-3 overlap", **stat("L5_top3_overlap")},
    ]).to_csv(OUT / "tables/representation_hierarchy_summary.csv", index=False)

    print(json.dumps({"fingerprint": summary["atlas_fingerprint"], "n": summary["n_matched"],
                      "reproducibility_vs_v2": repro, "layers": summary["layers"],
                      "rank_positive_separation": summary["rank_positive_separation"],
                      "delta_purine": summary["delta_purine"],
                      "matrix_reg_latent": matrix_reg["predictor_latent_fingerprint"]}, indent=2))
    print("\n=== per-analyte (rank-sorted) ===")
    print(df[["analyte", "family", "L1_latent_fingerprint", "L2_mss_motif", "L3a_theme_raw",
              "L3b_theme_identity", "L4_theme_rank_rho", "L4_rank_separation", "L5_top3_overlap",
              "L6_argmax_agreement", "delta_purine"]].to_string(index=False))


if __name__ == "__main__":
    main()
