"""Multi-level Raman -> Ag-SERS transfer analysis (ADDITIVE; frozen atlas unchanged).

The existing pure-Ag-SERS analysis measures ONE thing: cosine between the 24-component
coordinate vectors (LATENT FINGERPRINT PRESERVATION). This module adds the distinct
second layer — BIOCHEMICAL THEME PRESERVATION — plus MSS preservation, target-theme
retention, and redistribution structure, for every matched analyte. It tests, not
assumes, the thesis that Ag-SERS can redistribute the latent profile while preserving the
biochemical theme.

All computation uses the FROZEN engine (fingerprint 09ed804a...). Deterministic.
Writes results/v5_rebuild/pure_ag_sers_theme_preservation/{tables,artifacts}.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/Users/surajpg/projects/GAIRA")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "results/v5_rebuild/spike_validation/code"))
from gaira.engine import GAIRAEngine
from gaira.engine.mss import MSSLayer
from gaira.foundation import dataset as DS
from gaira.foundation.families_raman import family_of
from gaira.data.synonyms import canonical
import spike_lib as SL

OUT = REPO / "results/v5_rebuild/pure_ag_sers_theme_preservation"
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "artifacts").mkdir(parents=True, exist_ok=True)

# family -> expected high-level GAIRA theme(s). Mixed families carry >1 (never forced to 1).
FAM_THEME = {
    "purine": ["nucleic_purine"], "pyrimidine": ["nucleic_pyrimidine"],
    "nucleic_acid": ["nucleic_purine", "nucleic_pyrimidine"],
    "nucleoside": ["nucleic_purine", "nucleic_pyrimidine"],
    "protein": ["protein_peptide"],
    "amino_acid": ["aromatic_amino_acid", "protein_peptide"],
    "saccharide": ["saccharide_glycan"], "polysaccharide": ["saccharide_glycan"],
    "polyol": ["saccharide_glycan"],
    "fatty_acid": ["lipid_acyl"], "triglyceride": ["lipid_acyl"],
    "phospholipid": ["lipid_acyl"], "lipid": ["lipid_acyl", "sterol_membrane"],
    "sterol": ["sterol_membrane"], "organic_acid": ["organic_acid_metabolism"],
    "cofactor": ["sulfur_antioxidant", "redox_broad"],
    "small_nitrogenous": ["organic_acid_metabolism"],
}
ANALYTE_THEME = {  # sulfur chemistry overrides (chemisorb to Ag via S)
    "glutathione": ["sulfur_antioxidant"], "cysteine": ["sulfur_antioxidant"],
    "methionine": ["sulfur_antioxidant"], "ergothioneine": ["sulfur_antioxidant"],
    "riboflavin": ["redox_broad"],
}
# descriptive tiers (manually chosen, NOT learned biological classes)
TIERS = [("Excellent", 0.80), ("Good", 0.65), ("Moderate", 0.45), ("Weak", 0.25), ("Poor", 0.0)]


def tier(c):
    for n, t in TIERS:
        if c >= t:
            return n
    return "Poor"


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def jsd(p, q):
    p = np.clip(p, 0, None); q = np.clip(q, 0, None)
    p = p / (p.sum() + 1e-12); q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / (b[mask] + 1e-12))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    eng = GAIRAEngine()
    mss = MSSLayer.from_engine(eng)
    atlas = eng.atlas
    assert atlas.meta["fingerprint"] == "09ed804a40836f4a05a91ba10900cded"
    THEMES = eng.builder.onto.biochemical_theme_ids                      # 11
    MOTIFS = [m.id for m in mss.motifs if not m.non_biochemical]         # 12 biochemical

    def coords(V):
        return atlas.coordinates(np.atleast_2d(np.nan_to_num(V)))

    def profile(coord):
        """Return (bsv, theme_vec[11], mss_vec[12], ood, conf) for a 24-coord vector."""
        b = eng.infer(coordinates=np.asarray(coord, float), domain="buffer").bsv
        tv = np.array([b.composition[t] for t in THEMES])
        acts = {a.id: a.composition for a in mss.activate(b)}
        mv = np.array([acts.get(mid, 0.0) for mid in MOTIFS])
        return b, tv, mv, float(b.ood_score), float(b.overall_confidence)

    # ── Raman reference: per-analyte mean coords (full frozen corpus) ──
    corpus = DS.load_reference_corpus()
    Zr = coords(corpus.X)
    raman_by, raman_n = {}, {}
    for a in pd.unique(corpus.meta.analyte):
        mask = corpus.meta.analyte.values == a
        raman_by[a] = Zr[mask].mean(0); raman_n[a] = int(mask.sum())

    # ── pure Ag-SERS: per-analyte mean coords ──
    Xs, rs = SL.load_pure_sers()
    Zs = coords(Xs)
    sers_by, sers_n = {}, {}
    for a in pd.unique(rs.analyte):
        mask = rs.analyte.values == a
        sers_by[canonical(a)] = Zs[mask].mean(0); sers_n[canonical(a)] = int(mask.sum())

    matched = sorted(set(raman_by) & set(sers_by))

    # ── PASS 1: profile every matched analyte in both modalities ──
    prof = {}
    for a in matched:
        prof[a] = {"R": profile(raman_by[a]), "S": profile(sers_by[a])}
    # baseline theme vector = mean Raman theme composition across analytes.
    # Subtracting it isolates each analyte's DISTINCTIVE deviation from the shared
    # background, so we can test whether theme cosine is real preservation or just
    # the compositional-closure baseline that inflates every pairwise cosine.
    tbar = np.mean([prof[a]["R"][1] for a in matched], axis=0)

    # ── PASS 2: matched metrics + null / distinctive controls ──
    rows = []
    for a in matched:
        fam = family_of(a)
        expected = ANALYTE_THEME.get(a, FAM_THEME.get(fam, []))
        zr, zs = raman_by[a], sers_by[a]
        bR, tR, mR, oodR, confR = prof[a]["R"]
        bS, tS, mS, oodS, confS = prof[a]["S"]

        # distinctive (baseline-subtracted) theme vectors
        dR, dS = tR - tbar, tS - tbar
        c_theme_distinct = cos(dR, dS)
        # null: distinctive SERS profile of A vs distinctive Raman profile of every OTHER analyte
        null_cos = [cos(dR, prof[b]["S"][1] - tbar) for b in matched if b != a]
        theme_null_mean = float(np.mean(null_cos))
        # rank of the SELF match among all analytes' distinctive SERS profiles (1 = self is nearest)
        self_vs_all = sorted(matched, key=lambda b: -cos(dR, prof[b]["S"][1] - tbar))
        self_rank = self_vs_all.index(a) + 1

        c_component = cos(zr, zs)                                   # latent fingerprint
        c_theme = cos(tR, tS)                                       # theme preservation (composition)
        c_mss = cos(mR, mS)
        domR, domS = THEMES[int(tR.argmax())], THEMES[int(tS.argmax())]
        rankR = {t: r for r, t in enumerate(np.array(THEMES)[np.argsort(-tR)], 1)}
        rankS = {t: r for r, t in enumerate(np.array(THEMES)[np.argsort(-tS)], 1)}
        topR3 = set(np.array(THEMES)[np.argsort(-tR)[:3]])
        topS3 = set(np.array(THEMES)[np.argsort(-tS)[:3]])
        top3_jac = len(topR3 & topS3) / len(topR3 | topS3)
        top2 = len(set(np.array(THEMES)[np.argsort(-tR)[:2]]) &
                   set(np.array(THEMES)[np.argsort(-tS)[:2]])) / 2
        # expected-theme retention (best expected theme's rank)
        exp_rankR = min([rankR[t] for t in expected], default=None)
        exp_rankS = min([rankS[t] for t in expected], default=None)
        # redistribution
        dtheme = tS - tR
        gain_t = THEMES[int(dtheme.argmax())]; lose_t = THEMES[int(dtheme.argmin())]
        l1_theme = float(np.abs(dtheme).sum())
        dcomp = zs - zr
        gain_c = int(dcomp.argmax()); lose_c = int(dcomp.argmin())
        dmss = mS - mR
        gain_m = MOTIFS[int(dmss.argmax())]; lose_m = MOTIFS[int(dmss.argmin())]
        mssR3 = set(np.array(MOTIFS)[np.argsort(-mR)[:3]])
        mssS3 = set(np.array(MOTIFS)[np.argsort(-mS)[:3]])
        rows.append({
            "analyte": a, "family": fam, "expected_theme": "|".join(expected) or "mixed",
            "n_raman": raman_n[a], "n_sers": sers_n[a],
            "component_cosine": round(c_component, 4), "component_tier": tier(c_component),
            "theme_cosine": round(c_theme, 4), "theme_tier": tier(c_theme),
            "theme_cosine_distinct": round(c_theme_distinct, 4),
            "theme_null_mean": round(theme_null_mean, 4),
            "theme_separation": round(c_theme_distinct - theme_null_mean, 4),
            "self_is_nearest_theme": bool(self_rank == 1), "self_theme_rank": int(self_rank),
            "dominant_theme_match": bool(domR == domS),
            "raman_dominant": domR, "sers_dominant": domS,
            "top2_theme_overlap": round(top2, 3), "top3_theme_jaccard": round(top3_jac, 3),
            "expected_rank_raman": exp_rankR, "expected_rank_sers": exp_rankS,
            "expected_retained_top1": bool(exp_rankS == 1) if exp_rankS else None,
            "expected_retained_top3": bool(exp_rankS is not None and exp_rankS <= 3),
            "mss_cosine": round(c_mss, 4),
            "mss_top3_overlap": round(len(mssR3 & mssS3) / 3, 3),
            "dominant_mss_match": bool(MOTIFS[int(mR.argmax())] == MOTIFS[int(mS.argmax())]),
            "theme_redistribution": round(1 - c_theme, 4),
            "l1_theme_shift": round(l1_theme, 4), "theme_jsd": round(jsd(tR, tS), 4),
            "gained_theme": gain_t, "lost_theme": lose_t,
            "gained_component": f"c{gain_c}", "lost_component": f"c{lose_c}",
            "gained_mss": gain_m, "lost_mss": lose_m,
            "ood_raman": round(oodR, 4), "ood_sers": round(oodS, 4),
            "confidence_raman": round(confR, 4), "confidence_sers": round(confS, 4),
        })
    df = pd.DataFrame(rows)

    # ── the central-hypothesis quadrant ──
    # Theme axis uses the DISTINCTIVE (baseline-subtracted) cosine, NOT raw theme
    # cosine — raw is inflated to ~0.9 for every analyte by the shared compositional
    # baseline, so it cannot by itself discriminate preservation from background.
    def quadrant(r):
        comp_hi = r.component_cosine >= 0.55
        theme_hi = r.theme_cosine_distinct >= 0.50
        if comp_hi and theme_hi: return "Q1 identity preserved (both)"
        if not comp_hi and theme_hi: return "Q2 latent redistribution, theme survives"
        if comp_hi and not theme_hi: return "Q3 superficial coord match, theme changes"
        return "Q4 poor transfer (both)"
    df["quadrant"] = df.apply(quadrant, axis=1)
    df = df.sort_values("theme_cosine", ascending=False).reset_index(drop=True)
    df.to_csv(OUT / "tables/per_analyte_transfer_metrics.csv", index=False)

    # component vs theme (the key figure's data)
    df[["analyte", "family", "component_cosine", "theme_cosine", "theme_cosine_distinct",
        "theme_null_mean", "theme_separation", "self_is_nearest_theme",
        "dominant_theme_match", "quadrant"]].to_csv(
        OUT / "tables/component_vs_theme_preservation.csv", index=False)

    # by family
    fam = df.groupby("family").agg(
        n=("analyte", "size"),
        component_cosine_mean=("component_cosine", "mean"),
        theme_cosine_mean=("theme_cosine", "mean"),
        dominant_theme_preserved=("dominant_theme_match", "mean"),
        mss_cosine_mean=("mss_cosine", "mean"),
        expected_top3=("expected_retained_top3", "mean")).round(3).reset_index()
    fam.sort_values("theme_cosine_mean", ascending=False).to_csv(
        OUT / "tables/theme_preservation_by_family.csv", index=False)

    # dominant-theme confusion (raman dominant -> sers dominant)
    conf = df.groupby(["raman_dominant", "sers_dominant"]).size().reset_index(name="n")
    conf.to_csv(OUT / "tables/dominant_theme_confusion.csv", index=False)

    # mss preservation
    df[["analyte", "family", "mss_cosine", "mss_top3_overlap", "dominant_mss_match",
        "gained_mss", "lost_mss"]].to_csv(OUT / "tables/mss_preservation.csv", index=False)

    # ── summary ──
    q = df.quadrant.value_counts().to_dict()
    summary = {
        "atlas_fingerprint": atlas.meta["fingerprint"],
        "n_matched": int(len(df)),
        "component_cosine": {"median": round(float(df.component_cosine.median()), 4),
                             "mean": round(float(df.component_cosine.mean()), 4)},
        "theme_cosine_raw": {"median": round(float(df.theme_cosine.median()), 4),
                             "mean": round(float(df.theme_cosine.mean()), 4),
                             "note": "inflated by shared compositional baseline; not a stand-alone preservation measure"},
        "theme_cosine_distinct": {"median": round(float(df.theme_cosine_distinct.median()), 4),
                                  "mean": round(float(df.theme_cosine_distinct.mean()), 4),
                                  "note": "baseline-subtracted; the honest preservation signal"},
        "theme_null_mean": {"median": round(float(df.theme_null_mean.median()), 4),
                            "note": "distinctive cosine to OTHER analytes' SERS — the background floor"},
        "theme_separation": {"median": round(float(df.theme_separation.median()), 4),
                             "mean": round(float(df.theme_separation.mean()), 4),
                             "positive_count": int((df.theme_separation > 0).sum()),
                             "note": "distinct - null; >0 means the SERS theme profile resembles its OWN Raman profile more than a random one"},
        "self_is_nearest_theme": int(df.self_is_nearest_theme.sum()),
        "self_theme_rank": {"median": float(df.self_theme_rank.median()),
                            "mean": round(float(df.self_theme_rank.mean()), 2),
                            "top5_count": int((df.self_theme_rank <= 5).sum()),
                            "chance_median": round((len(df) + 1) / 2, 1),
                            "note": "rank of the correct analyte among all 51 by distinctive theme cosine; chance median ~26"},
        "dominant_theme_preserved": int(df.dominant_theme_match.sum()),
        "dominant_theme_preserved_rate": round(float(df.dominant_theme_match.mean()), 3),
        "expected_theme_top3_retained": int(df.expected_retained_top3.sum()),
        "mss_cosine": {"median": round(float(df.mss_cosine.median()), 4)},
        "quadrants": q,
        "n_latent_redistribution_theme_survives": int((df.quadrant.str.startswith("Q2")).sum()),
        "theme_gt_component": int((df.theme_cosine > df.component_cosine).sum()),
        "metric_definitions": {
            "component_cosine": "cosine(z_Raman, z_AgSERS) over the 24 NMF coordinates — LATENT FINGERPRINT PRESERVATION",
            "theme_cosine": "cosine(b_Raman, b_AgSERS) over the 11 biochemical THEME composition shares (RAW; baseline-inflated) — THEME PRESERVATION",
            "theme_cosine_distinct": "cosine of baseline-SUBTRACTED theme vectors (each minus mean Raman composition) — the honest theme-preservation signal",
            "theme_null_mean": "mean distinctive cosine of A's SERS theme profile to every OTHER analyte's — the background floor",
            "theme_separation": "theme_cosine_distinct - theme_null_mean; >0 = SERS resembles its own Raman theme more than a random analyte's",
            "self_is_nearest_theme": "the distinctive SERS theme profile's nearest Raman profile (over all analytes) is the SAME analyte",
            "mss_cosine": "cosine over the 12 biochemical MSS motif activations",
            "dominant_theme_match": "argmax(b_Raman) == argmax(b_AgSERS)",
            "expected_retained_top3": "family's expected theme is in the Ag-SERS top-3",
            "tiers": "manually-selected DESCRIPTIVE thresholds (0.80/0.65/0.45/0.25), not learned classes",
        },
    }
    (OUT / "artifacts/theme_preservation_summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "artifacts/per_analyte.json").write_text(df.to_json(orient="records", indent=1))

    print(json.dumps(summary, indent=2))
    print("\n=== the central test: component vs theme cosine (raw + distinctive + null) ===")
    print(df[["analyte", "family", "component_cosine", "theme_cosine", "theme_cosine_distinct",
              "theme_null_mean", "theme_separation", "self_is_nearest_theme",
              "dominant_theme_match", "quadrant"]].to_string(index=False))


if __name__ == "__main__":
    main()
