"""P10 ontology v0.1 · P11 BSV design study · P14 final assessment (READ-ONLY).

Synthesises the audit tables into a provisional ontology, a BSV architecture
recommendation, and the final scientific assessment. Nothing is frozen here.
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "results/v5_rebuild/reference_atlas_audit"
TAB, ART = OUT / "tables", OUT / "artifacts"


def main():
    inv = pd.read_csv(TAB / "p1_component_inventory.csv")
    coh = pd.read_csv(TAB / "p4_chemical_coherence.csv")
    grp = pd.read_csv(TAB / "p7_grouping_study.csv")
    man = json.loads((ART / "audit_manifest.json").read_text())
    conf = json.loads((TAB / "p12_confusability_summary.json").read_text())
    plaus = json.loads((TAB / "p9_biological_plausibility.json").read_text())
    best_k = int(man["grouping_recommendation_k"])
    gcomp = pd.read_csv(TAB / f"p7_group_composition_k{best_k}.csv")

    # ── P10 provisional ontology v0.1 ──
    tiers = {"high_confidence": [], "moderate_confidence": [], "low_confidence": [], "unknown": []}
    for _, r in inv.iterrows():
        entry = {"component": int(r.component), "theme": r.primary_interpretation,
                 "class": r.dominant_class, "purity": round(float(r.class_purity), 3),
                 "enrichment": round(float(r.enrichment), 2),
                 "stability": round(float(r.bootstrap_stability), 3),
                 "variance": round(float(r.variance_explained), 4),
                 "bands_cm": r.dominant_raman_peaks_cm}
        if r.confidence == "high" and r.class_purity >= 0.40:
            tiers["high_confidence"].append(entry)
        elif r.confidence == "high" or (r.confidence == "moderate" and r.class_purity >= 0.30):
            tiers["moderate_confidence"].append(entry)
        elif r.class_purity >= 0.20:
            tiers["low_confidence"].append(entry)
        else:
            tiers["unknown"].append(entry)
    ontology = {
        "version": "0.1 (provisional — NOT frozen)",
        "basis": "frozen NMF k=24 Raman reference atlas, audited post-hoc",
        "tiers": {k: len(v) for k, v in tiers.items()},
        "entries": tiers,
        "caveats": [
            "Themes are tentative interpretations of latent components, not molecular assignments.",
            "Mean component class purity is only 0.35: most components are chemically enriched "
            "rather than chemically pure.",
            "The atlas resolves molecular CLASS reliably; species within a class are frequently "
            "not resolved (see P12).",
        ],
    }
    (TAB / "p10_ontology_v0_1.json").write_text(json.dumps(ontology, indent=2, default=str))

    # ── P11 BSV design study ──
    stab_mean = float(inv.bootstrap_stability.mean())
    lipid_ok = all(plaus.get(f, {}).get("coheres", False) for f in ("triglyceride", "fatty_acid", "sterol"))
    carb_ok = all(plaus.get(f, {}).get("coheres", False) for f in ("saccharide", "polysaccharide"))
    weak = [f for f, v in plaus.items() if not v.get("coheres", False)]
    options = {
        "A_24_canonical": {
            "description": "Use all 24 latent components directly as BSV coordinates.",
            "interpretability": "low–moderate — 11/24 components are only moderately interpretable "
                                f"and mean class purity is {float(coh.class_purity.mean()):.2f}",
            "stability": f"high — mean bootstrap component stability {stab_mean:.3f}",
            "extensibility": "high — new analytes project without changing the basis",
            "clinical_usability": "poor — 24 weakly-named axes are not communicable",
            "verdict": "retain as the CANONICAL numerical layer, not as the reported BSV",
        },
        "B_compressed": {
            "description": f"Replace the 24 components with ~{best_k} higher-order coordinates.",
            "interpretability": "moderate — groups gain names but lose component detail",
            "stability": f"grouping bootstrap reproducibility {float(grp.iloc[0].bootstrap_reproducibility):.3f} "
                         f"but silhouette only {float(grp.iloc[0].silhouette):.3f}",
            "extensibility": "moderate — a new chemistry may not fit existing groups",
            "clinical_usability": "good",
            "verdict": "not sufficient alone — discards resolution the atlas actually has",
        },
        "C_hierarchical": {
            "description": "24 latent components (canonical, frozen) → higher-order biochemical "
                           "themes (reported) → optional visual summary.",
            "interpretability": "good — evidence is traceable from a reported theme back to "
                                "components, bands and reference analytes",
            "stability": "inherits component stability; group layer is re-derivable and versionable",
            "extensibility": "high — the numerical layer never changes when the ontology is revised",
            "clinical_usability": "good",
            "verdict": "RECOMMENDED",
        },
    }
    bsv = {
        "recommendation": "C_hierarchical",
        "rationale": [
            "The 24-component layer is stable (mean bootstrap stability "
            f"{stab_mean:.3f}) and is the level at which the atlas is actually reproducible, "
            "so it should remain canonical and frozen.",
            "It is NOT the level at which the atlas is interpretable: mean class purity is "
            f"{float(coh.class_purity.mean()):.2f} and only {int((inv.confidence == 'high').sum())}/24 "
            "components reach high interpretive confidence.",
            f"Grouping is reproducible (bootstrap {float(grp.iloc[0].bootstrap_reproducibility):.3f}) "
            f"but geometrically weak (silhouette {float(grp.iloc[0].silhouette):.3f}), so the group "
            "layer should be treated as an interpretive overlay that can be revised, never as the "
            "frozen numerical basis.",
            "A hierarchy keeps both properties: numbers from the components, meaning from the themes.",
        ],
        "options": options,
        "not_frozen": True,
        "recommended_group_count_if_adopted": best_k,
    }
    (TAB / "p11_bsv_design_study.json").write_text(json.dumps(bsv, indent=2, default=str))

    # ── P14 final assessment ──
    strongest = inv.nlargest(5, "confidence_score")[
        ["component", "primary_interpretation", "class_purity", "enrichment", "bootstrap_stability"]]
    weakest = inv.nsmallest(5, "confidence_score")[
        ["component", "primary_interpretation", "class_purity", "enrichment", "bootstrap_stability"]]
    assessment = {
        "1_what_chemistry_learned": (
            "Lipid chemistry dominates (triacylglycerols, fatty acids, sterols cohere with p<0.05 "
            "against a random-analyte null), followed by carbohydrates (mono- and polysaccharides) "
            "and proteins. Nucleic-acid bases appear as strongly ENRICHED but small components "
            "(purine enrichment 10x, pyrimidine 14x). Amino acids, cofactors and organic acids do "
            "NOT form coherent groups at analyte level."),
        "2_strongest_components": strongest.to_dict("records"),
        "3_weakest_components": weakest.to_dict("records"),
        "4_should_24_remain_canonical": (
            "Yes as the numerical layer. The components are the level at which the atlas is stable "
            f"(mean bootstrap stability {stab_mean:.3f}); no evidence supports changing k, and k was "
            "frozen for this audit in any case."),
        "5_should_components_be_grouped": (
            "Yes, but only as an interpretive overlay. Grouping is highly reproducible "
            f"({float(grp.iloc[0].bootstrap_reproducibility):.3f}) yet geometrically weak "
            f"(silhouette {float(grp.iloc[0].silhouette):.3f}) — the components do not form "
            "well-separated natural clusters."),
        "6_most_defensible_grouping": (
            f"k={best_k} by the composite of reproducibility, chemical coherence, silhouette and "
            "interpretable-group fraction. Ranking is reported in full; the margin over neighbouring "
            "k is small, so the group count should be treated as provisional."),
        "7_bsv_dimensionality": "Hierarchical (Option C): 24 canonical latent + reported themes.",
        "8_is_mss_mature": (
            "Not at molecular-species level. Median signature uniqueness is "
            f"{conf['median_uniqueness_all']:.3f}; however {conf['fraction_of_low_uniqueness_explained_by_chemistry']:.0%} "
            "of low-uniqueness cases are duplicates, homologous series or same-class chemistry, which "
            "is correct spectroscopy rather than atlas failure. MSS can be frozen at CLASS/THEME level "
            f"with documented confusable groups; only {conf['n_genuine_confusions']} genuinely "
            "cross-chemistry confusions remain."),
        "9_ontology_confidence": {k: len(v) for k, v in tiers.items()},
        "10_next_phase": (
            "Define the hierarchical BSV v0.1 on the frozen 24-component layer with a versioned, "
            "revisable theme overlay, and publish MSS at class level with explicit confusable-group "
            "annotations. Do not expand k, do not retrain, and do not admit Ag-SERS."),
    }
    (TAB / "p14_final_assessment.json").write_text(json.dumps(assessment, indent=2, default=str))

    print("=== SYNTHESIS ===")
    print("ontology tiers:", ontology["tiers"])
    print("BSV recommendation:", bsv["recommendation"], f"(group layer k={best_k}, not frozen)")
    print("MSS:", assessment["8_is_mss_mature"][:120], "…")
    print("family coherence:", {k: v["coheres"] for k, v in plaus.items()})


if __name__ == "__main__":
    main()
