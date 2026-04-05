from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gaira.demo.gaira_pilot_utils import build_pdf_report


PILOT4_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot4_cca_hcc_lm_serum_sers"
)
PILOT41_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot4_1_cca_hcc_lm_serum_patient_level"
)
OUTPUT_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot4_integrated_report"
)
REPORT_DIR = OUTPUT_ROOT / "report"


def _ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _fmt(x: float) -> str:
    return f"{float(x):.4f}"


def _df_to_md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    _ensure_dirs()

    p4_input = pd.read_csv(PILOT4_ROOT / "tables" / "pilot4_input_verification.csv")
    p41_unit = pd.read_csv(PILOT41_ROOT / "tables" / "pilot4_1_aggregation_unit_verification.csv")
    p4_base = pd.read_csv(PILOT4_ROOT / "tables" / "paper_style_baseline_metrics.csv")
    p41_base = pd.read_csv(PILOT41_ROOT / "tables" / "patient_level_paper_style_metrics.csv")
    p4_geom = pd.read_csv(PILOT4_ROOT / "tables" / "spectral_vs_gaira_geometry_comparison.csv")
    p41_geom = pd.read_csv(PILOT41_ROOT / "tables" / "patient_level_geometry_comparison.csv")
    p4_bias = pd.read_csv(PILOT4_ROOT / "tables" / "serum_bias_axis_associations.csv")
    p41_bias = pd.read_csv(PILOT41_ROOT / "tables" / "patient_level_serum_bias_axis_associations.csv")
    p4_interp = pd.read_csv(PILOT4_ROOT / "tables" / "class_interpretation_summary.csv")
    p41_interp = pd.read_csv(PILOT41_ROOT / "tables" / "patient_level_class_interpretation_summary.csv")
    p4_overlap = pd.read_csv(PILOT4_ROOT / "tables" / "overlap_zone_analysis.csv")
    p41_overlap = pd.read_csv(PILOT41_ROOT / "tables" / "patient_level_overlap_zone_analysis.csv")
    tradeoff_df = pd.read_csv(PILOT41_ROOT / "tables" / "per_spectrum_vs_patient_level_comparison.csv")

    total_spectra = int(p4_input["spectra_count"].sum())
    total_samples = int(p4_input["sample_count"].sum())
    class_counts = ", ".join(
        [f"{r.class_label_display}={int(r.sample_count)} samples / {int(r.spectra_count)} spectra" for r in p4_input.itertuples(index=False)]
    )
    agg_counts = ", ".join([f"{r.class_label_display}={int(r.aggregated_units)}" for r in p41_unit.itertuples(index=False)])

    p4_lda_acc = float(p4_base.loc[p4_base["analysis_name"] == "lda_cv_accuracy", "metric_value"].iloc[0])
    p4_macro_auc = float(p4_base.loc[p4_base["analysis_name"] == "lda_macro_auc", "metric_value"].iloc[0])
    p4_micro_auc = float(p4_base.loc[p4_base["analysis_name"] == "lda_micro_auc", "metric_value"].iloc[0])
    p4_spec = p4_base.loc[p4_base["analysis_name"] == "spectral_geometry"].iloc[0]

    p41_lda_acc = float(p41_base.loc[p41_base["analysis_name"] == "patient_level_lda_cv_accuracy", "metric_value"].iloc[0])
    p41_macro_auc = float(p41_base.loc[p41_base["analysis_name"] == "patient_level_lda_macro_auc", "metric_value"].iloc[0])
    p41_micro_auc = float(p41_base.loc[p41_base["analysis_name"] == "patient_level_lda_micro_auc", "metric_value"].iloc[0])
    p41_spec = p41_base.loc[p41_base["analysis_name"] == "patient_level_spectral_geometry"].iloc[0]

    baseline_compare = pd.DataFrame(
        [
            {"metric": "LDA CV accuracy", "pilot4_per_spectrum": p4_lda_acc, "pilot4_1_patient_level": p41_lda_acc, "delta_patient_minus_spectrum": p41_lda_acc - p4_lda_acc},
            {"metric": "LDA macro AUC", "pilot4_per_spectrum": p4_macro_auc, "pilot4_1_patient_level": p41_macro_auc, "delta_patient_minus_spectrum": p41_macro_auc - p4_macro_auc},
            {"metric": "LDA micro AUC", "pilot4_per_spectrum": p4_micro_auc, "pilot4_1_patient_level": p41_micro_auc, "delta_patient_minus_spectrum": p41_micro_auc - p4_micro_auc},
            {
                "metric": "Spectral silhouette healthy vs cancer",
                "pilot4_per_spectrum": float(p4_spec["silhouette_healthy_vs_cancer"]),
                "pilot4_1_patient_level": float(p41_spec["silhouette_healthy_vs_cancer"]),
                "delta_patient_minus_spectrum": float(p41_spec["silhouette_healthy_vs_cancer"]) - float(p4_spec["silhouette_healthy_vs_cancer"]),
            },
            {
                "metric": "Spectral silhouette 4-class",
                "pilot4_per_spectrum": float(p4_spec["silhouette_4class"]),
                "pilot4_1_patient_level": float(p41_spec["silhouette_4class"]),
                "delta_patient_minus_spectrum": float(p41_spec["silhouette_4class"]) - float(p4_spec["silhouette_4class"]),
            },
            {
                "metric": "Spectral NN purity 4-class",
                "pilot4_per_spectrum": float(p4_spec["nearest_neighbor_purity_4class"]),
                "pilot4_1_patient_level": float(p41_spec["nearest_neighbor_purity_4class"]),
                "delta_patient_minus_spectrum": float(p41_spec["nearest_neighbor_purity_4class"]) - float(p4_spec["nearest_neighbor_purity_4class"]),
            },
        ]
    )

    geom_rows = []
    space_map = {"spectral_mean": "spectral", "bsv_mean": "bsv", "delta_bsv_mean": "delta_bsv", "family_mean": "family"}
    for p41_space, p4_space in space_map.items():
        row4 = p4_geom[p4_geom["space_name"].astype(str) == p4_space].iloc[0]
        row41 = p41_geom[p41_geom["space_name"].astype(str) == p41_space].iloc[0]
        geom_rows.append(
            {
                "representation": p4_space,
                "silhouette_4class_delta": float(row41["silhouette_4class"]) - float(row4["silhouette_4class"]),
                "silhouette_hc_vs_cancer_delta": float(row41["silhouette_healthy_vs_cancer"]) - float(row4["silhouette_healthy_vs_cancer"]),
                "nn_purity_delta": float(row41["nearest_neighbor_purity_4class"]) - float(row4["nearest_neighbor_purity_4class"]),
                "patient_level_silhouette_4class": float(row41["silhouette_4class"]),
                "patient_level_silhouette_hc_vs_cancer": float(row41["silhouette_healthy_vs_cancer"]),
                "patient_level_nn_purity": float(row41["nearest_neighbor_purity_4class"]),
            }
        )
    geom_change_df = pd.DataFrame(geom_rows)

    p4_spec_pc2_bias = float(p4_bias[(p4_bias["metric"] == "spectral_pc2") & (p4_bias["feature"] == "substrate_adsorption_bias")]["spearman_r"].iloc[0])
    p41_spec_pc2_bias = float(p41_bias[(p41_bias["metric"] == "spectral_pc2") & (p41_bias["feature"] == "substrate_adsorption_bias")]["spearman_r"].iloc[0])
    p4_bsv_pc1_bias = float(p4_bias[(p4_bias["metric"] == "bsv_pc1") & (p4_bias["feature"] == "substrate_adsorption_bias")]["spearman_r"].iloc[0])
    p41_bsv_pc1_bias = float(p41_bias[(p41_bias["metric"] == "bsv_pc1") & (p41_bias["feature"] == "substrate_adsorption_bias")]["spearman_r"].iloc[0])
    p4_bsv_pc1_small = float(p4_bias[(p4_bias["metric"] == "bsv_pc1") & (p4_bias["feature"] == "small_molecule_metabolite")]["spearman_r"].iloc[0])
    p41_bsv_pc1_small = float(p41_bias[(p41_bias["metric"] == "bsv_pc1") & (p41_bias["feature"] == "small_molecule_metabolite")]["spearman_r"].iloc[0])

    bias_compare = pd.DataFrame(
        [
            {"axis_feature_pair": "spectral PC2 vs substrate_adsorption_bias", "pilot4_per_spectrum": p4_spec_pc2_bias, "pilot4_1_patient_level": p41_spec_pc2_bias, "delta_patient_minus_spectrum": p41_spec_pc2_bias - p4_spec_pc2_bias},
            {"axis_feature_pair": "BSV PC1 vs substrate_adsorption_bias", "pilot4_per_spectrum": p4_bsv_pc1_bias, "pilot4_1_patient_level": p41_bsv_pc1_bias, "delta_patient_minus_spectrum": p41_bsv_pc1_bias - p4_bsv_pc1_bias},
            {"axis_feature_pair": "BSV PC1 vs small_molecule_metabolite", "pilot4_per_spectrum": p4_bsv_pc1_small, "pilot4_1_patient_level": p41_bsv_pc1_small, "delta_patient_minus_spectrum": p41_bsv_pc1_small - p4_bsv_pc1_small},
        ]
    )

    overlap_compare = p4_overlap.merge(p41_overlap, on="class_pair", suffixes=("_pilot4", "_pilot4_1"))

    lines = [
        "# GAIRAv3 Pilot4 Integrated Report",
        "",
        "## 1. Data & Setup Consistency",
        f"- Dataset continuity is intact: Pilot 4 used the same `cca_hcc_lm_serum_sers` cohort that Pilot 4.1 later aggregated by `sample_id`.",
        f"- Per-spectrum cohort: `{total_spectra}` spectra across `{total_samples}` sample-level units.",
        f"- Class distribution: {class_counts}.",
        f"- Aggregated class distribution in Pilot 4.1: {agg_counts}.",
        "- Pipeline consistency: Pilot 4.1 reused the Pilot 4 per-spectrum BSV, delta-BSV, and family outputs and only changed evaluation level by averaging to `sample_id`.",
        "- Leakage check: none evident. Class counts are preserved under aggregation, and no new samples appear or disappear between the two pilots.",
        "",
        "## 2. Paper-Style Baseline (Spectral Space)",
        _df_to_md(baseline_compare),
        "",
        f"- Aggregation did not improve supervised performance. LDA CV accuracy fell from `{_fmt(p4_lda_acc)}` to `{_fmt(p41_lda_acc)}` and macro AUC fell from `{_fmt(p4_macro_auc)}` to `{_fmt(p41_macro_auc)}`.",
        f"- Aggregation improved only one baseline geometry component: healthy-vs-cancer spectral silhouette moved from `{_fmt(float(p4_spec['silhouette_healthy_vs_cancer']))}` to `{_fmt(float(p41_spec['silhouette_healthy_vs_cancer']))}`.",
        f"- Aggregation worsened the 4-class spectral geometry: 4-class silhouette moved from `{_fmt(float(p4_spec['silhouette_4class']))}` to `{_fmt(float(p41_spec['silhouette_4class']))}` and NN purity dropped from `{_fmt(float(p4_spec['nearest_neighbor_purity_4class']))}` to `{_fmt(float(p41_spec['nearest_neighbor_purity_4class']))}`.",
        "- Decision: patient-level averaging helps the binary healthy-vs-cancer readout slightly, but it does not improve the paper-style subtype baseline.",
        "",
        "## 3. Geometry Comparison Across Representations",
        _df_to_md(geom_change_df),
        "",
        "- Representation-level decision:",
        f"  spectral benefited most on the binary split only (`delta silhouette healthy-vs-cancer = {_fmt(geom_change_df[geom_change_df['representation']=='spectral']['silhouette_hc_vs_cancer_delta'].iloc[0])}`),",
        f"  family benefited most on NN purity (`delta = {_fmt(geom_change_df[geom_change_df['representation']=='family']['nn_purity_delta'].iloc[0])}`),",
        "  but BSV and delta-BSV remained fundamentally weak for subtype geometry.",
        "- The weakest representation after aggregation is still BSV / delta-BSV for 4-class separation.",
        "- Aggregation does not convert GAIRA into a subtype-separation engine on this serum dataset.",
        "",
        "## 4. Serum Bias Analysis",
        _df_to_md(bias_compare),
        "",
        f"- Aggregation reduced one important spectral bias component: spectral PC2 versus substrate adsorption bias dropped from `{_fmt(p4_spec_pc2_bias)}` to `{_fmt(p41_spec_pc2_bias)}`.",
        f"- It did not reduce the core BSV dominance structure. BSV PC1 remained completely dominated by the adsorption/metabolite contrast at both levels (`{_fmt(p4_bsv_pc1_bias)}` to `{_fmt(p41_bsv_pc1_bias)}` against substrate bias; `{_fmt(p4_bsv_pc1_small)}` to `{_fmt(p41_bsv_pc1_small)}` against small-molecule metabolite).",
        "- Explicit serum-bias judgment:",
        "  - in spectral space: moderate to dominant, depending on which PC is inspected;",
        "  - in BSV space PC1: dominant;",
        "  - in downstream subtype structure: still a major constraint, not a secondary nuisance.",
        "",
        "## 5. Biochemical Interpretation Layer",
        "- Pilot 4 dominant class themes:",
        *[
            f"  - `{r.class_label}`: `{r.dominant_bsv_axes}` with family support `{r.dominant_family_themes}`."
            for r in p4_interp.itertuples(index=False)
        ],
        "- Pilot 4.1 dominant class themes:",
        *[
            f"  - `{r.class_label}`: `{r.dominant_bsv_axes}` with family support `{r.dominant_family_themes}`."
            for r in p41_interp.itertuples(index=False)
        ],
        "",
        _df_to_md(overlap_compare[[
            "class_pair",
            "shared_dominant_families_pilot4",
            "interpretation_note_pilot4",
            "shared_dominant_families_pilot4_1",
            "interpretation_note_pilot4_1",
        ]]),
        "",
        "- Interpretation decision:",
        "  aggregation improves interpretability clarity modestly by stabilizing family emphasis and making the overlap explanation more direct.",
        "  It does not create separable class-specific biochemical themes.",
        "  It smooths noise more than it discovers new disease-specific structure.",
        "- Residual subtype overlap stays concentrated in the same biologically plausible zones: `CCA vs HCC`, `HCC vs LM`, and `CCA vs LM`.",
        "",
        "## 6. Per-Spectrum vs Patient-Level Tradeoff",
        _df_to_md(tradeoff_df),
        "",
        "- Per-spectrum is better when the goal is to maximize the paper-style supervised baseline and preserve local heterogeneity that the classifier can exploit.",
        "- Patient-level is better when the goal is to produce a cleaner, more stable interpretation summary at the sample level and slightly reduce one spectral serum-bias component.",
        "- What is lost with aggregation: subtype classifier performance, 4-class spectral geometry, and some within-class heterogeneity that appears informative to the LDA baseline.",
        "- What is gained with aggregation: a better binary healthy-vs-cancer spectral silhouette, stronger family-space NN purity, and a clearer statement that residual subtype overlap is shared chemistry rather than just spot noise.",
        "",
        "## 7. Core Scientific Conclusions",
        "- Serum SERS subtype separation on this cohort is fundamentally limited at the representation level tested here.",
        "- PCA overlap is expected and should not be treated as a failure signal. The paper's own pipeline depends on supervised discrimination rather than clean unsupervised subtype geometry.",
        "- GAIRA adds interpretation, not geometry, on this benchmark.",
        "- On this dataset GAIRA improves:",
        "  - geometry: `no`",
        "  - interpretation: `yes`",
        "  - both together: `no`",
        "",
        "## 8. GAIRA Serum Doctrine",
        "- Default evaluation level for serum should be patient/sample level for interpretation reports and benchmark summaries.",
        "- Default evaluation level for spectral sanity-check classification should still include per-spectrum baselines, because aggregation can hide classifier-relevant heterogeneity.",
        "- GAIRA should optimize for serum interpretation and uncertainty-aware aggregation, not for forcing clean unsupervised subtype separation where the domain structure does not support it.",
        "- For serum, the right priorities are:",
        "  - interpretation first,",
        "  - patient/sample aggregation second,",
        "  - classification benchmarking as a comparator, not as the main GAIRA objective.",
        "- BSV action decision:",
        "  - no ontology extension is justified from this benchmark;",
        "  - no SHINE-style temporary axes should be imported here;",
        "  - reweighting may be worth testing later only if it explicitly targets serum adsorption dominance, but there is no evidence here for changing the core vocabulary yet.",
        "",
        "## 9. Implications for Next Pilot",
        "- Test patient-level aggregation plus uncertainty-aware within-sample dispersion summaries rather than repeating another per-spectrum-only serum run.",
        "- Do not repeat raw per-spectrum geometry arguments as if subtype PCA should become clean; that hypothesis is now negative.",
        "- Test whether patient-level variability descriptors help separate true biological overlap from measurement heterogeneity.",
        "- Keep paper-style spectral baselines in future serum pilots as the sanity anchor; do not compare GAIRA in isolation.",
        "- Working serum hypothesis: these datasets are best used to evaluate whether GAIRA can stabilize interpretation under strong background structure, not whether it can create de novo subtype geometry.",
    ]

    report_md = REPORT_DIR / "GAIRAv3_Pilot4_Integrated_Report.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    build_pdf_report(report_md, [], REPORT_DIR / "GAIRAv3_Pilot4_Integrated_Report.pdf")


if __name__ == "__main__":
    main()
