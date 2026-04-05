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
OUTPUT_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot4_cca_hcc_lm_serum_sers_takeaway"
)
REPORT_DIR = OUTPUT_ROOT / "report"


def _ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _fmt(x: float) -> str:
    return f"{float(x):.4f}"


def main() -> None:
    _ensure_dirs()

    input_df = pd.read_csv(PILOT4_ROOT / "tables" / "pilot4_input_verification.csv")
    baseline_df = pd.read_csv(PILOT4_ROOT / "tables" / "paper_style_baseline_metrics.csv")
    geometry_df = pd.read_csv(PILOT4_ROOT / "tables" / "spectral_vs_gaira_geometry_comparison.csv")
    interp_df = pd.read_csv(PILOT4_ROOT / "tables" / "class_interpretation_summary.csv")
    overlap_df = pd.read_csv(PILOT4_ROOT / "tables" / "overlap_zone_analysis.csv")
    bias_df = pd.read_csv(PILOT4_ROOT / "tables" / "serum_bias_axis_associations.csv")

    total_samples = int(input_df["sample_count"].sum())
    total_spectra = int(input_df["spectra_count"].sum())
    paper_samples = int(input_df["paper_expected_cases"].sum())
    paper_spectra = 9095

    lda_acc = float(baseline_df.loc[baseline_df["analysis_name"] == "lda_cv_accuracy", "metric_value"].iloc[0])
    lda_macro_auc = float(baseline_df.loc[baseline_df["analysis_name"] == "lda_macro_auc", "metric_value"].iloc[0])
    spectral_row = geometry_df.loc[geometry_df["space_name"] == "spectral"].iloc[0]
    bsv_row = geometry_df.loc[geometry_df["space_name"] == "bsv"].iloc[0]
    family_row = geometry_df.loc[geometry_df["space_name"] == "family"].iloc[0]

    bias_pc2 = bias_df[(bias_df["metric"] == "spectral_pc2") & (bias_df["feature"] == "substrate_adsorption_bias")]
    bias_bsv_pc1 = bias_df[(bias_df["metric"] == "bsv_pc1") & (bias_df["feature"] == "substrate_adsorption_bias")]
    smallmol_bsv_pc1 = bias_df[(bias_df["metric"] == "bsv_pc1") & (bias_df["feature"] == "small_molecule_metabolite")]

    top_overlap = overlap_df.sort_values("spectral_overlap_proxy", ascending=False).iloc[0]
    class_lines = []
    for row in interp_df.itertuples(index=False):
        class_lines.append(
            f"- `{row.class_label}`: dominant themes `{row.dominant_bsv_axes}` with family support from `{row.dominant_family_themes}`; "
            f"notable depletion in `{row.notable_depletions}`."
        )

    lines = [
        "# Pilot 4 Takeaway Note",
        "",
        "## 1. What Dataset This Was",
        "- Serum SERS on the liver-cancer benchmark spanning healthy controls (`HA`) and three cancer groups (`CCA`, `HCC`, `LM`).",
        "- This was a paper-aligned benchmark, not a discovery-only sandbox. The right question was whether GAIRA could reproduce the paper's geometry and add interpretation on top of it.",
        f"- Local ingest is per-spectrum and contains `{total_spectra}` spectra across `{total_samples}` samples.",
        f"- The paper-final cohort was smaller: `{paper_spectra}` spectra across `{paper_samples}` accepted cases.",
        "- Local-vs-paper mismatch is real and must stay attached to every Pilot 4 conclusion:",
        f"  local class counts were CCA `{int(input_df.loc[input_df['class_label_display']=='CCA','sample_count'].iloc[0])}`, "
        f"HA `{int(input_df.loc[input_df['class_label_display']=='HA','sample_count'].iloc[0])}`, "
        f"HCC `{int(input_df.loc[input_df['class_label_display']=='HCC','sample_count'].iloc[0])}`, "
        f"LM `{int(input_df.loc[input_df['class_label_display']=='LM','sample_count'].iloc[0])}`; "
        "each is slightly above the paper's accepted-case counts.",
        "",
        "## 2. What Replicated Successfully",
        "- The paper-style story did replicate in the important sense.",
        f"- Spectral PCA remained weakly separated in unsupervised space: healthy-vs-cancer silhouette `{_fmt(spectral_row['silhouette_healthy_vs_cancer'])}`, "
        f"4-class silhouette `{_fmt(spectral_row['silhouette_4class'])}`.",
        "- That is consistent with the paper's qualitative geometry: healthy-vs-cancer is easier than subtype-vs-subtype, but overlap remains substantial.",
        f"- The supervised baseline landed in the paper's reported range: LDA CV accuracy `{_fmt(lda_acc)}`, macro AUC `{_fmt(lda_macro_auc)}`.",
        "- That matters more than exact PCA numbers. It shows the local ingest is scientifically usable for a same-task comparison even with the cohort mismatch.",
        "",
        "## 3. What GAIRA Did And Did Not Do",
        "- GAIRA did not improve subtype geometry on this serum dataset.",
        f"- BSV 4-class silhouette was `{_fmt(bsv_row['silhouette_4class'])}` and family-space 4-class silhouette was `{_fmt(family_row['silhouette_4class'])}`, both worse than the raw spectral baseline.",
        f"- Family space did mildly help the binary healthy-vs-cancer split (`{_fmt(family_row['silhouette_healthy_vs_cancer'])}`), but that is not the hard part of the benchmark.",
        "- GAIRA's value here was interpretive, not geometric.",
        "- It converted the paper's peak-level narrative into a stable biochemical-theme reading centered on broad nucleic-acid and small-molecule / purine-adjacent support, while keeping explicit serum-domain caveats in view.",
        "- That interpretive layer is useful even though the class means remain more similar than we would want for clean subtype discrimination.",
        "",
        "## 4. Why Subtype Separation Remains Hard",
        "- The overlap looks real rather than accidental.",
        f"- The hardest overlap zone in spectral space was `{top_overlap['class_pair']}` with overlap proxy `{_fmt(top_overlap['spectral_overlap_proxy'])}`; all major subtype pairs retained shared purine-adjacent family support.",
        "- Shared hepatobiliary biology plus serum background is the simplest explanation supported by the outputs.",
        f"- Spectral PC2 still tracked substrate adsorption bias strongly (Spearman `{_fmt(bias_pc2['spearman_r'].iloc[0])}`), which means a major classical axis is not purely disease-specific.",
        f"- BSV PC1 was effectively a substrate/metabolite contrast axis: substrate bias Spearman `{_fmt(bias_bsv_pc1['spearman_r'].iloc[0])}`, "
        f"small-molecule metabolite Spearman `{_fmt(smallmol_bsv_pc1['spearman_r'].iloc[0])}`.",
        "- So the subtype problem is not just classifier weakness. A large fraction of the dominant variance is still serum-background and adsorption-structured.",
        "",
        "## 5. Why This Benchmark Is Still Valuable",
        "- It is stronger than SHINE because it is clinically labeled, directly comparable to a same-dataset literature baseline, and produces realistic supervised performance without special handling.",
        "- It is also a better stress test for GAIRA than a synthetic or weakly annotated serum panel, because it asks both geometry and interpretation questions at the same time.",
        "- The right next-step value is patient-level aggregation and patient-level uncertainty analysis, not more per-spectrum geometry optimism.",
        "",
        "## 6. Hard Conclusion",
        "- Pilot 4 was a success as a benchmark-validation and interpretation exercise.",
        "- It was not a success if the claim is that GAIRA solved liver-cancer subtype separation in serum SERS.",
        "- Claims we should make:",
        "  - the local dataset is usable as a literature-aligned serum benchmark, with explicit cohort-mismatch caveats;",
        "  - the paper-style supervised story replicates;",
        "  - GAIRA adds biochemical-theme interpretation beyond peak-by-peak assignments;",
        "  - subtype overlap appears to reflect real serum-domain difficulty, not just a missing classifier.",
        "- Claims we should not make:",
        "  - that GAIRA cleanly separates `CCA`, `HCC`, and `LM` in unsupervised space;",
        "  - that BSV geometry outperformed the paper's classical spectral geometry;",
        "  - that the local ingest exactly matches the paper-final cohort.",
        "",
        "## Class-Level Reading",
        *class_lines,
        "",
        "Bottom line: Pilot 4 is a credible, paper-grounded serum benchmark where GAIRA helps more with interpretation than with subtype separation. That is still strategically useful, and it makes this dataset a better benchmark than SHINE.",
    ]

    report_md = REPORT_DIR / "Pilot4_takeaway_note.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    build_pdf_report(report_md, [], REPORT_DIR / "Pilot4_takeaway_note.pdf")


if __name__ == "__main__":
    main()
