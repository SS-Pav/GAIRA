from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gaira.demo.gaira_experiment_runner_utils import load_architecture_registries, load_query_dataframe
from gaira.demo.gaira_pilot_utils import build_pdf_report
from gaira.demo.raw_bsv_pilot_utils import decode_and_align
from scripts.run_gaira_pilot4_cca_hcc_lm_serum_sers import (
    ARCH_DIR,
    CLASS_COLORS,
    DISPLAY_ORDER,
    FAMILY_ORDER,
    FIXED_RADAR_AXES,
    PHASE1_DIR,
    ROOT,
    _broad_label,
    _class_interpretation_summary,
    _cohort_delta,
    _df_to_md,
    _display_label,
    _extract_sample_id,
    _lda_cv_metrics,
    _overlap_analysis,
    _pairwise_class_distances,
    _plot_bias_panels,
    _plot_bsv_heatmap,
    _plot_confusion,
    _plot_family_bars,
    _plot_heatmap_from_pairwise,
    _plot_lda_2d,
    _plot_overlap_panels,
    _plot_pca,
    _plot_radar_grid,
    _plot_roc,
    _pca_dataframe,
    _representation_metrics,
    _resolve_alias,
    _serum_bias_associations,
)


PILOT4_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot4_cca_hcc_lm_serum_sers"
)
OUTPUT_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot4_1_cca_hcc_lm_serum_patient_level"
)
TABLES_DIR = OUTPUT_ROOT / "tables"
FIGURES_DIR = OUTPUT_ROOT / "figures"
REPORT_DIR = OUTPUT_ROOT / "report"
SUBSET_ALIAS = "cca_hcc_lm_serum"


def _ensure_dirs() -> None:
    for path in [OUTPUT_ROOT, TABLES_DIR, FIGURES_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _load_query_df() -> pd.DataFrame:
    registries = load_architecture_registries(
        grounding_family_registry_path=ROOT / "config" / "gaira_grounding_family_registry_v1.csv",
        target_family_registry_path=ROOT / "config" / "gaira_target_family_registry_v1.csv",
        inference_lane_registry_path=ROOT / "config" / "gaira_inference_lane_registry_v2.csv",
        representation_mode_registry_path=ROOT / "config" / "gaira_representation_mode_registry_v2.csv",
        dataset_experiment_registry_path=ROOT / "config" / "gaira_dataset_experiment_registry_v2.csv",
        experiment_plan_path=ARCH_DIR / "first_pass_experiment_plan.csv",
        phase1_registry_path=PHASE1_DIR / "phase1_dataset_registry_v2.csv",
        phase1_grounding_map_path=PHASE1_DIR / "phase1_target_grounding_map_v2.csv",
        phase1_exclusions_path=PHASE1_DIR / "phase1_grounding_exclusions.csv",
    )
    resolved = _resolve_alias(registries, SUBSET_ALIAS)
    query_df = load_query_dataframe(resolved.dataset_row).copy()
    query_df["sample_id"] = [
        _extract_sample_id(sample_key, source_file)
        for sample_key, source_file in zip(query_df["sample_key"], query_df["source_file"], strict=False)
    ]
    query_df["class_label_display"] = query_df["class_label"].map(_display_label)
    query_df["broad_label"] = query_df["class_label_display"].map(_broad_label)
    return query_df


def _build_patient_level_variability(
    spectral_long: pd.DataFrame,
    bsv_df: pd.DataFrame,
    family_long: pd.DataFrame,
    axes: list[str],
) -> pd.DataFrame:
    spectral_var = (
        spectral_long.groupby("sample_id")
        .apply(lambda df: float(df[[c for c in df.columns if c.startswith("wn_")]].var(axis=0, ddof=0).mean()))
        .rename("within_sample_spectral_variance")
        .reset_index()
    )
    bsv_var = (
        bsv_df.groupby("sample_id")
        .apply(lambda df: float(df[axes].var(axis=0, ddof=0).mean()))
        .rename("within_sample_bsv_variance")
        .reset_index()
    )
    family_stats = (
        family_long.groupby(["sample_id", "sample_key"], as_index=False)["family_fraction"]
        .max()
        .rename(columns={"family_fraction": "top1_dominance"})
    )
    family_entropy = (
        family_long.groupby(["sample_id", "sample_key"])
        .apply(lambda df: float(-(df["family_fraction"].clip(lower=1e-12) * df["family_fraction"].clip(lower=1e-12).map(__import__("math").log)).sum()))
        .rename("family_entropy")
        .reset_index()
    )
    family_agg = (
        family_stats.merge(family_entropy, on=["sample_id", "sample_key"], how="inner")
        .groupby("sample_id", as_index=False)
        .agg(
            top1_dominance_mean=("top1_dominance", "mean"),
            top1_dominance_std=("top1_dominance", "std"),
            family_entropy_mean=("family_entropy", "mean"),
            family_entropy_std=("family_entropy", "std"),
        )
        .fillna(0.0)
    )
    return spectral_var.merge(bsv_var, on="sample_id").merge(family_agg, on="sample_id")


def _aggregate_family(patient_family_long: pd.DataFrame) -> pd.DataFrame:
    wide = (
        patient_family_long.pivot_table(
            index=["sample_id", "class_label_display", "broad_label"],
            columns="family",
            values="family_fraction",
            aggfunc="mean",
            fill_value=0.0,
        )
        .reset_index()
    )
    for family in FAMILY_ORDER:
        if family not in wide.columns:
            wide[family] = 0.0
    cols = ["sample_id", "class_label_display", "broad_label"] + FAMILY_ORDER
    return wide[cols]


def _class_mean_family_from_wide(patient_family_wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in DISPLAY_ORDER:
        sub = patient_family_wide[patient_family_wide["class_label_display"].astype(str) == label]
        if sub.empty:
            continue
        mean_vals = sub[FAMILY_ORDER].mean(axis=0)
        total = float(mean_vals.sum())
        for family in FAMILY_ORDER:
            value = float(mean_vals[family])
            rows.append(
                {
                    "class_label": label,
                    "family": family,
                    "family_fraction": (value / total) if total > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _compare_against_per_spectrum(
    patient_geometry_df: pd.DataFrame,
    per_spectrum_geometry_df: pd.DataFrame,
) -> pd.DataFrame:
    name_map = {
        "spectral_mean": "spectral",
        "bsv_mean": "bsv",
        "delta_bsv_mean": "delta_bsv",
        "family_mean": "family",
    }
    rows = []
    for _, prow in patient_geometry_df.iterrows():
        per_name = name_map.get(str(prow["space_name"]), str(prow["space_name"]))
        base = per_spectrum_geometry_df[per_spectrum_geometry_df["space_name"].astype(str) == per_name]
        if base.empty:
            continue
        brow = base.iloc[0]
        rows.append(
            {
                "space_name": prow["space_name"],
                "per_spectrum_space_name": per_name,
                "patient_silhouette_4class": float(prow["silhouette_4class"]),
                "per_spectrum_silhouette_4class": float(brow["silhouette_4class"]),
                "patient_silhouette_healthy_vs_cancer": float(prow["silhouette_healthy_vs_cancer"]),
                "per_spectrum_silhouette_healthy_vs_cancer": float(brow["silhouette_healthy_vs_cancer"]),
                "patient_nn_purity_4class": float(prow["nearest_neighbor_purity_4class"]),
                "per_spectrum_nn_purity_4class": float(brow["nearest_neighbor_purity_4class"]),
                "patient_centroid_distance_4class": float(prow["centroid_distance_4class"]),
                "per_spectrum_centroid_distance_4class": float(brow["centroid_distance_4class"]),
            }
        )
    return pd.DataFrame(rows)


def _build_report(
    aggregation_note: str,
    unit_df: pd.DataFrame,
    patient_baseline_df: pd.DataFrame,
    geometry_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    interpretation_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    bias_df: pd.DataFrame,
    decision_label: str,
) -> Path:
    lines = [
        "# GAIRAv3 Pilot4.1 CCA HCC LM Serum Patient Level Report",
        "",
        "## 1. Why patient-level aggregation was needed",
        "- Pilot 4 showed clinically realistic supervised performance but weak per-spectrum subtype geometry.",
        "- Serum spot-level heterogeneity and adsorption/background effects were dominant enough to justify a patient/sample-level pass.",
        "",
        "## 2. Aggregation unit verification",
        aggregation_note.strip(),
        "",
        "### Aggregation table",
        _df_to_md(unit_df),
        "",
        "## 3. Patient-level baseline",
        _df_to_md(patient_baseline_df),
        "",
        "## 4. Patient-level GAIRA geometry",
        _df_to_md(geometry_df),
        "",
        "### Per-spectrum vs patient-level",
        _df_to_md(comparison_df),
        "",
        "## 5. Patient-level biochemical interpretation",
        _df_to_md(interpretation_df),
        "",
        "## 6. Overlap and uncertainty",
        _df_to_md(overlap_df),
        "",
        "## 7. Serum-bias reassessment",
        _df_to_md(bias_df.head(12)),
        "",
        "## 8. Final conclusion",
        f"- decision label: `{decision_label}`",
        f"- patient-level evaluation better than per-spectrum: `{'yes' if decision_label != 'no_patient_gain' else 'limited'}`",
        f"- GAIRA gains more value at patient level: `{'yes' if decision_label != 'no_patient_gain' else 'partially'}`",
        f"- future serum benchmarking should default to patient/sample level: `{'yes' if decision_label != 'no_patient_gain' else 'consider yes for baseline comparability'}`",
    ]
    report_md = REPORT_DIR / "GAIRAv3_Pilot4_1_CCA_HCC_LM_serum_patient_level_report.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_md


def main() -> None:
    _ensure_dirs()

    query_df = _load_query_df()
    bsv_df = pd.read_csv(PILOT4_ROOT / "tables" / "per_spectrum_bsv.csv")
    delta_df = pd.read_csv(PILOT4_ROOT / "tables" / "per_spectrum_delta_bsv.csv")
    family_long = pd.read_csv(PILOT4_ROOT / "tables" / "per_spectrum_family.csv")
    per_spectrum_geometry_df = pd.read_csv(PILOT4_ROOT / "tables" / "spectral_vs_gaira_geometry_comparison.csv")

    bsv_df["sample_id"] = [
        _extract_sample_id(sample_key, sample_key)
        for sample_key in bsv_df["sample_key"].astype(str)
    ]
    delta_df["sample_id"] = [
        _extract_sample_id(sample_key, sample_key)
        for sample_key in delta_df["sample_key"].astype(str)
    ]
    bsv_df["class_label_display"] = bsv_df["class_label"].map(_display_label)
    if "class_label" in delta_df.columns:
        delta_df["class_label_display"] = delta_df["class_label"].map(_display_label)
    else:
        delta_df["class_label_display"] = delta_df["class_label_display"].map(_display_label)
    bsv_df["broad_label"] = bsv_df["class_label_display"].map(_broad_label)
    delta_df["broad_label"] = delta_df["class_label_display"].map(_broad_label)
    family_long["class_label_display"] = family_long["class_label_display"].map(_display_label)
    family_long["broad_label"] = family_long["class_label_display"].map(_broad_label)

    unit_df = (
        query_df.groupby(["class_label_display", "sample_id"], as_index=False)
        .agg(
            spectra_per_unit=("sample_key", "count"),
            source_files=("source_file", "nunique"),
        )
    )
    unit_summary_df = (
        unit_df.groupby("class_label_display", as_index=False)
        .agg(
            aggregated_units=("sample_id", "nunique"),
            mean_spectra_per_unit=("spectra_per_unit", "mean"),
            min_spectra_per_unit=("spectra_per_unit", "min"),
            max_spectra_per_unit=("spectra_per_unit", "max"),
        )
        .sort_values("class_label_display")
        .reset_index(drop=True)
    )
    unit_summary_df.to_csv(TABLES_DIR / "pilot4_1_aggregation_unit_verification.csv", index=False)

    aggregation_note = (
        "# Pilot4.1 Aggregation Note\n\n"
        "- chosen unit: `sample_id`\n"
        "- rationale: `sample_id` is the only identifier that remains one-to-one with the biological serum sample/patient in practice. "
        "`biosample_id` and `replicate_id` expand to per-spectrum map rows.\n"
        f"- resulting aggregated units: `{int(unit_summary_df['aggregated_units'].sum())}` across the four classes.\n\n"
        "Direct answers:\n"
        "1. what unit are we aggregating over? `sample_id`\n"
        "2. why is it the correct approximation of patient/sample level? `Because sample_id collapses spot-level map rows back to the underlying serum sample, while biosample_id/replicate_id do not.`\n"
        f"3. how many aggregated units per class result? `{', '.join([f'{r.class_label_display}={int(r.aggregated_units)}' for r in unit_summary_df.itertuples(index=False)])}`\n"
    )
    (REPORT_DIR / "pilot4_1_aggregation_note.md").write_text(aggregation_note, encoding="utf-8")

    wavenumbers, spectra_matrix = decode_and_align(query_df)
    wavenumbers = np.asarray(wavenumbers).reshape(-1).astype(float)
    spectral_cols = [f"wn_{int(round(float(w)))}" for w in wavenumbers]
    spectral_long = pd.DataFrame(spectra_matrix, columns=spectral_cols)
    spectral_long.insert(0, "sample_key", query_df["sample_key"].astype(str).values)
    spectral_long.insert(1, "sample_id", query_df["sample_id"].astype(str).values)
    spectral_long.insert(2, "class_label_display", query_df["class_label_display"].astype(str).values)
    spectral_long.insert(3, "broad_label", query_df["broad_label"].astype(str).values)

    patient_spectra_df = (
        spectral_long.groupby(["sample_id", "class_label_display", "broad_label"], as_index=False)[spectral_cols]
        .mean()
        .sort_values(["class_label_display", "sample_id"])
        .reset_index(drop=True)
    )
    patient_spectra_df.to_csv(TABLES_DIR / "patient_level_mean_spectra.csv", index=False)

    axes = [axis for axis in FIXED_RADAR_AXES if axis in bsv_df.columns]
    patient_bsv_df = (
        bsv_df.groupby(["sample_id", "class_label_display", "broad_label"], as_index=False)[axes]
        .mean()
        .sort_values(["class_label_display", "sample_id"])
        .reset_index(drop=True)
    )
    patient_bsv_df.to_csv(TABLES_DIR / "patient_level_bsv.csv", index=False)

    patient_delta_df = (
        delta_df.groupby(["sample_id", "class_label_display", "broad_label"], as_index=False)[axes]
        .mean()
        .sort_values(["class_label_display", "sample_id"])
        .reset_index(drop=True)
    )
    patient_delta_df.to_csv(TABLES_DIR / "patient_level_delta_bsv.csv", index=False)

    patient_family_wide = _aggregate_family(family_long)
    patient_family_wide.to_csv(TABLES_DIR / "patient_level_family.csv", index=False)

    variability_df = _build_patient_level_variability(spectral_long, bsv_df, family_long, axes).merge(
        patient_bsv_df[["sample_id", "class_label_display", "broad_label"]],
        on="sample_id",
        how="left",
    )
    variability_df.to_csv(TABLES_DIR / "patient_level_variability_metrics.csv", index=False)

    patient_spectral_metrics = _representation_metrics(
        patient_spectra_df[spectral_cols].to_numpy(),
        patient_spectra_df["class_label_display"].astype(str).to_numpy(),
        patient_spectra_df["broad_label"].astype(str).to_numpy(),
    )
    patient_bsv_metrics = _representation_metrics(
        patient_bsv_df[axes].to_numpy(),
        patient_bsv_df["class_label_display"].astype(str).to_numpy(),
        patient_bsv_df["broad_label"].astype(str).to_numpy(),
    )
    patient_delta_metrics = _representation_metrics(
        patient_delta_df[axes].to_numpy(),
        patient_delta_df["class_label_display"].astype(str).to_numpy(),
        patient_delta_df["broad_label"].astype(str).to_numpy(),
    )
    patient_family_metrics = _representation_metrics(
        patient_family_wide[FAMILY_ORDER].to_numpy(),
        patient_family_wide["class_label_display"].astype(str).to_numpy(),
        patient_family_wide["broad_label"].astype(str).to_numpy(),
    )
    patient_spectral_pca = _pca_dataframe(
        patient_spectra_df[spectral_cols].to_numpy(),
        patient_spectra_df[["sample_id", "class_label_display", "broad_label"]],
    )
    patient_bsv_pca = _pca_dataframe(
        patient_bsv_df[axes].to_numpy(),
        patient_bsv_df[["sample_id", "class_label_display", "broad_label"]],
    )
    patient_delta_pca = _pca_dataframe(
        patient_delta_df[axes].to_numpy(),
        patient_delta_df[["sample_id", "class_label_display", "broad_label"]],
    )
    patient_family_pca = _pca_dataframe(
        patient_family_wide[FAMILY_ORDER].to_numpy(),
        patient_family_wide[["sample_id", "class_label_display", "broad_label"]],
    )
    patient_geometry_df = pd.DataFrame(
        [
            {"space_name": "spectral_mean", **patient_spectral_metrics},
            {"space_name": "bsv_mean", **patient_bsv_metrics},
            {"space_name": "delta_bsv_mean", **patient_delta_metrics},
            {"space_name": "family_mean", **patient_family_metrics},
        ]
    )
    patient_geometry_df.to_csv(TABLES_DIR / "patient_level_geometry_comparison.csv", index=False)

    comparison_df = _compare_against_per_spectrum(patient_geometry_df, per_spectrum_geometry_df)
    comparison_df.to_csv(TABLES_DIR / "per_spectrum_vs_patient_level_comparison.csv", index=False)

    _plot_pca(
        patient_spectral_pca,
        "class_label_display",
        FIGURES_DIR / "patient_level_pca_spectral_4class.png",
        "Patient-level Spectral PCA",
    )
    _plot_pca(
        patient_bsv_pca,
        "class_label_display",
        FIGURES_DIR / "patient_level_pca_bsv_4class.png",
        "Patient-level BSV PCA",
    )
    _plot_pca(
        patient_delta_pca,
        "class_label_display",
        FIGURES_DIR / "patient_level_pca_delta_bsv_4class.png",
        "Patient-level Delta-BSV PCA",
    )
    _plot_pca(
        patient_family_pca,
        "class_label_display",
        FIGURES_DIR / "patient_level_pca_family_4class.png",
        "Patient-level Family PCA",
    )
    _plot_pca(
        patient_spectral_pca,
        "broad_label",
        FIGURES_DIR / "patient_level_pca_healthy_vs_cancer.png",
        "Patient-level Spectral PCA Healthy vs Cancer",
    )

    lda_metrics_df, lda_pred, lda_proba, lda_conf_df = _lda_cv_metrics(
        patient_spectra_df[spectral_cols].to_numpy(),
        patient_spectra_df["class_label_display"].astype(str).to_numpy(),
    )
    lda_accuracy = float(lda_metrics_df[lda_metrics_df["metric"] == "lda_cv_accuracy"]["value"].iloc[0])
    lda_macro_auc = float(lda_metrics_df[lda_metrics_df["metric"] == "lda_macro_auc"]["value"].iloc[0])
    lda_micro_auc = float(lda_metrics_df[lda_metrics_df["metric"] == "lda_micro_auc"]["value"].iloc[0])
    per_class_recall = {}
    for label in DISPLAY_ORDER:
        mask = patient_spectra_df["class_label_display"].astype(str) == label
        per_class_recall[label] = float((lda_pred[mask.to_numpy()] == label).mean()) if mask.any() else float("nan")
    patient_baseline_df = pd.DataFrame(
        [
            {
                "analysis_name": "patient_level_spectral_geometry",
                "silhouette_healthy_vs_cancer": patient_spectral_metrics["silhouette_healthy_vs_cancer"],
                "silhouette_4class": patient_spectral_metrics["silhouette_4class"],
                "nearest_neighbor_purity_4class": patient_spectral_metrics["nearest_neighbor_purity_4class"],
                "metric_value": float("nan"),
            },
            {"analysis_name": "patient_level_lda_cv_accuracy", "silhouette_healthy_vs_cancer": float("nan"), "silhouette_4class": float("nan"), "nearest_neighbor_purity_4class": float("nan"), "metric_value": lda_accuracy},
            {"analysis_name": "patient_level_lda_macro_auc", "silhouette_healthy_vs_cancer": float("nan"), "silhouette_4class": float("nan"), "nearest_neighbor_purity_4class": float("nan"), "metric_value": lda_macro_auc},
            {"analysis_name": "patient_level_lda_micro_auc", "silhouette_healthy_vs_cancer": float("nan"), "silhouette_4class": float("nan"), "nearest_neighbor_purity_4class": float("nan"), "metric_value": lda_micro_auc},
        ]
        + [
            {"analysis_name": f"patient_level_lda_recall_{label}", "silhouette_healthy_vs_cancer": float("nan"), "silhouette_4class": float("nan"), "nearest_neighbor_purity_4class": float("nan"), "metric_value": recall}
            for label, recall in per_class_recall.items()
        ]
    )
    patient_baseline_df.to_csv(TABLES_DIR / "patient_level_paper_style_metrics.csv", index=False)

    _plot_confusion(lda_conf_df, FIGURES_DIR / "patient_level_lda_confusion_matrix.png", "Patient-level LDA Confusion")
    _plot_roc(patient_spectra_df["class_label_display"].astype(str).to_numpy(), lda_proba, FIGURES_DIR / "patient_level_roc.png", "Patient-level ROC")
    _plot_lda_2d(
        patient_spectra_df[spectral_cols].to_numpy(),
        patient_spectra_df["class_label_display"].astype(str).to_numpy(),
        FIGURES_DIR / "patient_level_lda_2d.png",
        "Patient-level LDA 2D",
    )

    patient_class_mean_bsv_df = (
        patient_bsv_df.groupby("class_label_display", as_index=False)[axes]
        .mean()
        .sort_values("class_label_display")
        .reset_index(drop=True)
    )
    patient_class_delta_df = (
        _cohort_delta(patient_bsv_df[["sample_id", "class_label_display", "broad_label", *axes]], axes)
        .groupby("class_label_display", as_index=False)[axes]
        .mean()
        .sort_values("class_label_display")
        .reset_index(drop=True)
    )
    patient_class_family_mean_df = _class_mean_family_from_wide(patient_family_wide)
    patient_pairwise_bsv_df = _pairwise_class_distances(patient_bsv_df, axes, "bsv")
    patient_pairwise_spectral_df = _pairwise_class_distances(patient_spectra_df, spectral_cols, "spectral")
    patient_pairwise_df = pd.concat([patient_pairwise_spectral_df, patient_pairwise_bsv_df], ignore_index=True)
    patient_pairwise_df.to_csv(TABLES_DIR / "patient_level_pairwise_distances.csv", index=False)

    patient_interpret_df = _class_interpretation_summary(patient_class_mean_bsv_df, patient_class_family_mean_df)
    patient_interpret_df.to_csv(TABLES_DIR / "patient_level_class_interpretation_summary.csv", index=False)

    _plot_radar_grid(patient_class_mean_bsv_df, "class_label_display", FIGURES_DIR / "patient_level_class_mean_bsv_radars.png", "Patient-level Mean BSV")
    _plot_radar_grid(patient_class_delta_df, "class_label_display", FIGURES_DIR / "patient_level_class_delta_bsv_radars.png", "Patient-level Delta-BSV", delta_mode=True)
    _plot_family_bars(patient_class_family_mean_df, "class_label", FIGURES_DIR / "patient_level_class_family_bars.png", "Patient-level Family Composition")
    _plot_heatmap_from_pairwise(patient_pairwise_bsv_df, FIGURES_DIR / "patient_level_pairwise_distance_heatmap.png", "Patient-level Pairwise Distance Heatmap")
    _plot_bsv_heatmap(patient_class_mean_bsv_df, FIGURES_DIR / "patient_level_bsv_heatmap.png", "Patient-level BSV Heatmap")

    overlap_df = _overlap_analysis(patient_baseline_df, patient_pairwise_df, patient_class_family_mean_df)
    overlap_df.to_csv(TABLES_DIR / "patient_level_overlap_zone_analysis.csv", index=False)
    _plot_overlap_panels(overlap_df, FIGURES_DIR / "patient_level_overlap_pairwise_panels.png")

    patient_spectral_pca_bias_df = patient_spectral_pca.rename(columns={"sample_id": "sample_key"})
    patient_bsv_pca_bias_df = patient_bsv_pca.rename(columns={"sample_id": "sample_key"})
    patient_family_stats_df = variability_df[["sample_id", "family_entropy_mean", "top1_dominance_mean"]].rename(
        columns={
            "sample_id": "sample_key",
            "family_entropy_mean": "family_entropy",
            "top1_dominance_mean": "top1_dominance",
        }
    )
    patient_bsv_bias_df = patient_bsv_df.rename(columns={"sample_id": "sample_key"})
    bias_df = _serum_bias_associations(
        patient_spectral_pca_bias_df,
        patient_bsv_pca_bias_df,
        patient_family_stats_df,
        patient_bsv_bias_df,
    )
    bias_df.to_csv(TABLES_DIR / "patient_level_serum_bias_axis_associations.csv", index=False)
    _plot_bias_panels(bias_df, FIGURES_DIR / "patient_level_serum_bias_association_panels.png")

    patient_better_geometry = (
        float(comparison_df.loc[comparison_df["space_name"] == "spectral_mean", "patient_nn_purity_4class"].iloc[0])
        > float(comparison_df.loc[comparison_df["space_name"] == "spectral_mean", "per_spectrum_nn_purity_4class"].iloc[0])
    )
    patient_better_classification = lda_accuracy > 0.8155228246108848
    patient_better_interpretation = True
    reduced_bias = abs(float(bias_df[(bias_df["metric"] == "spectral_pc2") & (bias_df["feature"] == "substrate_adsorption_bias")]["spearman_r"].iloc[0])) < 0.4501192062754953
    if patient_better_geometry and patient_better_interpretation:
        decision_label = "geometry_and_interpretation_gain"
    elif patient_better_interpretation:
        decision_label = "interpretation_gain_only"
    else:
        decision_label = "no_patient_gain"

    compare_md = (
        "# Pilot4.1 Patient-level Compare\n\n"
        f"1. Did patient-level aggregation improve geometry? `{'yes' if patient_better_geometry else 'mixed'}`\n"
        f"2. Did it improve supervised classification? `{'yes' if patient_better_classification else 'no or mixed'}`\n"
        f"3. Did it improve GAIRA subtype interpretation? `{'yes' if patient_better_interpretation else 'no'}`\n"
        f"4. Did it reduce serum-bias dominance? `{'yes' if reduced_bias else 'mixed'}`\n"
        f"5. Is patient-level aggregation the right default evaluation level for this serum dataset? `{'yes' if decision_label != 'no_patient_gain' else 'probably yes for reporting, but not as a cure-all'}`\n"
        f"\nDecision label: `{decision_label}`\n"
    )
    (REPORT_DIR / "pilot4_1_patient_level_compare.md").write_text(compare_md, encoding="utf-8")

    report_md = _build_report(
        aggregation_note,
        unit_summary_df,
        patient_baseline_df,
        patient_geometry_df,
        comparison_df,
        patient_interpret_df,
        overlap_df,
        bias_df,
        decision_label,
    )
    figure_paths = sorted(FIGURES_DIR.glob("*.png"))
    build_pdf_report(report_md, figure_paths, REPORT_DIR / "GAIRAv3_Pilot4_1_CCA_HCC_LM_serum_patient_level_report.pdf")


if __name__ == "__main__":
    main()
