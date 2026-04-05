#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gaira.demo.v8_analysis_utils import (
    STATE_COLORS,
    THEME_COLORS,
    THEME_ORDER,
    cluster_composition_summary,
    cluster_label_enrichment,
    compute_theme_profiles,
    knn_label_metrics,
    load_v7_common,
    normalize_rows,
    sampled_silhouette,
    save_heatmap,
    save_scatter,
)


DEFAULT_OUTPUT_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/v8_serum_stress_analysis_v1")
SERUM_DATASETS = {
    "cca_hcc_lm_serum_sers",
    "covid_serum_raman",
    "cspp_serum",
    "ergothioneine_serum",
    "serum_ag_colloids",
    "serum_protocol_comparison",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v8 serum stress/inflammation analysis.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--knn-k", type=int, default=6)
    parser.add_argument("--top-k-grounding", type=int, default=12)
    return parser.parse_args()


def harmonize_serum_state(row: pd.Series) -> tuple[str, str, str]:
    dataset_id = str(row.get("dataset_id", ""))
    label = str(row.get("label_optional", ""))
    subclass = str(row.get("subclass_label", ""))
    record_kind = str(row.get("record_kind", ""))
    if record_kind == "class_summary":
        return ("intermediate_or_ambiguous", "low", "summary artifact retained for completeness")

    if dataset_id == "cca_hcc_lm_serum_sers":
        if label == "healthy_control":
            return ("low_stress_inflammation", "high", "explicit healthy control in liver serum cohort")
        if label in {"cca", "hcc", "lm"}:
            return ("high_stress_inflammation", "high", "explicit hepatobiliary disease cohort label")

    if dataset_id == "covid_serum_raman":
        if label in {"healthy_control", "tube_control"}:
            return ("low_stress_inflammation", "high", "explicit healthy or tube-control serum state")
        if label == "covid_confirmed":
            return ("high_stress_inflammation", "high", "explicit confirmed inflammatory infection cohort")
        if label == "suspected_case":
            return ("intermediate_or_ambiguous", "medium", "suspected inflammatory state retained as ambiguous")

    if dataset_id == "serum_ag_colloids":
        if label in {"Serum", "SerumMerck", "SerumSigma"}:
            return ("low_stress_inflammation", "medium", "commercial or donor serum reference")
        return ("intermediate_or_ambiguous", "low", "spiked serum or component reference, not a direct biological state")

    if dataset_id in {"cspp_serum", "serum_protocol_comparison", "ergothioneine_serum"}:
        if label in {"Bkg", "standard", "unprocessed"}:
            return ("low_stress_inflammation", "low", "process-control spectrum, kept as low-stress reference only")
        return ("intermediate_or_ambiguous", "low", "protocol, spiking, calibration, or process-variation archive")

    if subclass == "released_zip_archive" and label == "healthy_control":
        return ("low_stress_inflammation", "medium", "healthy serum label")
    return ("intermediate_or_ambiguous", "low", "no defensible broad serum stress mapping")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    common = load_v7_common()
    metadata = common["metadata"].copy()  # type: ignore[assignment]
    embeddings = common["embeddings"]  # type: ignore[assignment]
    projection = common["projection"].copy()  # type: ignore[assignment]
    cluster_assignments = common["cluster_assignments"].copy()  # type: ignore[assignment]
    cluster_summary = common["cluster_summary"].copy()  # type: ignore[assignment]
    grounding_theme_table = common["grounding_theme_table"].copy()  # type: ignore[assignment]

    serum = metadata[(metadata["sample_type"] == "serum") & (metadata["dataset_id"].isin(SERUM_DATASETS))].copy()
    harmonized = serum.apply(harmonize_serum_state, axis=1, result_type="expand")
    harmonized.columns = ["harmonized_state", "state_confidence", "state_notes"]
    serum = pd.concat([serum.reset_index(drop=True), harmonized], axis=1)
    serum["sample_key"] = serum["sample_key"].astype(str)

    serum_state_table = serum[
        [
            "sample_key",
            "dataset_id",
            "label_optional",
            "family_label",
            "subclass_label",
            "record_kind",
            "harmonized_state",
            "state_confidence",
            "state_notes",
        ]
    ].copy()
    serum_state_table.to_csv(args.output_dir / "serum_harmonized_state_table.csv", index=False)

    serum_idx = serum.index.to_numpy()
    Z_serum = normalize_rows(embeddings[serum_idx])
    serum_clear = serum[serum["harmonized_state"].isin(["low_stress_inflammation", "high_stress_inflammation"])].copy()
    Z_serum_clear = normalize_rows(embeddings[serum_clear.index.to_numpy()])

    dataset_knn = knn_label_metrics(Z_serum, serum["dataset_id"].to_numpy(), k=args.knn_k)
    state_knn = knn_label_metrics(Z_serum_clear, serum_clear["harmonized_state"].to_numpy(), k=args.knn_k) if len(serum_clear) else {
        "nn_purity": float("nan"),
        "neighbor_entropy": float("nan"),
        "top1_match": float("nan"),
    }
    state_silhouette = sampled_silhouette(Z_serum_clear, serum_clear["harmonized_state"].to_numpy(), seed=args.seed) if len(serum_clear) else float("nan")
    dataset_silhouette = sampled_silhouette(Z_serum, serum["dataset_id"].to_numpy(), seed=args.seed)

    serum_assign = cluster_assignments[
        cluster_assignments["sample_key"].astype(str).isin(serum["sample_key"])
    ][["sample_key", "within_type_cluster_id"]].copy()
    serum_assign["sample_key"] = serum_assign["sample_key"].astype(str)
    serum = serum.merge(serum_assign, on="sample_key", how="left")

    state_enrichment = cluster_label_enrichment(
        serum[["sample_key", "within_type_cluster_id", "harmonized_state"]].dropna(subset=["within_type_cluster_id"]),
        cluster_col="within_type_cluster_id",
        label_col="harmonized_state",
    )
    state_enrichment.to_csv(args.output_dir / "serum_cluster_state_enrichment.csv", index=False)

    cluster_dataset_entropy = []
    for cluster_id, group in serum.groupby("within_type_cluster_id", dropna=True):
        dataset_probs = group["dataset_id"].value_counts(normalize=True)
        state_probs = group["harmonized_state"].value_counts(normalize=True)
        cluster_dataset_entropy.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": int(len(group)),
                "dataset_count": int(group["dataset_id"].nunique()),
                "state_count": int(group["harmonized_state"].nunique()),
                "dataset_entropy": float(-(dataset_probs * np.log2(dataset_probs + 1e-12)).sum()),
                "state_entropy": float(-(state_probs * np.log2(state_probs + 1e-12)).sum()),
            }
        )
    cluster_entropy_df = pd.DataFrame(cluster_dataset_entropy)
    serum_cluster_summary = cluster_summary[
        (cluster_summary["cluster_scope"] == "within_type_cluster_id")
        & (cluster_summary["dominant_sample_type"] == "serum")
    ][["cluster_id", "cross_dataset_mixed", "dataset_pure", "interpretation_label"]].copy()
    cluster_entropy_df = cluster_entropy_df.merge(
        serum_cluster_summary,
        left_on="cluster_id",
        right_on="cluster_id",
        how="left",
    )
    cluster_entropy_df["cross_dataset_mixed"] = cluster_entropy_df["cross_dataset_mixed"].fillna(False).astype(bool)

    grounding_idx = metadata.index[metadata["sample_key"].astype(str).isin(grounding_theme_table["sample_key"].astype(str))].to_numpy()
    serum_profiles = compute_theme_profiles(
        Z_serum,
        normalize_rows(embeddings[grounding_idx]),
        grounding_theme_table["grounding_theme"].astype(str).to_numpy(),
        top_k=args.top_k_grounding,
    )
    serum_profiles.insert(0, "sample_key", serum["sample_key"].to_numpy())
    serum_profiles.insert(1, "within_type_cluster_id", serum["within_type_cluster_id"].to_numpy())
    serum_profiles.insert(2, "dataset_id", serum["dataset_id"].to_numpy())
    serum_profiles.insert(3, "harmonized_state", serum["harmonized_state"].to_numpy())

    cluster_profile_values, cluster_profile_metrics = cluster_composition_summary(
        serum_profiles,
        cluster_col="within_type_cluster_id",
    )
    cluster_profiles = cluster_profile_values.rename(columns={"within_type_cluster_id": "cluster_id"}).merge(
        cluster_profile_metrics.rename(columns={"within_type_cluster_id": "cluster_id"}),
        on="cluster_id",
        how="left",
    )
    cluster_profiles.to_csv(args.output_dir / "serum_per_cluster_composition_profiles.csv", index=False)

    linkage_rows = []
    stress_enrichment = (
        state_enrichment[state_enrichment["harmonized_state"] == "high_stress_inflammation"][
            ["within_type_cluster_id", "log2_odds"]
        ]
        .rename(columns={"within_type_cluster_id": "cluster_id", "log2_odds": "high_stress_log2_odds"})
    )
    linked = cluster_profiles.merge(stress_enrichment, on="cluster_id", how="left")
    for theme in THEME_ORDER:
        corr = linked[[theme, "high_stress_log2_odds"]].corr(method="pearson").iloc[0, 1]
        linkage_rows.append({"metric": f"{theme}_vs_high_stress_log2_odds_corr", "value": float(corr)})
    linkage_df = pd.DataFrame(linkage_rows)
    linkage_df.to_csv(args.output_dir / "serum_state_composition_linkage_metrics.csv", index=False)

    serum_metrics = pd.DataFrame(
        [
            {"metric": "n_spectra", "value": float(len(serum))},
            {"metric": "n_clear_state_spectra", "value": float(len(serum_clear))},
            {"metric": "cluster_count", "value": float(serum["within_type_cluster_id"].nunique())},
            {"metric": "nn_purity_dataset", "value": dataset_knn["nn_purity"]},
            {"metric": "nn_purity_harmonized_state", "value": state_knn["nn_purity"]},
            {"metric": "top1_match_dataset", "value": dataset_knn["top1_match"]},
            {"metric": "top1_match_harmonized_state", "value": state_knn["top1_match"]},
            {"metric": "neighbor_entropy_dataset", "value": dataset_knn["neighbor_entropy"]},
            {"metric": "neighbor_entropy_harmonized_state", "value": state_knn["neighbor_entropy"]},
            {"metric": "silhouette_dataset", "value": dataset_silhouette},
            {"metric": "silhouette_harmonized_state", "value": state_silhouette},
            {"metric": "cross_dataset_mixed_cluster_count", "value": float(cluster_entropy_df["cross_dataset_mixed"].sum())},
            {"metric": "mean_cluster_dataset_entropy", "value": float(cluster_entropy_df["dataset_entropy"].mean())},
            {"metric": "mean_cluster_state_entropy", "value": float(cluster_entropy_df["state_entropy"].mean())},
        ]
    )
    serum_metrics.to_csv(args.output_dir / "serum_metrics.csv", index=False)

    serum_projection = projection[(projection["sample_type"] == "serum") & (projection["dataset_id"].isin(SERUM_DATASETS))].copy()
    serum_projection["sample_key"] = serum_projection["sample_key"].astype(str)
    serum_projection = serum_projection.merge(
        serum_state_table[["sample_key", "harmonized_state"]],
        on="sample_key",
        how="left",
    )
    serum_projection = serum_projection.merge(serum_assign, on="sample_key", how="left")
    serum_projection = serum_projection.merge(
        cluster_profiles[["cluster_id", "dominant_theme", "secondary_theme", "composition_coherence", "cluster_size"]],
        left_on="within_type_cluster_id",
        right_on="cluster_id",
        how="left",
    )
    centroid_map = (
        serum_projection.groupby("within_type_cluster_id", as_index=False)
        .agg(
            dim1=("dim1", "mean"),
            dim2=("dim2", "mean"),
            cluster_size=("sample_key", "count"),
            harmonized_state=("harmonized_state", lambda s: s.value_counts().index[0] if len(s.dropna()) else "unmapped"),
            dominant_theme=("dominant_theme", "first"),
        )
    )
    centroid_map = centroid_map.merge(
        cluster_entropy_df[["cluster_id", "cross_dataset_mixed", "dataset_entropy"]],
        left_on="within_type_cluster_id",
        right_on="cluster_id",
        how="left",
    )
    centroid_map["mixing_category"] = centroid_map["cross_dataset_mixed"].fillna(False).map({True: "mixed", False: "single_dataset"})

    save_scatter(
        centroid_map,
        x="dim1",
        y="dim2",
        hue="mixing_category",
        style="mixing_category",
        size="cluster_size",
        title="Serum latent structure",
        output_path=args.output_dir / "serum_neutral_latent_map.png",
        palette={"mixed": "#bb6a34", "single_dataset": "#6e7882"},
    )
    save_scatter(
        serum_projection.fillna({"harmonized_state": "unmapped"}),
        x="dim1",
        y="dim2",
        hue="harmonized_state",
        style=None,
        size=None,
        title="Serum latent map by broad stress/inflammation state",
        output_path=args.output_dir / "serum_latent_map_by_state.png",
        palette={
            "low_stress_inflammation": STATE_COLORS["control_like"],
            "high_stress_inflammation": STATE_COLORS["stress_or_toxicity_like"],
            "intermediate_or_ambiguous": STATE_COLORS["intermediate_or_ambiguous"],
            "unmapped": STATE_COLORS["unmapped"],
        },
    )
    save_scatter(
        centroid_map.fillna({"dominant_theme": "unresolved"}),
        x="dim1",
        y="dim2",
        hue="dominant_theme",
        style="mixing_category",
        size="cluster_size",
        title="Serum latent map by dominant biochemical theme",
        output_path=args.output_dir / "serum_latent_map_by_theme.png",
        palette=THEME_COLORS,
    )
    save_heatmap(
        cluster_profiles.set_index("cluster_id")[THEME_ORDER],
        title="Serum cluster composition heatmap",
        output_path=args.output_dir / "serum_cluster_composition_heatmap.png",
    )
    serum_state_heatmap = (
        state_enrichment.pivot(index="within_type_cluster_id", columns="harmonized_state", values="cluster_fraction")
        .fillna(0.0)
        .sort_index()
    )
    save_heatmap(
        serum_state_heatmap,
        title="Serum cluster state heatmap",
        output_path=args.output_dir / "serum_cluster_state_heatmap.png",
        cmap="rocket",
    )
    scatter_df = linked[["protein_peptide_associated", "oxidative_redox_associated", "high_stress_log2_odds", "cluster_size"]].copy()
    scatter_df = scatter_df.fillna(0.0)
    save_scatter(
        scatter_df.rename(columns={"protein_peptide_associated": "protein_theme", "high_stress_log2_odds": "state_enrichment"}),
        x="protein_theme",
        y="state_enrichment",
        hue="oxidative_redox_associated",
        style=None,
        size="cluster_size",
        title="Serum composition versus high-stress enrichment",
        output_path=args.output_dir / "serum_composition_vs_state_scatter.png",
        palette="mako",
    )

    summary_lines = [
        "# v8 Serum Stress Summary",
        "",
        "This serum analysis uses broad and deliberately conservative state harmonization. Only explicit healthy-control, hepatobiliary-disease, and confirmed inflammatory cohorts are treated as strong biological states; protocol, spiking, and calibration archives are retained mainly as ambiguity or control context.",
        "",
        f"- spectra analyzed: {len(serum):,}",
        f"- clear low/high state spectra: {len(serum_clear):,}",
        f"- dataset nn purity: {dataset_knn['nn_purity']:.4f}",
        f"- broad-state nn purity: {state_knn['nn_purity']:.4f}" if len(serum_clear) else "- broad-state nn purity: not computable",
        f"- within-type serum clusters represented: {serum['within_type_cluster_id'].nunique():,}",
        f"- cross-dataset mixed serum clusters: {int(cluster_entropy_df['cross_dataset_mixed'].sum())}",
        "",
        "Assessment:",
        "- Serum is not yet ready for a confident dedicated invariance head. The current shared latent space remains strongly cohort- and protocol-structured, and the most defensible biological grouping lives almost entirely inside the liver and COVID cohorts.",
        "- The useful signal is still real: healthy liver controls, hepatobiliary disease spectra, and inflammatory COVID serum are separable enough to justify a cautious stress/inflammation track design study.",
        "- The next serum step should be better cohort harmonization and cleaner anchor construction, not immediate full-scale specialized retraining.",
    ]
    (args.output_dir / "serum_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
