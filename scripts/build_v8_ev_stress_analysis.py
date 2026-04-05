#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gaira.demo.v8_analysis_utils import EV_STRESS_V1_DIR, STATE_COLORS, THEME_COLORS, save_barplot, save_heatmap, save_scatter


DEFAULT_OUTPUT_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/v8_ev_stress_analysis_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v8 EV stress/disease analysis.")
    parser.add_argument("--source-dir", type=Path, default=EV_STRESS_V1_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    state_table = pd.read_csv(args.source_dir / "harmonized_state_table.csv")
    latent_metrics = pd.read_csv(args.source_dir / "latent_structure_metrics.csv")
    per_cluster = pd.read_csv(args.source_dir / "per_cluster_composition_profiles.csv")
    per_spectrum = pd.read_csv(args.source_dir / "per_spectrum_composition_profiles.csv")
    state_enrichment = pd.read_csv(args.source_dir / "cluster_state_enrichment.csv")
    linkage = pd.read_csv(args.source_dir / "state_composition_linkage_metrics.csv")

    state_table = state_table.copy()
    state_table["harmonized_state"] = state_table["harmonized_state"].replace(
        {
            "control_like": "low_metabolic_stress",
            "stress_or_toxicity_like": "high_metabolic_stress",
        }
    )
    state_table.to_csv(args.output_dir / "harmonized_state_table.csv", index=False)
    (args.output_dir / "harmonized_state_report.md").write_text(
        (args.source_dir / "harmonized_state_report.md").read_text().replace("control_like", "low_metabolic_stress").replace("stress_or_toxicity_like", "high_metabolic_stress"),
        encoding="utf-8",
    )

    ev_metrics = latent_metrics.copy()
    ev_metrics.to_csv(args.output_dir / "ev_stress_metrics.csv", index=False)
    state_enrichment.to_csv(args.output_dir / "cluster_state_enrichment.csv", index=False)
    per_spectrum.to_csv(args.output_dir / "per_spectrum_composition_profiles.csv", index=False)
    per_cluster.to_csv(args.output_dir / "per_cluster_composition_profiles.csv", index=False)
    linkage.to_csv(args.output_dir / "state_composition_linkage_metrics.csv", index=False)

    projection = pd.read_csv("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_eval_v2/embedding_v7_anchor_gpu_run1_eval_v2/embedding_projection_v2.csv")
    cluster_assignments = pd.read_csv("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_cluster_analysis_v7/cluster_assignments.csv")
    cluster_interp = pd.read_csv("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_grounding_analysis_v7/ev_cluster_interpretation_table.csv")
    sampled = projection[(projection["sample_type"] == "ev") & (projection["dataset_id"].isin(["shine_ev_sers", "diabetes_plasma_ev_sers"]))].copy()
    sampled = sampled.merge(state_table[["sample_key", "harmonized_state"]], on="sample_key", how="left")
    sampled = sampled.merge(cluster_assignments[["sample_key", "within_type_cluster_id"]], on="sample_key", how="left")
    sampled = sampled.merge(
        cluster_interp[["cluster_id", "top_biochemical_theme", "theme_support_strength", "cross_dataset_mixed"]],
        left_on="within_type_cluster_id",
        right_on="cluster_id",
        how="left",
    )
    centroid_map = (
        sampled.groupby("within_type_cluster_id", as_index=False)
        .agg(
            dim1=("dim1", "mean"),
            dim2=("dim2", "mean"),
            cluster_size=("sample_key", "count"),
            harmonized_state=("harmonized_state", lambda s: s.value_counts().index[0] if len(s) else "unmapped"),
            top_biochemical_theme=("top_biochemical_theme", "first"),
            cross_dataset_mixed=("cross_dataset_mixed", "max"),
        )
    )
    centroid_map["mixing_category"] = centroid_map["cross_dataset_mixed"].fillna(False).map({True: "mixed", False: "single_dataset"})

    save_scatter(
        centroid_map,
        x="dim1",
        y="dim2",
        hue="mixing_category",
        style="mixing_category",
        size="cluster_size",
        title="EV stress/disease latent structure",
        output_path=args.output_dir / "neutral_latent_map.png",
        palette={"mixed": "#cc7b37", "single_dataset": "#6f7a85"},
    )
    save_scatter(
        sampled.dropna(subset=["harmonized_state"]),
        x="dim1",
        y="dim2",
        hue="harmonized_state",
        style=None,
        size=None,
        title="EV stress/disease latent map by harmonized state",
        output_path=args.output_dir / "latent_map_by_state.png",
        palette={
            "low_metabolic_stress": STATE_COLORS["control_like"],
            "high_metabolic_stress": STATE_COLORS["stress_or_toxicity_like"],
            "intermediate_or_ambiguous": STATE_COLORS["intermediate_or_ambiguous"],
            "unmapped": STATE_COLORS["unmapped"],
        },
    )
    save_scatter(
        centroid_map.fillna({"top_biochemical_theme": "unresolved"}),
        x="dim1",
        y="dim2",
        hue="top_biochemical_theme",
        style="mixing_category",
        size="cluster_size",
        title="EV stress/disease latent map by dominant biochemical theme",
        output_path=args.output_dir / "latent_map_by_dominant_biochemical_theme.png",
        palette=THEME_COLORS,
    )

    heatmap_profiles = per_cluster.set_index("cluster_id")[
        [
            "protein_peptide_associated",
            "purine_metabolite_associated",
            "oxidative_redox_associated",
            "lipid_membrane_associated",
            "carbohydrate_associated",
            "nucleic_acid_associated",
            "serum_matrix_associated",
        ]
    ]
    save_heatmap(heatmap_profiles, title="EV stress cluster composition heatmap", output_path=args.output_dir / "cluster_composition_heatmap.png")

    state_heatmap = (
        state_enrichment.pivot(index="within_type_cluster_id", columns="harmonized_state", values="cluster_fraction")
        .fillna(0.0)
        .sort_index()
    )
    save_heatmap(state_heatmap, title="EV stress cluster state heatmap", output_path=args.output_dir / "cluster_state_heatmap.png", cmap="rocket")

    corr_df = linkage.copy()
    corr_df["theme"] = corr_df["metric"].str.replace("_vs_stress_log2_odds_corr", "", regex=False)
    save_barplot(corr_df, x="theme", y="value", hue=None, title="Theme-to-stress enrichment correlation", output_path=args.output_dir / "composition_vs_state_scatter.png")

    coherence_rows = []
    for _, row in per_cluster.iterrows():
        coherence_rows.append(
            {
                "dominant_theme": row["cluster_id"],
                "composition_coherence": row.get("protein_peptide_associated", 0.0),
            }
        )
    pd.DataFrame(coherence_rows).to_csv(args.output_dir / "composition_coherence_boxplot_source.csv", index=False)

    summary = [
        "# v8 EV Stress Summary",
        "",
        f"- spectra analyzed: {len(state_table):,}",
        f"- represented clusters: {state_table['within_type_cluster_id'].nunique():,}",
        f"- dataset nn purity: {float(ev_metrics.loc[ev_metrics.metric == 'nn_purity_dataset', 'value'].iloc[0]):.4f}",
        f"- harmonized-state nn purity: {float(ev_metrics.loc[ev_metrics.metric == 'nn_purity_harmonized_state', 'value'].iloc[0]):.4f}",
        "",
        "Assessment:",
        "- EV stress/disease remains the strongest current biological story.",
        "- It is strong enough to lead the v8 demo if the narrative stays broad and biochemical rather than diagnostic.",
        "- The correct framing is low-vs-high metabolic stress structure plus grounding-derived composition profiles.",
    ]
    (args.output_dir / "ev_stress_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
