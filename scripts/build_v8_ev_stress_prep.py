#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gaira.demo.v8_analysis_utils import STATE_COLORS, cluster_label_enrichment, load_v7_common, normalize_rows, save_barplot, save_heatmap, save_scatter
from gaira.demo.v8_master_utils import MASTER_EV_DIR, ensure_dir, safe_csv, safe_text
from gaira.demo.v8_theme_utils import (
    MASTER_THEME_COLORS,
    MASTER_THEME_ORDER,
    compute_split_theme_profiles,
    split_grounding_theme_table,
    split_existing_composition_frame,
    write_theme_split_note,
)


DEFAULT_SOURCE_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/v8_ev_stress_analysis_v1")
TARGET_DATASETS = {"shine_ev_sers", "diabetes_plasma_ev_sers"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v8 EV stress/state prep outputs.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=MASTER_EV_DIR)
    parser.add_argument("--top-k-grounding", type=int, default=12)
    return parser.parse_args()


def top_two_themes(row: pd.Series) -> tuple[str, str]:
    ordered = row[MASTER_THEME_ORDER].sort_values(ascending=False)
    top_theme = str(ordered.index[0]) if len(ordered) else "unresolved"
    second_theme = str(ordered.index[1]) if len(ordered) > 1 and ordered.iloc[1] > 0 else "none"
    return top_theme, second_theme


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    common = load_v7_common()
    metadata = common["metadata"].copy()  # type: ignore[assignment]
    embeddings = common["embeddings"]  # type: ignore[assignment]
    projection = common["projection"].copy()  # type: ignore[assignment]
    cluster_assignments = common["cluster_assignments"].copy()  # type: ignore[assignment]
    grounding_theme_table = common["grounding_theme_table"].copy()  # type: ignore[assignment]

    state_table = safe_csv(args.source_dir / "harmonized_state_table.csv")
    latent_metrics = safe_csv(args.source_dir / "ev_stress_metrics.csv")
    prior_state_enrichment = safe_csv(args.source_dir / "cluster_state_enrichment.csv")

    if state_table.empty:
        raise FileNotFoundError(f"Missing EV stress state table in {args.source_dir}")

    state_table["sample_key"] = state_table["sample_key"].astype(str)
    state_table = state_table[state_table["dataset_id"].isin(TARGET_DATASETS)].copy()
    state_table.rename(columns={"harmonized_state": "ev_state"}, inplace=True)
    state_table["ev_state"] = state_table["ev_state"].replace(
        {
            "low_metabolic_stress": "control_like",
            "high_metabolic_stress": "stress_or_toxicity_like",
        }
    )
    state_table.to_csv(args.output_dir / "ev_state_mapping_table.csv", index=False)
    source_report = safe_text(args.source_dir / "harmonized_state_report.md")
    note = "\n\nTheme split note:\n- The legacy `purine_metabolite_associated` bucket is split downstream into `purine_associated` and `general_metabolite_associated` for v8 prep outputs.\n"
    (args.output_dir / "ev_state_mapping_report.md").write_text(source_report + note, encoding="utf-8")

    ev_meta = metadata[(metadata["sample_type"] == "ev") & (metadata["dataset_id"].isin(TARGET_DATASETS))].copy()
    ev_meta["sample_key"] = ev_meta["sample_key"].astype(str)
    ev_meta = ev_meta.merge(state_table[["sample_key", "ev_state"]], on="sample_key", how="left")
    ev_assign = cluster_assignments[cluster_assignments["sample_key"].astype(str).isin(ev_meta["sample_key"])].copy()
    ev_assign["sample_key"] = ev_assign["sample_key"].astype(str)
    ev_meta = ev_meta.merge(ev_assign[["sample_key", "within_type_cluster_id"]], on="sample_key", how="left")

    split_grounding = split_grounding_theme_table(grounding_theme_table)
    split_grounding.to_csv(args.output_dir / "grounding_theme_table_split.csv", index=False)
    write_theme_split_note(args.output_dir / "theme_split_note.md", split_grounding)

    grounding_idx = metadata.index[metadata["sample_key"].astype(str).isin(split_grounding["sample_key"].astype(str))].to_numpy()
    ev_profiles = compute_split_theme_profiles(
        normalize_rows(embeddings[ev_meta.index.to_numpy()]),
        normalize_rows(embeddings[grounding_idx]),
        split_grounding,
        top_k=args.top_k_grounding,
    )
    ev_profiles.insert(0, "sample_key", ev_meta["sample_key"].to_numpy())
    ev_profiles.insert(1, "dataset_id", ev_meta["dataset_id"].to_numpy())
    ev_profiles.insert(2, "label_optional", ev_meta["label_optional"].to_numpy())
    ev_profiles.insert(3, "ev_state", ev_meta["ev_state"].to_numpy())
    ev_profiles.insert(4, "within_type_cluster_id", ev_meta["within_type_cluster_id"].to_numpy())
    ev_profiles.to_csv(args.output_dir / "ev_spectrum_composition_profiles.csv", index=False)

    cluster_profiles = ev_profiles.groupby("within_type_cluster_id", as_index=False)[MASTER_THEME_ORDER].mean()
    cluster_profiles["composition_coherence"] = (
        ev_profiles.groupby("within_type_cluster_id")[MASTER_THEME_ORDER]
        .apply(lambda frame: float(1.0 / (1.0 + (frame - frame.mean()).pow(2).sum(axis=1).pow(0.5).mean())))
        .to_numpy()
    )
    cluster_profiles["within_cluster_theme_variance"] = (
        ev_profiles.groupby("within_type_cluster_id")[MASTER_THEME_ORDER]
        .apply(lambda frame: float(frame.var(ddof=0).mean()))
        .to_numpy()
    )
    cluster_profiles["cluster_size"] = ev_profiles.groupby("within_type_cluster_id").size().to_numpy()
    cluster_profiles[["top_theme", "secondary_theme"]] = cluster_profiles.apply(
        lambda row: pd.Series(top_two_themes(row)),
        axis=1,
    )
    cluster_profiles.to_csv(args.output_dir / "ev_cluster_composition_profiles.csv", index=False)

    state_enrichment = cluster_label_enrichment(
        ev_meta[["sample_key", "within_type_cluster_id", "ev_state"]].dropna(subset=["within_type_cluster_id"]),
        cluster_col="within_type_cluster_id",
        label_col="ev_state",
    )
    state_enrichment.to_csv(args.output_dir / "ev_cluster_state_enrichment.csv", index=False)

    state_pivot = (
        state_enrichment.pivot(index="within_type_cluster_id", columns="ev_state", values="cluster_fraction")
        .fillna(0.0)
        .reset_index()
    )
    dataset_counts = ev_meta.groupby(["within_type_cluster_id", "dataset_id"]).size().reset_index(name="count")
    dataset_counts["dataset_fraction"] = dataset_counts.groupby("within_type_cluster_id")["count"].transform(lambda s: s / s.sum())
    dataset_pivot = dataset_counts.pivot(index="within_type_cluster_id", columns="dataset_id", values="dataset_fraction").fillna(0.0).reset_index()

    cluster_interp = cluster_profiles.merge(state_pivot, on="within_type_cluster_id", how="left").merge(dataset_pivot, on="within_type_cluster_id", how="left")
    if "cluster_id" in safe_csv(Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_grounding_analysis_v7/ev_cluster_interpretation_table.csv")).columns:
        grounding_interp = pd.read_csv("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_grounding_analysis_v7/ev_cluster_interpretation_table.csv")
        cluster_interp = cluster_interp.merge(
            grounding_interp[
                [
                    "cluster_id",
                    "cross_dataset_mixed",
                    "theme_support_strength",
                    "interpretation_summary",
                    "uncertainty_notes",
                    "caveat_notes",
                ]
            ],
            left_on="within_type_cluster_id",
            right_on="cluster_id",
            how="left",
        )
        if "cluster_id" in cluster_interp.columns:
            cluster_interp = cluster_interp.drop(columns=["cluster_id"])
    else:
        cluster_interp["cross_dataset_mixed"] = False
        cluster_interp["theme_support_strength"] = "moderate"
        cluster_interp["interpretation_summary"] = ""
        cluster_interp["uncertainty_notes"] = ""
        cluster_interp["caveat_notes"] = ""

    cluster_interp["dataset_composition_json"] = cluster_interp.apply(
        lambda row: json.dumps(
            {col: float(row[col]) for col in dataset_pivot.columns if col != "within_type_cluster_id" and pd.notna(row.get(col)) and float(row[col]) > 0},
            sort_keys=True,
        ),
        axis=1,
    )
    cluster_interp["state_composition_json"] = cluster_interp.apply(
        lambda row: json.dumps(
            {col: float(row[col]) for col in state_pivot.columns if col != "within_type_cluster_id" and pd.notna(row.get(col)) and float(row[col]) > 0},
            sort_keys=True,
        ),
        axis=1,
    )
    cluster_interp["support_strength"] = cluster_interp["theme_support_strength"].fillna("moderate")
    cluster_interp["concise_interpretation_summary"] = cluster_interp.apply(
        lambda row: (
            f"{row['top_theme']} primary, {row['secondary_theme']} secondary; "
            f"{'cross-dataset mixed' if bool(row.get('cross_dataset_mixed', False)) else 'dataset-skewed'} cluster with "
            f"{row['support_strength']} grounding support."
        ),
        axis=1,
    )
    cluster_interp.rename(columns={"within_type_cluster_id": "cluster_id"}, inplace=True)
    cluster_interp.to_csv(args.output_dir / "ev_cluster_interpretation_table.csv", index=False)

    ev_cluster_state_table = cluster_interp[
        [
            "cluster_id",
            "cluster_size",
            "dataset_composition_json",
            "state_composition_json",
            "top_theme",
            "secondary_theme",
            *MASTER_THEME_ORDER,
            "support_strength",
            "cross_dataset_mixed",
            "concise_interpretation_summary",
        ]
    ].copy()
    ev_cluster_state_table.to_csv(args.output_dir / "ev_cluster_state_table.csv", index=False)

    linkage_rows = []
    stress_scores = state_enrichment[state_enrichment["ev_state"] == "stress_or_toxicity_like"][["within_type_cluster_id", "log2_odds"]].rename(
        columns={"within_type_cluster_id": "cluster_id", "log2_odds": "stress_log2_odds"}
    )
    linked = cluster_profiles.rename(columns={"within_type_cluster_id": "cluster_id"}).merge(stress_scores, on="cluster_id", how="left")
    for theme in MASTER_THEME_ORDER:
        corr = linked[[theme, "stress_log2_odds"]].corr(method="pearson").iloc[0, 1]
        linkage_rows.append({"metric": f"{theme}_vs_stress_log2_odds_corr", "value": float(corr)})
    linkage_df = pd.DataFrame(linkage_rows)
    linkage_df.to_csv(args.output_dir / "ev_state_theme_linkage_metrics.csv", index=False)

    latent_metrics.to_csv(args.output_dir / "ev_latent_structure_metrics.csv", index=False)

    sampled = projection[(projection["sample_type"] == "ev") & (projection["dataset_id"].isin(TARGET_DATASETS))].copy()
    sampled["sample_key"] = sampled["sample_key"].astype(str)
    sampled = sampled.merge(state_table[["sample_key", "ev_state"]], on="sample_key", how="left")
    sampled = sampled.merge(ev_assign[["sample_key", "within_type_cluster_id"]], on="sample_key", how="left")
    sampled = sampled.merge(
        cluster_profiles[["within_type_cluster_id", "top_theme", "secondary_theme", "cluster_size"]],
        on="within_type_cluster_id",
        how="left",
    )
    sampled = sampled.merge(
        cluster_interp[["cluster_id", "cross_dataset_mixed", "support_strength"]],
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
            top_theme=("top_theme", "first"),
            cross_dataset_mixed=("cross_dataset_mixed", "max"),
            dominant_state=("ev_state", lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else "unmapped"),
            dominant_dataset=("dataset_id", lambda s: s.mode().iloc[0] if not s.empty else "unknown"),
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
        title="EV stress latent structure",
        output_path=args.output_dir / "neutral_latent_map.png",
        palette={"mixed": "#cc7b37", "single_dataset": "#6f7a85"},
    )
    save_scatter(
        sampled.fillna({"ev_state": "unmapped"}),
        x="dim1",
        y="dim2",
        hue="ev_state",
        style="dataset_id",
        size=None,
        title="EV latent map by harmonized state",
        output_path=args.output_dir / "latent_map_by_state.png",
        palette=STATE_COLORS,
    )
    save_scatter(
        centroid_map.fillna({"top_theme": "unresolved"}),
        x="dim1",
        y="dim2",
        hue="top_theme",
        style="mixing_category",
        size="cluster_size",
        title="EV latent map by dominant biochemical theme",
        output_path=args.output_dir / "latent_map_by_dominant_biochemical_theme.png",
        palette=MASTER_THEME_COLORS,
    )
    save_heatmap(
        cluster_profiles.set_index("within_type_cluster_id")[MASTER_THEME_ORDER],
        title="EV cluster composition heatmap",
        output_path=args.output_dir / "cluster_composition_heatmap.png",
    )
    state_heatmap = state_enrichment.pivot(index="within_type_cluster_id", columns="ev_state", values="cluster_fraction").fillna(0.0)
    save_heatmap(state_heatmap, title="EV cluster state heatmap", output_path=args.output_dir / "cluster_state_heatmap.png", cmap="rocket")

    linkage_plot = linkage_df.copy()
    linkage_plot["theme"] = linkage_plot["metric"].str.replace("_vs_stress_log2_odds_corr", "", regex=False)
    save_barplot(linkage_plot, x="theme", y="value", hue=None, title="EV theme vs stress linkage", output_path=args.output_dir / "composition_vs_state_scatter.png")

    cluster_rank = stress_scores.merge(cluster_profiles.rename(columns={"within_type_cluster_id": "cluster_id"}), on="cluster_id", how="left")
    save_barplot(cluster_rank.sort_values("stress_log2_odds", ascending=False).head(20), x="cluster_id", y="stress_log2_odds", hue="top_theme", title="Stress-enriched EV clusters", output_path=args.output_dir / "state_enrichment_ranked_clusters.png")
    save_scatter(
        centroid_map,
        x="dim1",
        y="dim2",
        hue="dominant_dataset",
        style="dominant_state",
        size="cluster_size",
        title="EV cluster map by dataset and state",
        output_path=args.output_dir / "dataset_vs_state_cluster_map.png",
        palette="deep",
    )

    summary_lines = [
        "# v8 EV Stress Summary",
        "",
        f"- spectra analyzed: {len(state_table):,}",
        f"- represented clusters: {state_table['within_type_cluster_id'].nunique():,}" if "within_type_cluster_id" in state_table.columns else f"- represented clusters: {ev_meta['within_type_cluster_id'].nunique():,}",
        f"- dataset nn purity: {float(latent_metrics.loc[latent_metrics.metric == 'nn_purity_dataset', 'value'].iloc[0]):.4f}",
        f"- harmonized-state nn purity: {float(latent_metrics.loc[latent_metrics.metric == 'nn_purity_harmonized_state', 'value'].iloc[0]):.4f}",
        "",
        "Readout:",
        "- EV stress/state remains the strongest current biology-learning target.",
        "- The correct interpretation is unsupervised latent neighborhoods first, then broad biochemical composition profiles layered on using grounding retrieval.",
        "- The new master outputs split the legacy purine/metabolite channel into `purine_associated` and `general_metabolite_associated` for clearer v8 reporting.",
    ]
    (args.output_dir / "ev_stress_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    spec_lines = [
        "# v8 EV Stress GPU Training Spec",
        "",
        "Backbone checkpoint:",
        "- Freeze the current shared starting point at `embedding_v7_anchor_gpu_run1` weights.",
        "",
        "Datasets:",
        "- include `shine_ev_sers`",
        "- include `diabetes_plasma_ev_sers`",
        "- exclude `small2023_ev` from this branch because it stays a specialized benchmark head problem",
        "",
        "Proposed objective:",
        "- initialize from the v7 shared encoder",
        "- keep within-sample-type EV training only",
        "- preserve instance positives",
        "- keep anchor positives for shared EV state anchors",
        "- add explicit state-supervised contrastive pressure on `control_like` vs `stress_or_toxicity_like` while retaining broad-theme composition reporting downstream",
        "- keep dataset-aware hard negatives so the model does not collapse obvious cohort or acquisition differences",
        "",
        "Suggested starting weights:",
        "- instance_positive_weight = 0.80",
        "- anchor_positive_weight = 0.10",
        "- state_positive_weight = 0.12",
        "- hard_negative_weight = 0.05",
        "- variance_regularization_weight = 0.02",
        "",
        "Suggested run length:",
        "- 30 to 40 epochs from the frozen v7 initialization",
        "",
        "Success metrics to improve relative to the frozen shared baseline:",
        "- higher `nn_purity_harmonized_state`",
        "- higher `top1_match_harmonized_state`",
        "- more cross-dataset mixed EV clusters",
        "- stronger but still cautious oxidative/redox and state linkage without collapsing all EV geometry into one axis",
    ]
    (args.output_dir / "ev_v8_training_spec.md").write_text("\n".join(spec_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
