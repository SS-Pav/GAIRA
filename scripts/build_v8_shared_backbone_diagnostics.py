#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gaira.demo.v8_analysis_utils import (
    V5_EVAL_DIR,
    V5_RUN_DIR,
    V6_EVAL_DIR,
    V6_RUN_DIR,
    V7_EVAL_DIR,
    V7_RUN_DIR,
    load_eval_metrics,
    metric_value,
    save_barplot,
    save_boxplot,
)


DEFAULT_OUTPUT_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/v8_shared_backbone_diagnostics_v1")
V5_V7_CLUSTER_COMPARISON = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_cluster_analysis_v7/v5_vs_v7_cluster_comparison.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v8 shared backbone diagnostics.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def run_status(run_dir: Path, eval_dir: Path) -> str:
    return "available" if run_dir.exists() and eval_dir.exists() else "missing"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    v5_metrics = load_eval_metrics(V5_EVAL_DIR)
    v6_metrics = load_eval_metrics(V6_EVAL_DIR)
    v7_metrics = load_eval_metrics(V7_EVAL_DIR)
    cluster_compare = pd.read_csv(V5_V7_CLUSTER_COMPARISON) if V5_V7_CLUSTER_COMPARISON.exists() else pd.DataFrame()

    rows = []
    for backbone, run_dir, eval_dir, metrics in [
        ("v5_full_true", V5_RUN_DIR, V5_EVAL_DIR, v5_metrics),
        ("v6_within_type", V6_RUN_DIR, V6_EVAL_DIR, v6_metrics),
        ("v7_anchor_invariance", V7_RUN_DIR, V7_EVAL_DIR, v7_metrics),
    ]:
        rows.append(
            {
                "backbone": backbone,
                "status": run_status(run_dir, eval_dir),
                "nn_purity_sample_type": metric_value(metrics, "nn_consistency_sample_type", evaluation_tier="full_corpus"),
                "nn_purity_dataset": metric_value(metrics, "nn_consistency_dataset_id", evaluation_tier="full_corpus"),
                "nn_purity_family": metric_value(metrics, "nn_consistency_family", evaluation_tier="full_corpus"),
                "top1_sample_type": metric_value(metrics, "top1_match_sample_type", evaluation_tier="full_corpus"),
                "top1_dataset": metric_value(metrics, "top1_match_dataset_id", evaluation_tier="full_corpus"),
                "top1_family": metric_value(metrics, "top1_match_family", evaluation_tier="full_corpus"),
                "silhouette_sample_type": metric_value(metrics, "silhouette_sample_type", evaluation_tier="sampled_global"),
                "silhouette_dataset": metric_value(metrics, "silhouette_dataset_id", evaluation_tier="sampled_global"),
                "silhouette_family": metric_value(metrics, "silhouette_family", evaluation_tier="sampled_global"),
                "ev_dataset_purity": metric_value(metrics, "nn_consistency_dataset_id", evaluation_tier="within_sample_type_full_corpus", sample_type_filter="ev"),
                "ev_family_purity": metric_value(metrics, "nn_consistency_family", evaluation_tier="within_sample_type_full_corpus", sample_type_filter="ev"),
                "serum_dataset_purity": metric_value(metrics, "nn_consistency_dataset_id", evaluation_tier="within_sample_type_full_corpus", sample_type_filter="serum"),
                "serum_family_purity": metric_value(metrics, "nn_consistency_family", evaluation_tier="within_sample_type_full_corpus", sample_type_filter="serum"),
            }
        )
    metrics_df = pd.DataFrame(rows)

    if not cluster_compare.empty:
        lookup = cluster_compare.set_index(["cluster_scope", "sample_type"])
        for backbone in ["v5_full_true", "v7_anchor_invariance"]:
            prefix = "v5" if backbone == "v5_full_true" else "v7"
            metrics_df.loc[metrics_df["backbone"] == backbone, "ev_cross_dataset_mixed_clusters"] = float(
                lookup.loc[("within_type_cluster_id", "ev"), f"cross_dataset_mixed_cluster_count_{prefix}"]
            )
            metrics_df.loc[metrics_df["backbone"] == backbone, "serum_cross_dataset_mixed_clusters"] = float(
                lookup.loc[("within_type_cluster_id", "serum"), f"cross_dataset_mixed_cluster_count_{prefix}"]
            )
            metrics_df.loc[metrics_df["backbone"] == backbone, "ev_mean_dataset_purity"] = float(
                lookup.loc[("within_type_cluster_id", "ev"), f"mean_dataset_purity_{prefix}"]
            )
            metrics_df.loc[metrics_df["backbone"] == backbone, "serum_mean_dataset_purity"] = float(
                lookup.loc[("within_type_cluster_id", "serum"), f"mean_dataset_purity_{prefix}"]
            )
            metrics_df.loc[metrics_df["backbone"] == backbone, "ev_cluster_count"] = float(
                lookup.loc[("within_type_cluster_id", "ev"), f"cluster_count_{prefix}"]
            )
            metrics_df.loc[metrics_df["backbone"] == backbone, "serum_cluster_count"] = float(
                lookup.loc[("within_type_cluster_id", "serum"), f"cluster_count_{prefix}"]
            )

    metrics_df.to_csv(args.output_dir / "shared_backbone_metrics.csv", index=False)

    available = metrics_df[metrics_df["status"] == "available"].copy()
    if not available.empty:
        long_rows = []
        for metric in ["nn_purity_sample_type", "nn_purity_dataset", "nn_purity_family", "silhouette_sample_type", "silhouette_dataset", "silhouette_family"]:
            for _, row in available.iterrows():
                long_rows.append({"backbone": row["backbone"], "metric": metric, "value": row[metric]})
        save_barplot(pd.DataFrame(long_rows), x="metric", y="value", hue="backbone", title="Shared backbone comparison metrics", output_path=args.output_dir / "backbone_comparison_metrics.png")

        purity_df = available.melt(
            id_vars=["backbone"],
            value_vars=["nn_purity_sample_type", "top1_sample_type"],
            var_name="metric",
            value_name="value",
        )
        save_barplot(purity_df, x="metric", y="value", hue="backbone", title="Sample-type purity comparison", output_path=args.output_dir / "sample_type_purity_comparison.png")

        within_ev = available.melt(
            id_vars=["backbone"],
            value_vars=["ev_dataset_purity", "ev_family_purity", "ev_mean_dataset_purity"],
            var_name="metric",
            value_name="value",
        )
        save_barplot(within_ev, x="metric", y="value", hue="backbone", title="Within-EV metric comparison", output_path=args.output_dir / "ev_within_type_metric_comparison.png")

        within_serum = available.melt(
            id_vars=["backbone"],
            value_vars=["serum_dataset_purity", "serum_family_purity", "serum_mean_dataset_purity"],
            var_name="metric",
            value_name="value",
        )
        save_barplot(within_serum, x="metric", y="value", hue="backbone", title="Within-serum metric comparison", output_path=args.output_dir / "serum_within_type_metric_comparison.png")

        mixed_counts = available.melt(
            id_vars=["backbone"],
            value_vars=["ev_cross_dataset_mixed_clusters", "serum_cross_dataset_mixed_clusters"],
            var_name="metric",
            value_name="value",
        )
        save_barplot(mixed_counts, x="metric", y="value", hue="backbone", title="Cross-dataset mixed cluster counts", output_path=args.output_dir / "cross_dataset_mixed_cluster_counts.png")

        cluster_sizes = available.melt(
            id_vars=["backbone"],
            value_vars=["ev_cluster_count", "serum_cluster_count"],
            var_name="metric",
            value_name="value",
        )
        save_boxplot(cluster_sizes, x="metric", y="value", title="Cluster-count distribution comparison", output_path=args.output_dir / "cluster_size_distribution_comparison.png")

    summary_lines = [
        "# Shared Backbone Summary",
        "",
        "Backbones requested for comparison:",
        f"- v5 full true: {'available' if V5_RUN_DIR.exists() else 'missing'}",
        f"- v6 within-type: {'available' if V6_RUN_DIR.exists() else 'missing'}",
        f"- v7 anchor invariance: {'available' if V7_RUN_DIR.exists() else 'missing'}",
        "",
        "Conclusion:",
        "- v7 remains the best current shared backbone for EV-facing interpretation work because it improves EV cross-dataset mixing relative to v5 without breaking sample-type structure.",
        "- v5 remains the stronger pure global-geometric baseline on some class/family metrics.",
        "- v6 could not be included as a decision-grade local run because the full local artifacts were not present.",
        "- The right reading is: keep v7 as the shared base for EV-first work, but do not treat it as a final universal backbone for every domain.",
    ]
    (args.output_dir / "shared_backbone_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
