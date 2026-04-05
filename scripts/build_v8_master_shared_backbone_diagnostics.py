#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gaira.demo.v8_analysis_utils import load_eval_metrics, metric_value, save_barplot
from gaira.demo.v8_master_utils import MASTER_SHARED_DIR, SHARED_V1_DIR, dataset_status_rows, ensure_dir, safe_csv


V5_V7_CLUSTER_COMPARISON = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_cluster_analysis_v7/v5_vs_v7_cluster_comparison.csv")
V5_RUN_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_v5_full_true_gpu_run1")
V5_EVAL_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_eval_v2/embedding_v5_full_true_gpu_run1_eval_v2")
V6_RUN_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_v6_within_type_gpu_run1")
V6_EVAL_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_eval_v2/embedding_v6_within_type_gpu_run1_eval_v2")
V7_RUN_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_v7_anchor_gpu_run1")
V7_EVAL_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_eval_v2/embedding_v7_anchor_gpu_run1_eval_v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v8 master shared-backbone diagnostics.")
    parser.add_argument("--source-dir", type=Path, default=SHARED_V1_DIR)
    parser.add_argument("--output-dir", type=Path, default=MASTER_SHARED_DIR)
    return parser.parse_args()


def run_status(run_dir: Path, eval_dir: Path) -> str:
    return "available" if run_dir.exists() and eval_dir.exists() else "missing"


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    prior_metrics = safe_csv(args.source_dir / "shared_backbone_metrics.csv")
    cluster_compare = safe_csv(V5_V7_CLUSTER_COMPARISON)

    rows = []
    for backbone, run_dir, eval_dir in [
        ("v5_shared", V5_RUN_DIR, V5_EVAL_DIR),
        ("v6_shared", V6_RUN_DIR, V6_EVAL_DIR),
        ("v7_shared", V7_RUN_DIR, V7_EVAL_DIR),
    ]:
        metrics = load_eval_metrics(eval_dir)
        rows.append(
            {
                "backbone": backbone,
                "status": run_status(run_dir, eval_dir),
                "nn_purity_sample_type": metric_value(metrics, "nn_consistency_sample_type", evaluation_tier="full_corpus"),
                "top1_sample_type": metric_value(metrics, "top1_match_sample_type", evaluation_tier="full_corpus"),
                "nn_purity_dataset": metric_value(metrics, "nn_consistency_dataset_id", evaluation_tier="full_corpus"),
                "top1_dataset": metric_value(metrics, "top1_match_dataset_id", evaluation_tier="full_corpus"),
                "nn_purity_family": metric_value(metrics, "nn_consistency_family", evaluation_tier="full_corpus"),
                "top1_family": metric_value(metrics, "top1_match_family", evaluation_tier="full_corpus"),
                "silhouette_sample_type": metric_value(metrics, "silhouette_sample_type", evaluation_tier="sampled_global"),
                "silhouette_dataset": metric_value(metrics, "silhouette_dataset_id", evaluation_tier="sampled_global"),
                "silhouette_family": metric_value(metrics, "silhouette_family", evaluation_tier="sampled_global"),
                "ev_dataset_purity": metric_value(metrics, "nn_consistency_dataset_id", evaluation_tier="within_sample_type_full_corpus", sample_type_filter="ev"),
                "ev_family_purity": metric_value(metrics, "nn_consistency_family", evaluation_tier="within_sample_type_full_corpus", sample_type_filter="ev"),
                "serum_dataset_purity": metric_value(metrics, "nn_consistency_dataset_id", evaluation_tier="within_sample_type_full_corpus", sample_type_filter="serum"),
                "serum_family_purity": metric_value(metrics, "nn_consistency_family", evaluation_tier="within_sample_type_full_corpus", sample_type_filter="serum"),
                "grounding_dataset_purity": metric_value(metrics, "nn_consistency_dataset_id", evaluation_tier="within_sample_type_full_corpus", sample_type_filter="grounding"),
                "grounding_family_purity": metric_value(metrics, "nn_consistency_family", evaluation_tier="within_sample_type_full_corpus", sample_type_filter="grounding"),
            }
        )
    metrics_df = pd.DataFrame(rows)

    if not cluster_compare.empty:
        lookup = cluster_compare.set_index(["cluster_scope", "sample_type"])
        for backbone, prefix in [("v5_shared", "v5"), ("v7_shared", "v7")]:
            for sample_type in ["ev", "serum", "grounding"]:
                if ("within_type_cluster_id", sample_type) not in lookup.index:
                    continue
                metrics_df.loc[metrics_df["backbone"] == backbone, f"{sample_type}_cross_dataset_mixed_clusters"] = float(
                    lookup.loc[("within_type_cluster_id", sample_type), f"cross_dataset_mixed_cluster_count_{prefix}"]
                )
                metrics_df.loc[metrics_df["backbone"] == backbone, f"{sample_type}_mean_dataset_purity"] = float(
                    lookup.loc[("within_type_cluster_id", sample_type), f"mean_dataset_purity_{prefix}"]
                )
                metrics_df.loc[metrics_df["backbone"] == backbone, f"{sample_type}_cluster_count"] = float(
                    lookup.loc[("within_type_cluster_id", sample_type), f"cluster_count_{prefix}"]
                )

    metrics_df.to_csv(args.output_dir / "shared_backbone_metrics.csv", index=False)
    if not prior_metrics.empty:
        comparison = prior_metrics.merge(metrics_df, on=["backbone", "status"], how="outer", suffixes=("_legacy", "_master"))
    else:
        comparison = metrics_df.copy()
    comparison.to_csv(args.output_dir / "shared_backbone_comparison.csv", index=False)
    dataset_status_rows().to_csv(args.output_dir / "artifact_status.csv", index=False)

    available = metrics_df[metrics_df["status"] == "available"].copy()
    if not available.empty:
        long_rows = []
        for metric in [
            "nn_purity_sample_type",
            "nn_purity_dataset",
            "nn_purity_family",
            "silhouette_sample_type",
            "silhouette_dataset",
            "silhouette_family",
        ]:
            for _, row in available.iterrows():
                long_rows.append({"backbone": row["backbone"], "metric": metric, "value": row[metric]})
        save_barplot(pd.DataFrame(long_rows), x="metric", y="value", hue="backbone", title="Shared backbone comparison metrics", output_path=args.output_dir / "backbone_comparison_metrics.png")

        sample_type_df = available.melt(
            id_vars=["backbone"],
            value_vars=["nn_purity_sample_type", "top1_sample_type"],
            var_name="metric",
            value_name="value",
        )
        save_barplot(sample_type_df, x="metric", y="value", hue="backbone", title="Sample-type purity comparison", output_path=args.output_dir / "sample_type_purity_comparison.png")

        for sample_type, output_name in [
            ("ev", "ev_within_type_metric_comparison.png"),
            ("serum", "serum_within_type_metric_comparison.png"),
            ("grounding", "grounding_within_type_metric_comparison.png"),
        ]:
            cols = [f"{sample_type}_dataset_purity", f"{sample_type}_family_purity", f"{sample_type}_mean_dataset_purity"]
            subset = available.melt(id_vars=["backbone"], value_vars=[c for c in cols if c in available.columns], var_name="metric", value_name="value")
            if not subset.empty:
                save_barplot(subset, x="metric", y="value", hue="backbone", title=f"Within-{sample_type} metric comparison", output_path=args.output_dir / output_name)

        mixed_cols = [c for c in available.columns if c.endswith("_cross_dataset_mixed_clusters")]
        if mixed_cols:
            mixed_df = available.melt(id_vars=["backbone"], value_vars=mixed_cols, var_name="metric", value_name="value")
            save_barplot(mixed_df, x="metric", y="value", hue="backbone", title="Cross-dataset mixed cluster counts", output_path=args.output_dir / "cross_dataset_mixed_cluster_counts.png")

        cluster_cols = [c for c in available.columns if c.endswith("_cluster_count")]
        if cluster_cols:
            cluster_df = available.melt(id_vars=["backbone"], value_vars=cluster_cols, var_name="metric", value_name="value")
            save_barplot(cluster_df, x="metric", y="value", hue="backbone", title="Cluster size distribution comparison", output_path=args.output_dir / "cluster_size_distribution_comparison.png")

    summary_lines = [
        "# Shared Backbone Summary",
        "",
        "Available shared backbones inspected for the v8 master rollout:",
        f"- v5 shared: {run_status(V5_RUN_DIR, V5_EVAL_DIR)}",
        f"- v6 shared: {run_status(V6_RUN_DIR, V6_EVAL_DIR)}",
        f"- v7 shared: {run_status(V7_RUN_DIR, V7_EVAL_DIR)}",
        "",
        "Readout:",
        "- v7 remains the shared backbone to freeze for v8 because it preserves the sample-type scaffold while improving EV cross-dataset mixing relative to v5.",
        "- v5 remains a useful comparison baseline, especially for broad global geometry and some family-level metrics.",
        "- v6 full shared artifacts are still treated as missing locally rather than guessed.",
        "- The correct operational decision is: freeze v7 as the shared base, then branch EV stress and small2023 work off that decision instead of retraining another generic shared encoder first.",
    ]
    (args.output_dir / "shared_backbone_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
