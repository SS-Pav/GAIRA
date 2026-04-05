#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA

from gaira.demo.v8_analysis_utils import save_barplot, save_heatmap, save_scatter
from gaira.demo.v8_master_utils import MASTER_SMALL_DIR, SMALL2023_BENCHMARK_V1_DIR, ensure_dir, safe_csv, safe_text
from gaira.demo.v8_theme_utils import MASTER_THEME_ORDER, split_existing_composition_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v8 small2023 specialized-head prep outputs.")
    parser.add_argument("--source-dir", type=Path, default=SMALL2023_BENCHMARK_V1_DIR)
    parser.add_argument("--output-dir", type=Path, default=MASTER_SMALL_DIR)
    return parser.parse_args()


def write_model_metrics(metrics: pd.DataFrame, model_name: str, output_path: Path) -> None:
    subset = metrics[metrics["model"] == model_name].copy()
    if subset.empty:
        subset = pd.DataFrame([{"metric": "status", "value": float("nan"), "model": model_name, "status": "missing"}])
    subset.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    metrics = safe_csv(args.source_dir / "small2023_benchmark_metrics.csv")
    comparison = safe_csv(args.source_dir / "small2023_direct_vs_embedding_comparison.csv")
    class_summary = safe_csv(args.source_dir / "small2023_class_composition_summary.csv")

    if metrics.empty:
        raise FileNotFoundError(f"Missing benchmark metrics in {args.source_dir}")

    write_model_metrics(metrics, "direct", args.output_dir / "small2023_direct_metrics.csv")
    write_model_metrics(metrics, "v5", args.output_dir / "small2023_v5_metrics.csv")
    write_model_metrics(metrics, "v7", args.output_dir / "small2023_v7_metrics.csv")
    write_model_metrics(metrics, "v2_specialized", args.output_dir / "small2023_v2_metrics.csv")
    comparison.to_csv(args.output_dir / "small2023_comparison_table.csv", index=False)

    split_class_summary = split_existing_composition_frame(class_summary)
    split_class_summary.to_csv(args.output_dir / "small2023_class_composition_summary.csv", index=False)
    if not split_class_summary.empty and "class_label" in split_class_summary.columns:
        save_heatmap(
            split_class_summary.set_index("class_label")[MASTER_THEME_ORDER],
            title="small2023 class composition heatmap",
            output_path=args.output_dir / "class_composition_heatmap.png",
        )
        coords = PCA(n_components=2, random_state=7).fit_transform(split_class_summary[MASTER_THEME_ORDER].to_numpy(dtype=float))
        scatter_df = split_class_summary.copy()
        scatter_df["dim1"] = coords[:, 0]
        scatter_df["dim2"] = coords[:, 1]
        scatter_df["dominant_theme"] = scatter_df[MASTER_THEME_ORDER].idxmax(axis=1)
        save_scatter(
            scatter_df,
            x="dim1",
            y="dim2",
            hue="dominant_theme",
            style="class_label",
            size=None,
            title="small2023 class composition scatter",
            output_path=args.output_dir / "class_composition_scatter.png",
            palette="deep",
        )

    for name in [
        "direct_map_by_class.png",
        "direct_map_by_probe.png",
        "v5_embedding_map_by_class.png",
        "v5_embedding_map_by_probe.png",
        "v7_embedding_map_by_class.png",
        "v7_embedding_map_by_probe.png",
        "v2_embedding_map_by_class.png",
        "v2_embedding_map_by_probe.png",
        "direct_vs_embedding_metric_bars.png",
    ]:
        source = args.source_dir / name
        dest = args.output_dir / name
        if source.exists():
            dest.write_bytes(source.read_bytes())

    model_scorecard = comparison.copy()
    if not model_scorecard.empty:
        metric_rows = []
        for _, row in model_scorecard.iterrows():
            metric = row["metric"]
            for model in ["direct", "v5", "v7", "v2_specialized"]:
                if model not in row or pd.isna(row[model]):
                    continue
                metric_rows.append({"model": model, "metric": metric, "value": float(row[model])})
        metric_df = pd.DataFrame(metric_rows)
        if not metric_df.empty:
            subset = metric_df[metric_df["metric"].isin(["silhouette_class", "silhouette_probe", "nn_purity_class", "nn_purity_probe"])].copy()
            subset["group"] = subset["metric"].map(
                {
                    "silhouette_class": "class_structure",
                    "silhouette_probe": "probe_structure",
                    "nn_purity_class": "class_neighborhood",
                    "nn_purity_probe": "probe_neighborhood",
                }
            )
            save_barplot(subset, x="model", y="value", hue="group", title="small2023 probe-transfer scorecard", output_path=args.output_dir / "probe_transfer_scorecard.png")

    summary_lines = [
        "# small2023 Specialized Prep Summary",
        "",
        "Readout:",
        "- The shared backbone improves class silhouette over direct processed spectra, but not enough to replace the old specialized probe-invariant benchmark.",
        "- The v2 specialized embedding still dominates on the explicit class-vs-probe benchmark, which means small2023 remains a specialized-head problem rather than a shared-backbone victory lap.",
        "- The correct v8 branch is to keep small2023 separate from the EV stress/state branch.",
    ]
    source_summary = safe_text(args.source_dir / "small2023_summary.md")
    (args.output_dir / "small2023_summary.md").write_text("\n".join(summary_lines) + "\n\n" + source_summary, encoding="utf-8")

    spec_lines = [
        "# small2023 v8 Specialized Training Spec",
        "",
        "Checkpoint strategy:",
        "- initialize from the frozen v7 shared backbone encoder",
        "- add a specialized small2023 head / fine-tuning branch rather than expecting the shared EV stress encoder to solve probe invariance by itself",
        "",
        "Objective design:",
        "- class-separation positive pressure on the cell-line classes",
        "- explicit probe nuisance reduction across `normedprobe1` and `normedprobe2`",
        "- retain class-preserving augmentations only",
        "- monitor for class collapse while probe structure drops",
        "",
        "Suggested loss terms:",
        "- class-supervised contrastive loss",
        "- probe-adversarial or probe-confusion auxiliary objective",
        "- centroid-margin regularization across class labels",
        "",
        "Success metrics:",
        "- class silhouette approaches or exceeds the old v2 benchmark",
        "- probe silhouette drops toward the old v2 benchmark instead of staying near the shared-backbone levels",
        "- class nn purity stays high while probe nn purity drops",
        "- cross-probe transfer score improves relative to v5 and v7 shared embeddings",
    ]
    (args.output_dir / "small2023_v8_training_spec.md").write_text("\n".join(spec_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
