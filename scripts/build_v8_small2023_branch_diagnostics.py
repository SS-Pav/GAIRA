#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gaira.demo.small2023_branch_utils import (
    V5_RUN_DIR,
    V7_RUN_DIR,
    load_branch_run,
    load_direct_mode,
    mode_labels,
    pca_plot_frame,
    representation_scorecard,
    try_load_shared_subset,
    try_load_v2_summary,
)
from gaira.demo.v8_analysis_utils import save_barplot, save_scatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build small2023 branch diagnostics.")
    parser.add_argument("--mode", choices=["cellline", "mixture"], required=True)
    parser.add_argument("--branch-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--knn-k", type=int, default=6)
    parser.add_argument("--max-comparison-samples", type=int, default=12000)
    return parser.parse_args()


def balanced_sample_metadata(
    metadata: pd.DataFrame,
    *,
    seed: int,
    max_samples: int,
) -> pd.DataFrame:
    if len(metadata) <= max_samples:
        return metadata.copy().reset_index(drop=True)
    groups = []
    grouped = metadata.groupby(["class_label", "probe_label"], dropna=False, sort=True)
    per_group = max(1, max_samples // max(grouped.ngroups, 1))
    for _, group in grouped:
        groups.append(group.sample(n=min(per_group, len(group)), random_state=seed))
    sampled = pd.concat(groups, ignore_index=True).drop_duplicates("sample_key")
    if len(sampled) < max_samples:
        remaining = metadata.loc[~metadata["sample_key"].isin(sampled["sample_key"])]
        if not remaining.empty:
            sampled = pd.concat(
                [
                    sampled,
                    remaining.sample(
                        n=min(max_samples - len(sampled), len(remaining)),
                        random_state=seed,
                    ),
                ],
                ignore_index=True,
            )
    return sampled.head(max_samples).reset_index(drop=True)


def plot_representation(
    metadata: pd.DataFrame,
    values: np.ndarray,
    *,
    seed: int,
    prefix: str,
    output_dir: Path,
    max_points: int = 8000,
) -> None:
    plot_meta = metadata.copy()
    plot_values = values
    if len(plot_meta) > max_points:
        grouped = plot_meta.groupby(["class_label", "probe_label"], dropna=False, sort=True)
        groups = max(grouped.ngroups, 1)
        per_group = max(200, max_points // groups)
        sampled_groups = []
        for _, group in grouped:
            sampled_groups.append(group.sample(n=min(per_group, len(group)), random_state=seed))
        plot_meta = pd.concat(sampled_groups, ignore_index=True)
        keep_keys = plot_meta["sample_key"].astype(str).tolist()
        index_map = pd.Series(np.arange(len(metadata)), index=metadata["sample_key"].astype(str))
        plot_values = values[index_map.loc[keep_keys].to_numpy()]
    plot_df = pca_plot_frame(plot_values, plot_meta, seed=seed)
    save_scatter(
        plot_df,
        x="dim1",
        y="dim2",
        hue="class_label",
        style=None,
        size=None,
        title=f"{prefix} by class",
        output_path=output_dir / f"{prefix}_map_by_class.png",
    )
    if plot_df["probe_label"].fillna("").astype(str).nunique() > 1:
        save_scatter(
            plot_df,
            x="dim1",
            y="dim2",
            hue="probe_label",
            style=None,
            size=None,
            title=f"{prefix} by probe",
            output_path=output_dir / f"{prefix}_map_by_probe.png",
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    branch_meta, branch_values, run_config = load_branch_run(args.branch_run_dir, mode=args.mode)
    branch_meta["class_label"] = branch_meta["branch_primary_label"].astype(str)
    branch_meta["probe_label"] = branch_meta["branch_secondary_label"].astype(str)
    branch_meta = balanced_sample_metadata(
        branch_meta,
        seed=args.seed,
        max_samples=args.max_comparison_samples,
    )
    branch_values = branch_values[branch_meta["source_index"].to_numpy()]
    branch_sample_keys = branch_meta["sample_key"].astype(str).tolist()

    direct_meta, direct_values = load_direct_mode(args.mode, sample_keys=branch_sample_keys)
    direct_meta = direct_meta.rename(columns={"subclass_label": "probe_label"})
    direct_meta["class_label"] = direct_meta["class_label"].astype(str)
    direct_meta["probe_label"] = direct_meta["probe_label"].astype(str)
    shared_sample_keys = direct_meta["sample_key"].astype(str).tolist()

    branch_keep = branch_meta["sample_key"].astype(str).isin(shared_sample_keys)
    branch_meta = branch_meta[branch_keep].copy().reset_index(drop=True)
    branch_values = branch_values[branch_keep.to_numpy()]
    representations: list[tuple[str, pd.DataFrame, np.ndarray]] = [
        ("direct", direct_meta.copy().reset_index(drop=True), direct_values),
        ("specialized_branch", branch_meta.copy(), branch_values),
    ]

    for name, run_dir in [("shared_v5", V5_RUN_DIR), ("shared_v7", V7_RUN_DIR)]:
        loaded = try_load_shared_subset(run_dir, shared_sample_keys, mode=args.mode)
        if loaded is None:
            continue
        meta, values = loaded
        meta["class_label"] = meta["label_optional"].astype(str)
        meta["probe_label"] = meta["subclass_label"].astype(str)
        representations.append((name, meta[["sample_key", "class_label", "probe_label"]].copy(), values))

    metric_rows = []
    transfer_rows = []
    for name, meta, values in representations:
        scorecard = representation_scorecard(
            values,
            meta["class_label"].to_numpy(),
            meta["probe_label"].to_numpy(),
            seed=args.seed,
            k=args.knn_k,
        )
        for metric, value in scorecard.items():
            metric_rows.append({"representation": name, "metric": metric, "value": value})
        if meta["probe_label"].fillna("").astype(str).nunique() > 1:
            from gaira.demo.small2023_branch_utils import cross_probe_transfer_metrics

            transfer = cross_probe_transfer_metrics(values, meta["class_label"].to_numpy(), meta["probe_label"].to_numpy())
            transfer["representation"] = name
            transfer_rows.append(transfer)
        plot_representation(meta, values, seed=args.seed, prefix=name, output_dir=args.output_dir)

    metric_df = pd.DataFrame(metric_rows)
    v2_summary = try_load_v2_summary(args.mode)
    if v2_summary is not None and not v2_summary.empty:
        metric_df = pd.concat([metric_df, v2_summary], ignore_index=True)
    metric_df.to_csv(args.output_dir / "representation_scorecard.csv", index=False)

    transfer_df = pd.concat(transfer_rows, ignore_index=True) if transfer_rows else pd.DataFrame(
        [{"representation": "not_available", "direction": "not_available", "train_probe": "", "test_probe": "", "accuracy": float("nan"), "macro_f1": float("nan")}]
    )
    transfer_df.to_csv(args.output_dir / "cross_probe_transfer_metrics.csv", index=False)

    class_plot = metric_df[metric_df["metric"].isin(["silhouette_class", "nn_purity_class", "top1_match_class", "class_predict_macro_f1"])].copy()
    probe_plot = metric_df[metric_df["metric"].isin(["silhouette_probe", "nn_purity_probe", "top1_match_probe", "probe_predict_macro_f1"])].copy()
    gap_plot = metric_df[metric_df["metric"].isin(["class_probe_silhouette_ratio", "class_probe_nn_purity_ratio", "class_minus_probe_predictability_gap"])].copy()
    transfer_plot = transfer_df.dropna(subset=["macro_f1"]).copy()

    if not class_plot.empty:
        save_barplot(class_plot, x="metric", y="value", hue="representation", title=f"{args.mode} class metrics", output_path=args.output_dir / "class_metric_scorecard.png")
    if not probe_plot.empty:
        save_barplot(probe_plot, x="metric", y="value", hue="representation", title=f"{args.mode} probe metrics", output_path=args.output_dir / "probe_metric_scorecard.png")
    if not gap_plot.empty:
        save_barplot(gap_plot, x="metric", y="value", hue="representation", title=f"{args.mode} class-vs-probe gap metrics", output_path=args.output_dir / "class_probe_gap_scorecard.png")
    if not transfer_plot.empty:
        save_barplot(transfer_plot, x="direction", y="macro_f1", hue="representation", title=f"{args.mode} cross-probe transfer macro F1", output_path=args.output_dir / "cross_probe_transfer_scorecard.png")

    summary_lines = [
        f"# small2023 {args.mode} Branch Diagnostics",
        "",
        f"- Classes analyzed: {', '.join(mode_labels(args.mode))}",
        f"- Specialized branch run: `{args.branch_run_dir}`",
        f"- Shared backbone init: `{run_config.get('init_checkpoint', '')}`",
        "",
        "This comparison treats geometry plots as descriptive only. The decision metrics are the class/probe scorecards and cross-probe transfer tests.",
    ]
    if args.mode == "cellline":
        summary_lines.extend(
            [
                "",
                "Important limitation:",
                "- The cell-line subset only contains one probe family (`fig3_norm_archive`).",
                "- Cross-probe nuisance suppression cannot be tested directly for this branch.",
                "- Cell-line readiness must therefore be judged on class structure and composition coherence, not on cross-probe transfer.",
            ]
        )
    (args.output_dir / "diagnostic_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
