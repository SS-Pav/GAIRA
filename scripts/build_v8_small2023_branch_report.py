#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build small2023 branch summary report.")
    parser.add_argument("--mode", choices=["cellline", "mixture"], required=True)
    parser.add_argument("--branch-run-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--composition-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def metric_value(df: pd.DataFrame, representation: str, metric: str) -> float:
    subset = df[(df["representation"] == representation) & (df["metric"] == metric)]
    if subset.empty:
        return float("nan")
    return float(subset.iloc[0]["value"])


def main() -> None:
    args = parse_args()
    scorecard = pd.read_csv(args.diagnostic_dir / "representation_scorecard.csv")
    transfer = pd.read_csv(args.diagnostic_dir / "cross_probe_transfer_metrics.csv")
    composition = pd.read_csv(args.composition_dir / "class_composition_summary.csv")

    shared_v7_class = metric_value(scorecard, "shared_v7", "class_predict_macro_f1")
    shared_v7_probe = metric_value(scorecard, "shared_v7", "probe_predict_macro_f1")
    branch_class = metric_value(scorecard, "specialized_branch", "class_predict_macro_f1")
    branch_probe = metric_value(scorecard, "specialized_branch", "probe_predict_macro_f1")
    branch_silhouette_class = metric_value(scorecard, "specialized_branch", "silhouette_class")
    branch_silhouette_probe = metric_value(scorecard, "specialized_branch", "silhouette_probe")
    branch_transfer = metric_value(scorecard, "specialized_branch", "cross_probe_transfer_macro_f1_mean")
    shared_transfer = metric_value(scorecard, "shared_v7", "cross_probe_transfer_macro_f1_mean")
    top_theme_diversity = int(composition["top_theme"].nunique()) if "top_theme" in composition.columns else 0

    if args.mode == "cellline":
        readiness = "needs parameter tuning first"
        if branch_class > shared_v7_class and branch_silhouette_class > metric_value(scorecard, "shared_v7", "silhouette_class"):
            readiness = "not ready yet"
    else:
        if branch_class > shared_v7_class and (np.isnan(shared_v7_probe) or branch_probe < shared_v7_probe) and branch_transfer > shared_transfer:
            readiness = "ready for GPU"
        elif branch_class >= shared_v7_class:
            readiness = "needs parameter tuning first"
        else:
            readiness = "not ready yet"

    lines = [
        f"# small2023 {args.mode} Branch Report",
        "",
        f"- Branch run: `{args.branch_run_dir}`",
        f"- Readiness: **{readiness}**",
        "",
        "Core comparison against shared v7:",
        f"- class macro F1: {shared_v7_class:.4f} -> {branch_class:.4f}",
        f"- probe macro F1: {shared_v7_probe:.4f} -> {branch_probe:.4f}",
        f"- class silhouette: {metric_value(scorecard, 'shared_v7', 'silhouette_class'):.4f} -> {branch_silhouette_class:.4f}",
        f"- probe silhouette: {metric_value(scorecard, 'shared_v7', 'silhouette_probe'):.4f} -> {branch_silhouette_probe:.4f}",
        f"- cross-probe transfer macro F1: {shared_transfer:.4f} -> {branch_transfer:.4f}",
        "",
        "Composition interpretability:",
        f"- class count: {len(composition)}",
        f"- distinct top themes across classes: {top_theme_diversity}",
        "",
        "Blunt interpretation:",
    ]

    if args.mode == "cellline":
        lines.extend(
            [
                "- The branch can improve cell-line class structure, but cross-probe nuisance suppression is not testable here because the cell-line subset has only one probe family.",
                "- This means the branch is learning class structure in a constrained archive regime, not proving probe-invariant biology.",
            ]
        )
    else:
        if branch_class > shared_v7_class and (np.isnan(shared_v7_probe) or branch_probe < shared_v7_probe):
            lines.append("- The specialized branch is moving in the right direction: more biological class structure with less probe recoverability than shared v7.")
        elif branch_class > shared_v7_class:
            lines.append("- The specialized branch is improving class structure, but probe nuisance is still too recoverable.")
        else:
            lines.append("- The specialized branch is still learning a mix of class and nuisance structure rather than clearly improving biological separation.")
    args.output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
