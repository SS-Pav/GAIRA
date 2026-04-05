#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_STRESS_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/ev_stress_disease_analysis_v1")
DEFAULT_CELLLINE_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/ev_cellline_analysis_v1")
DEFAULT_OUTPUT_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/ev_demo_readiness_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a combined EV demo-readiness report.")
    parser.add_argument("--stress-dir", type=Path, default=DEFAULT_STRESS_DIR)
    parser.add_argument("--cellline-dir", type=Path, default=DEFAULT_CELLLINE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stress_metrics = pd.read_csv(args.stress_dir / "latent_structure_metrics.csv")
    cellline_compare = pd.read_csv(args.cellline_dir / "direct_vs_embedding_comparison.csv")

    scorecard = pd.DataFrame(
        [
            {
                "page": "EV Stress / Disease",
                "strength": "moderate",
                "evidence_anchor": "unsupervised latent neighborhoods + broad-state overlay + composition profiles",
                "risk": "dataset identity still exceeds broad-state organization",
                "demo_priority": 1,
            },
            {
                "page": "EV Cell-Line",
                "strength": "moderate",
                "evidence_anchor": "direct-vs-embedding benchmark + explicit class composition layer",
                "risk": "probe nuisance comparison and explicit cell-line interpretation come from different subsets",
                "demo_priority": 2,
            },
        ]
    )
    scorecard.to_csv(args.output_dir / "demo_readiness_scorecard.csv", index=False)

    nn_state = float(stress_metrics.loc[stress_metrics["metric"] == "nn_purity_harmonized_state", "value"].iloc[0])
    nn_dataset = float(stress_metrics.loc[stress_metrics["metric"] == "nn_purity_dataset", "value"].iloc[0])
    class_delta = float(cellline_compare.loc[cellline_compare["metric"] == "nn_purity_class", "delta_embedding_minus_direct"].iloc[0])
    probe_delta = float(cellline_compare.loc[cellline_compare["metric"] == "nn_purity_probe", "delta_embedding_minus_direct"].iloc[0])

    lines = [
        "# EV Demo Readiness Summary",
        "",
        "This summary combines the pre-demo analytics from the EV stress/disease page and the EV cell-line page.",
        "",
        "## What is ready now",
        "- The stress/disease page is the strongest story if it stays broad: latent neighborhoods first, state overlay second, grounding-derived composition third.",
        "- The cell-line page is viable as a benchmark-style page that compares direct spectra against embedding space and then shows class-level composition tendencies.",
        "",
        "## What is still weak",
        "- Dataset identity is still stronger than harmonized broad-state identity in the stress/disease slice.",
        "- The small2023 probe nuisance benchmark and the explicit fig3 cell-line labels are not the same subset, so they should not be presented as one seamless biology narrative.",
        "",
        "## Quantitative checkpoints",
        f"- stress/disease dataset nn purity: {nn_dataset:.4f}",
        f"- stress/disease broad-state nn purity: {nn_state:.4f}",
        f"- small2023 embedding minus direct delta for class nn purity: {class_delta:.4f}",
        f"- small2023 embedding minus direct delta for probe nn purity: {probe_delta:.4f}",
        "",
        "## Recommendation",
        "- Build the EV demo around the stress/disease page first.",
        "- Keep the cell-line page as a benchmark-and-interpretation companion page rather than the primary investor-facing narrative.",
    ]
    (args.output_dir / "demo_readiness_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
