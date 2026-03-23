from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("/Volumes/SSD_Rad/GAIRA_DATA")
OUTPUT_DIR = ROOT / "processed" / "query_aware_context_routing_polish"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "report"


def main() -> None:
    required_paths = [
        OUTPUT_DIR / "routing_metric_failure_modes.md",
        OUTPUT_DIR / "routing_metric_definitions.md",
        TABLE_DIR / "family_specific_metrics.csv",
        TABLE_DIR / "normalized_routing_scores.csv",
        TABLE_DIR / "before_after_metrics.csv",
        TABLE_DIR / "family_best_routing_summary.csv",
        REPORT_DIR / "query_aware_context_routing_polish_report.md",
        REPORT_DIR / "query_aware_context_routing_polish_report.pdf",
        FIGURE_DIR / "figure1_routing_family_design_map.png",
        FIGURE_DIR / "figure2_query_family_forced_family_performance_heatmap.png",
        FIGURE_DIR / "figure3_query_family_forced_family_contamination_heatmap.png",
        FIGURE_DIR / "figure8_final_routing_summary.png",
    ]
    for path in required_paths:
        print(f"{path.name}: {path.exists()}")

    normalized_df = pd.read_csv(TABLE_DIR / "normalized_routing_scores.csv")
    best_df = pd.read_csv(TABLE_DIR / "family_best_routing_summary.csv")

    print("\nBest routing family by intended family:")
    print(best_df[["intended_family", "forced_family", "normalized_routing_score"]].to_string(index=False))

    print("\nNormalized score head:")
    print(
        normalized_df[
            [
                "intended_family",
                "forced_family",
                "normalized_routing_score",
                "mean_top1_support_correct",
                "mean_cross_domain_contamination",
            ]
        ]
        .head(12)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
