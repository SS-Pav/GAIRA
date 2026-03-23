from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("/Volumes/SSD_Rad/GAIRA_DATA")
OUTPUT_DIR = ROOT / "processed" / "query_aware_context_routing_final_polish"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "report"


def main() -> None:
    required_paths = [
        OUTPUT_DIR / "final_visual_failure_modes.md",
        OUTPUT_DIR / "contamination_metric_definitions.md",
        TABLE_DIR / "near_family_overlap_summary.csv",
        TABLE_DIR / "cross_domain_contamination_summary.csv",
        TABLE_DIR / "contamination_split_by_family.csv",
        TABLE_DIR / "routing_winner_margin_summary.csv",
        TABLE_DIR / "final_routing_status_summary.csv",
        TABLE_DIR / "representative_case_table.csv",
        TABLE_DIR / "revised_family_specific_metrics.csv",
        TABLE_DIR / "revised_normalized_routing_scores.csv",
        REPORT_DIR / "query_aware_context_routing_final_polish_report.md",
        REPORT_DIR / "query_aware_context_routing_final_polish_report.pdf",
        FIGURE_DIR / "figure2_intended_vs_forced_performance_heatmap.png",
        FIGURE_DIR / "figure3_near_family_overlap_heatmap.png",
        FIGURE_DIR / "figure4_cross_domain_contamination_heatmap.png",
        FIGURE_DIR / "figure5_winner_runner_up_margin_panel.png",
        FIGURE_DIR / "figure8_final_routing_status_summary.png",
    ]
    for path in required_paths:
        print(f"{path.name}: {path.exists()}")

    winner_df = pd.read_csv(TABLE_DIR / "routing_winner_margin_summary.csv")
    print("\nWinner margin summary:")
    print(winner_df[["intended_family", "best_forced_family", "runner_up_forced_family", "winner_margin", "winner_margin_category", "final_status"]].to_string(index=False))


if __name__ == "__main__":
    main()
