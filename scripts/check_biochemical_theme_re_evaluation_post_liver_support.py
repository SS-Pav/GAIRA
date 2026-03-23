from pathlib import Path
import sys

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_storage_paths, require_data_root_exists

    storage_paths = require_data_root_exists()
    base_dir = storage_paths["processed_data"] / "biochemical_theme_re_evaluation_post_liver_support"

    required = [
        base_dir / "raw_outputs" / "query_outputs_before.csv",
        base_dir / "raw_outputs" / "query_outputs_after.csv",
        base_dir / "raw_outputs" / "theme_outputs_before_long.csv",
        base_dir / "raw_outputs" / "theme_outputs_after_long.csv",
        base_dir / "raw_outputs" / "hcc_query_outputs_before.csv",
        base_dir / "raw_outputs" / "hcc_query_outputs_after.csv",
        base_dir / "tables" / "before_after_metrics.csv",
        base_dir / "tables" / "support_visibility_before_after.csv",
        base_dir / "tables" / "representative_case_table.csv",
        base_dir / "tables" / "per_track_summary.csv",
        base_dir / "tables" / "hcc_holdout_before_after_metrics.csv",
        base_dir / "tables" / "hcc_holdout_calibrated_metrics_current.csv",
        base_dir / "tables" / "track_improvement_summary.csv",
        base_dir / "report" / "biochemical_theme_re_evaluation_post_liver_support_report.md",
        base_dir / "report" / "biochemical_theme_re_evaluation_post_liver_support_report.pdf",
        base_dir / "figures" / "figure1_evaluation_suite_overview.png",
        base_dir / "figures" / "figure8_final_usefulness_summary.png",
    ]
    for path in required:
        print(f"{path.name}: {path.exists()}")

    metrics_path = base_dir / "tables" / "before_after_metrics.csv"
    hcc_path = base_dir / "tables" / "hcc_holdout_before_after_metrics.csv"
    support_path = base_dir / "tables" / "support_visibility_before_after.csv"
    if not metrics_path.exists() or not hcc_path.exists() or not support_path.exists():
        raise SystemExit("Missing one or more required post-liver-support re-evaluation outputs.")

    metrics_df = pd.read_csv(metrics_path)
    hcc_df = pd.read_csv(hcc_path)
    support_df = pd.read_csv(support_path)

    required_tracks = {
        "controlled_analyte_specificity",
        "ev_mixture_coherence",
        "covid_serum_usefulness",
        "liver_serum_cohort_reasoning",
    }
    found_tracks = set(metrics_df["track_name"].dropna().astype(str).unique().tolist())
    missing_tracks = sorted(required_tracks - found_tracks)
    if missing_tracks:
        raise SystemExit(f"Missing expected tracks in before_after_metrics.csv: {', '.join(missing_tracks)}")

    required_hcc_metrics = {
        "theme_space_silhouette",
        "mean_abs_theme_effect_size",
        "mean_positive_confidence",
        "mean_caution_score",
    }
    found_hcc_metrics = set(hcc_df["metric_name"].dropna().astype(str).unique().tolist())
    missing_hcc = sorted(required_hcc_metrics - found_hcc_metrics)
    if missing_hcc:
        raise SystemExit(f"Missing expected HCC metrics: {', '.join(missing_hcc)}")

    liver_rows = support_df[support_df["track_name"] == "liver_serum_cohort_reasoning"].copy()
    if liver_rows.empty:
        raise SystemExit("No liver_serum_cohort_reasoning rows found in support visibility table.")

    print("\nTrack deltas:")
    print(
        metrics_df.groupby("track_name", as_index=False)["delta"]
        .sum()
        .sort_values("delta", ascending=False)
        .to_string(index=False)
    )

    print("\nHCC before/after metrics:")
    print(hcc_df.to_string(index=False))

    print("\nLiver-serum support visibility:")
    print(
        liver_rows[
            [
                "query_label",
                "n_liver_support_hits_before",
                "n_liver_support_hits_after",
                "top_tier2_source_dataset_before",
                "top_tier2_source_dataset_after",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
