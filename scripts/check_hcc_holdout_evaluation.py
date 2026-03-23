from pathlib import Path
import sys

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_storage_paths, require_data_root_exists

    storage_paths = require_data_root_exists()
    base_dir = storage_paths["processed_data"] / "hcc_holdout_evaluation"
    required = [
        base_dir / "eval_db" / "gaira_hcc_holdout_eval.duckdb",
        base_dir / "raw_outputs" / "hcc_holdout_inference_results.json",
        base_dir / "raw_outputs" / "hcc_holdout_query_outputs.csv",
        base_dir / "raw_outputs" / "hcc_holdout_theme_outputs_long.csv",
        base_dir / "tables" / "hcc_holdout_group_theme_summary.csv",
        base_dir / "tables" / "hcc_holdout_usefulness_metrics.csv",
        base_dir / "cases" / "hcc_holdout_representative_cases.csv",
        base_dir / "report" / "hcc_holdout_evaluation_report.md",
        base_dir / "report" / "hcc_holdout_evaluation_report.pdf",
        base_dir / "figures" / "figure1_hcc_holdout_design.png",
        base_dir / "figures" / "figure8_holdout_usefulness_summary.png",
    ]
    for path in required:
        print(f"{path.name}: {path.exists()}")

    query_path = base_dir / "raw_outputs" / "hcc_holdout_query_outputs.csv"
    theme_path = base_dir / "raw_outputs" / "hcc_holdout_theme_outputs_long.csv"
    metric_path = base_dir / "tables" / "hcc_holdout_usefulness_metrics.csv"
    if query_path.exists():
        query_df = pd.read_csv(query_path)
        print(f"Query rows: {len(query_df)}")
        if "class_label" in query_df.columns:
            print("Query class counts:")
            print(query_df["class_label"].value_counts().sort_index().to_string())
    if theme_path.exists():
        theme_df = pd.read_csv(theme_path)
        print(f"Theme rows: {len(theme_df)}")
        if "theme_name" in theme_df.columns:
            print(f"Unique themes: {theme_df['theme_name'].nunique()}")
    if metric_path.exists():
        metric_df = pd.read_csv(metric_path)
        print("Usefulness metrics:")
        print(metric_df.to_string(index=False))


if __name__ == "__main__":
    main()
