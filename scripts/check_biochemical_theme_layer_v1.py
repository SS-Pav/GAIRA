from pathlib import Path
import sys

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_storage_paths, require_data_root_exists

    storage_paths = require_data_root_exists()
    base_dir = storage_paths["processed_data"] / "biochemical_theme_layer_v1"

    query_path = base_dir / "raw_outputs" / "theme_query_outputs.csv"
    theme_path = base_dir / "raw_outputs" / "theme_per_query_outputs_long.csv"
    metric_path = base_dir / "tables" / "theme_track_metrics.csv"
    report_md = base_dir / "report" / "biochemical_theme_layer_v1_report.md"
    report_pdf = base_dir / "report" / "biochemical_theme_layer_v1_report.pdf"

    for path in [query_path, theme_path, metric_path, report_md, report_pdf]:
        print(f"{path.name}: {path.exists()}")

    if not query_path.exists() or not theme_path.exists():
        return

    query_df = pd.read_csv(query_path)
    theme_df = pd.read_csv(theme_path)
    print(f"Query rows: {len(query_df)}")
    print(f"Theme rows: {len(theme_df)}")
    print("Tracks:")
    print(query_df["track_name"].value_counts().to_string())
    print("Top themes by mean score:")
    print(
        theme_df.groupby("theme_name", as_index=False)["score"]
        .mean()
        .sort_values("score", ascending=False)
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
