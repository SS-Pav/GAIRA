import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_storage_config, resolve_storage_path

    storage_config = get_storage_config()
    processed_root = resolve_storage_path(storage_config.get("processed_data"))
    if processed_root is None:
        print("The storage config is missing processed_data.")
        return

    output_dir = processed_root / "hcc_serum_paper_comparison"
    metrics_path = output_dir / "hcc_serum_paper_comparison_metrics.csv"
    comparison_path = output_dir / "hcc_serum_pcalda_vs_gaira_v1_comparison.csv"
    summary_path = output_dir / "hcc_serum_paper_comparison_summary.txt"

    if not metrics_path.exists():
        print(f"Metrics file not found: {metrics_path}")
        return

    metrics_df = pd.read_csv(metrics_path)
    comparison_df = pd.read_csv(comparison_path)

    print("PCA-LDA reproduction variants:")
    print(metrics_df.to_string(index=False))
    print("\nComparison versus GAIRA v1:")
    print(comparison_df.to_string(index=False))
    print("\nSummary path:")
    print(summary_path)


if __name__ == "__main__":
    main()
