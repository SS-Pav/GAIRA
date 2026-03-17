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

    benchmark_dir = processed_root / "hcc_serum_benchmark_v1"
    metrics_path = benchmark_dir / "hcc_serum_benchmark_metrics.csv"
    batch_dist_path = benchmark_dir / "hcc_serum_batch_distribution.csv"
    batch_diag_path = benchmark_dir / "hcc_serum_batch_diagnostic_metrics.csv"
    geometry_path = benchmark_dir / "hcc_serum_geometry_metrics.csv"
    summary_path = benchmark_dir / "hcc_serum_benchmark_summary.txt"

    if not metrics_path.exists():
        print(f"Benchmark metrics not found: {metrics_path}")
        return

    metrics_df = pd.read_csv(metrics_path)
    batch_dist_df = pd.read_csv(batch_dist_path)
    batch_diag_df = pd.read_csv(batch_diag_path)
    geometry_df = pd.read_csv(geometry_path)

    print("Mean benchmark metrics by split/model:")
    print(
        metrics_df.groupby(["split_name", "model_name"])[["accuracy", "balanced_accuracy", "roc_auc"]]
        .mean()
        .reset_index()
        .to_string(index=False)
    )
    print("\nBatch distribution:")
    print(batch_dist_df.to_string(index=False))
    print("\nBatch diagnostic mean metrics:")
    print(batch_diag_df[["accuracy", "balanced_accuracy"]].mean().to_string())
    print("\nGeometry metrics:")
    print(geometry_df.to_string(index=False))
    print("\nSummary path:")
    print(summary_path)


if __name__ == "__main__":
    main()
