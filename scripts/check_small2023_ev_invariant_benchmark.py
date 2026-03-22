from pathlib import Path

import pandas as pd


def main() -> None:
    output_dir = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/small2023_ev_invariant_embedding")
    baseline_path = output_dir / "baseline_cross_probe_metrics.csv"
    embedding_path = output_dir / "embedding_cross_probe_metrics.csv"
    geometry_path = output_dir / "geometry_metrics.csv"
    distance_path = output_dir / "class_probe_distance_summary.csv"
    counts_path = output_dir / "benchmark_sample_counts.csv"
    summary_path = output_dir / "embedding_summary.txt"

    required_paths = [baseline_path, embedding_path, geometry_path, distance_path, counts_path, summary_path]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        print("Missing benchmark outputs:")
        for path in missing:
            print(f"  {path}")
        return

    counts_df = pd.read_csv(counts_path)
    baseline_df = pd.read_csv(baseline_path)
    embedding_df = pd.read_csv(embedding_path)
    geometry_df = pd.read_csv(geometry_path)
    distance_df = pd.read_csv(distance_path)

    print("Benchmark sample counts:")
    print(counts_df.to_string(index=False))
    print("\nBaseline metrics:")
    print(baseline_df.to_string(index=False))
    print("\nEmbedding metrics:")
    print(embedding_df.to_string(index=False))
    print("\nGeometry metrics:")
    print(geometry_df.to_string(index=False))
    print("\nCross-probe centroid distance summary:")
    print(distance_df.to_string(index=False))
    print("\nSummary excerpt:")
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
