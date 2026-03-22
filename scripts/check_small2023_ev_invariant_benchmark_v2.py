from pathlib import Path

import pandas as pd


def main() -> None:
    output_dir = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/small2023_ev_invariant_embedding_v2")
    required = [
        output_dir / "small2023_ev_invariant_dataset_v2.npz",
        output_dir / "small2023_ev_invariant_dataset_v2_metadata.csv",
        output_dir / "benchmark_sample_counts_v2.csv",
        output_dir / "comparison_cross_probe_metrics_v2.csv",
        output_dir / "geometry_metrics_v2.csv",
        output_dir / "class_probe_distance_summary_v2.csv",
        output_dir / "mixture_ordering_summary_v2.csv",
        output_dir / "embedding_tsne_by_class_v2.png",
        output_dir / "embedding_tsne_by_probe_v2.png",
        output_dir / "v1_vs_v2_metric_comparison.png",
        output_dir / "embedding_summary_v2.txt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("Missing v2 benchmark outputs:")
        for path in missing:
            print(f"  {path}")
        return

    counts_df = pd.read_csv(output_dir / "benchmark_sample_counts_v2.csv")
    comparison_df = pd.read_csv(output_dir / "comparison_cross_probe_metrics_v2.csv")
    geometry_df = pd.read_csv(output_dir / "geometry_metrics_v2.csv")
    distance_df = pd.read_csv(output_dir / "class_probe_distance_summary_v2.csv")
    mixture_df = pd.read_csv(output_dir / "mixture_ordering_summary_v2.csv")

    print("Benchmark sample counts v2:")
    print(counts_df.to_string(index=False))
    print("\nComparison metrics v2:")
    print(comparison_df.to_string(index=False))
    print("\nGeometry metrics v2:")
    print(geometry_df.to_string(index=False))
    print("\nCentroid distance summary v2:")
    print(distance_df.to_string(index=False))
    print("\nMixture ordering summary v2:")
    print(mixture_df.to_string(index=False))
    print("\nSummary excerpt:")
    print((output_dir / "embedding_summary_v2.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
