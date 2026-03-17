from __future__ import annotations

from pathlib import Path

import pandas as pd

V3_DIR = Path("/Volumes/SSD_SPG/GAIRA_DATA/processed/small2023_ev_invariant_embedding_v3")


def main() -> None:
    counts_df = pd.read_csv(V3_DIR / "benchmark_sample_counts_v3.csv")
    comparison_df = pd.read_csv(V3_DIR / "comparison_cross_probe_metrics_v3.csv")
    geometry_df = pd.read_csv(V3_DIR / "geometry_metrics_v3.csv")
    distance_df = pd.read_csv(V3_DIR / "class_probe_distance_summary_v3.csv")
    mixture_df = pd.read_csv(V3_DIR / "mixture_ordering_summary_v3.csv")
    summary_text = (V3_DIR / "embedding_summary_v3.txt").read_text(encoding="utf-8")

    print("Benchmark sample counts v3:")
    print(counts_df.to_string(index=False))
    print()

    print("Comparison metrics v3:")
    print(comparison_df.to_string(index=False))
    print()

    print("Geometry metrics v3:")
    print(geometry_df.to_string(index=False))
    print()

    print("Centroid distance summary v3:")
    print(distance_df.to_string(index=False))
    print()

    print("Mixture ordering summary v3:")
    print(mixture_df.to_string(index=False))
    print()

    print("Summary excerpt:")
    print("\n".join(summary_text.splitlines()[:40]))


if __name__ == "__main__":
    main()
