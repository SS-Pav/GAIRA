from __future__ import annotations

from pathlib import Path

import pandas as pd

VALIDATION_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/small2023_ev_invariant_embedding_v2_validation")


def main() -> None:
    duplicate_df = pd.read_csv(VALIDATION_DIR / "duplicate_audit_summary.csv")
    nn_df = pd.read_csv(VALIDATION_DIR / "cross_probe_nearest_neighbor_summary.csv")
    seed_df = pd.read_csv(VALIDATION_DIR / "multi_seed_transfer_summary.csv")
    ablation_df = pd.read_csv(VALIDATION_DIR / "v2_ablation_metrics.csv")
    summary_text = (VALIDATION_DIR / "v2_validation_summary.txt").read_text(encoding="utf-8")

    print("Duplicate audit summary:")
    print(duplicate_df.to_string(index=False))
    print()

    print("Nearest-neighbor summary preview:")
    print(nn_df.head(12).to_string(index=False))
    print()

    print("Multi-seed transfer summary:")
    print(seed_df.to_string(index=False))
    print()

    print("Ablation metrics:")
    print(ablation_df.to_string(index=False))
    print()

    print("Validation summary excerpt:")
    print("\n".join(summary_text.splitlines()[:30]))


if __name__ == "__main__":
    main()
