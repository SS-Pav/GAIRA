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

    matches_dir = processed_root / "shine_class_reference_matches"
    summary_path = matches_dir / "shine_class_consensus_summary.csv"

    if not summary_path.exists():
        print(f"Consensus summary file not found: {summary_path}")
        return

    summary_df = pd.read_csv(summary_path)
    if summary_df.empty:
        print("Consensus summary is empty.")
        return

    suspicious_df = summary_df[summary_df["suspicious_flag"] == True].copy()
    class_counter = (
        summary_df["top_biochemical_class_1"]
        .value_counts(dropna=False)
        .rename_axis("top_biochemical_class_1")
        .reset_index(name="count")
    )
    region_counter = (
        summary_df["dominant_region_1"]
        .value_counts(dropna=False)
        .rename_axis("dominant_region_1")
        .reset_index(name="count")
    )

    print(f"Number of class consensus rows: {len(summary_df)}")
    print("\nTop primary biochemical consensus classes:")
    print(class_counter.to_string(index=False))
    print("\nTop primary Raman regions:")
    print(region_counter.to_string(index=False))
    print("\nPer-class interpretations:")
    print(
        summary_df[
            ["class_label", "subclass_label", "top_biochemical_class_1", "dominant_region_1", "interpretation_text"]
        ].to_string(index=False)
    )
    print("\nSuspicious or unstable classes:")
    if suspicious_df.empty:
        print("None flagged.")
    else:
        print(
            suspicious_df[
                ["class_label", "subclass_label", "top_component_examples", "interpretation_text"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
