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

    report_path = (
        processed_root
        / "shine_class_reference_matches"
        / "shine_class_explanation_report_v2.csv"
    )
    if not report_path.exists():
        print(f"Explanation report v2 not found: {report_path}")
        return

    report_df = pd.read_csv(report_path)
    chunk_1_count = int(report_df["supporting_chunk_1"].fillna("").str.strip().ne("").sum())
    chunk_2_count = int(report_df["supporting_chunk_2"].fillna("").str.strip().ne("").sum())
    chunk_3_count = int(report_df["supporting_chunk_3"].fillna("").str.strip().ne("").sum())
    confidence_df = (
        report_df["confidence_tier"]
        .value_counts(dropna=False)
        .rename_axis("confidence_tier")
        .reset_index(name="count")
    )

    print(f"Explanation report v2 rows: {len(report_df)}")
    print(f"Rows with non-empty supporting_chunk_1: {chunk_1_count}")
    print(f"Rows with non-empty supporting_chunk_2: {chunk_2_count}")
    print(f"Rows with non-empty supporting_chunk_3: {chunk_3_count}")
    print("\nConfidence tier distribution:")
    print(confidence_df.to_string(index=False))
    print("\nFirst 10 rows:")
    print(
        report_df[
            [
                "class_label",
                "subclass_label",
                "evidence_summary",
                "confidence_tier",
                "explanation_text",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
