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

    report_path = processed_root / "shine_class_reference_matches" / "shine_class_explanation_report_v4.csv"
    retrieval_path = processed_root / "shine_class_reference_matches" / "shine_class_supporting_chunks_v4.csv"

    if not report_path.exists():
        print(f"Explanation report v4 not found: {report_path}")
        return
    if not retrieval_path.exists():
        print(f"Supporting chunks v4 file not found: {retrieval_path}")
        return

    report_df = pd.read_csv(report_path)
    retrieval_df = pd.read_csv(retrieval_path)

    confidence_df = (
        report_df["confidence_tier"]
        .value_counts(dropna=False)
        .rename_axis("confidence_tier")
        .reset_index(name="count")
    )
    top_chunk_df = retrieval_df[retrieval_df["chunk_rank"] == 1].copy()
    unique_top_chunks = int(top_chunk_df["chunk_id"].replace("", pd.NA).dropna().nunique())
    reused_chunks_df = (
        top_chunk_df["chunk_id"]
        .value_counts(dropna=False)
        .rename_axis("chunk_id")
        .reset_index(name="count")
        .sort_values(["count", "chunk_id"], ascending=[False, True])
    )
    role_distribution_df = (
        retrieval_df["chunk_role"]
        .value_counts(dropna=False)
        .rename_axis("chunk_role")
        .reset_index(name="count")
    )
    confounder_top_rows = int((top_chunk_df["chunk_role"] == "confounder_or_caution").sum())

    print(f"Explanation report v4 rows: {len(report_df)}")
    print("\nConfidence tier distribution:")
    print(confidence_df.to_string(index=False))
    print(f"\nUnique top chunks used: {unique_top_chunks}")
    print("\nMost reused top chunks:")
    print(reused_chunks_df.head(10).to_string(index=False))
    print("\nRole distribution among selected chunks:")
    print(role_distribution_df.to_string(index=False))
    print(f"\nRows where top chunk is confounder-role: {confounder_top_rows}")
    print("\nFirst 10 rows:")
    print(
        report_df[
            [
                "class_label",
                "subclass_label",
                "supporting_role_1",
                "supporting_role_2",
                "supporting_role_3",
                "confidence_tier",
                "confidence_reason",
                "evidence_summary",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
