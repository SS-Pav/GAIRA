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

    report_path = processed_root / "shine_class_reference_matches" / "shine_class_explanation_report_v3.csv"
    retrieval_path = processed_root / "shine_class_reference_matches" / "shine_class_supporting_chunks_v3.csv"

    if not report_path.exists():
        print(f"Explanation report v3 not found: {report_path}")
        return

    report_df = pd.read_csv(report_path)
    retrieval_df = pd.read_csv(retrieval_path) if retrieval_path.exists() else pd.DataFrame()

    chunk_1_count = int(report_df["supporting_chunk_1"].fillna("").str.strip().ne("").sum())
    chunk_2_count = int(report_df["supporting_chunk_2"].fillna("").str.strip().ne("").sum())
    chunk_3_count = int(report_df["supporting_chunk_3"].fillna("").str.strip().ne("").sum())
    confidence_df = (
        report_df["confidence_tier"]
        .value_counts(dropna=False)
        .rename_axis("confidence_tier")
        .reset_index(name="count")
    )

    if retrieval_df.empty:
        unique_top_chunks = 0
        reused_chunks_df = pd.DataFrame(columns=["chunk_id", "count"])
    else:
        top_chunk_df = retrieval_df[retrieval_df["chunk_rank"] == 1].copy()
        unique_top_chunks = int(top_chunk_df["chunk_id"].replace("", pd.NA).dropna().nunique())
        reused_chunks_df = (
            top_chunk_df["chunk_id"]
            .value_counts(dropna=False)
            .rename_axis("chunk_id")
            .reset_index(name="count")
            .sort_values(["count", "chunk_id"], ascending=[False, True])
        )

    print(f"Explanation report v3 rows: {len(report_df)}")
    print(f"Rows with non-empty supporting_chunk_1: {chunk_1_count}")
    print(f"Rows with non-empty supporting_chunk_2: {chunk_2_count}")
    print(f"Rows with non-empty supporting_chunk_3: {chunk_3_count}")
    print(f"Unique top chunks used: {unique_top_chunks}")
    print("\nConfidence tier distribution:")
    print(confidence_df.to_string(index=False))
    print("\nMost reused top chunks:")
    if reused_chunks_df.empty:
        print("(no retrieval rows)")
    else:
        print(reused_chunks_df.head(10).to_string(index=False))
    print("\nFirst 10 rows:")
    print(
        report_df[
            [
                "class_label",
                "subclass_label",
                "evidence_summary",
                "confidence_tier",
                "confidence_reason",
                "explanation_text",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
