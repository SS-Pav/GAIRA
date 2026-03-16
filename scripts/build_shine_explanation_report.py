import sys
from pathlib import Path

import pandas as pd


def get_supporting_text(group_df: pd.DataFrame, rank: int) -> str:
    """Return one supporting chunk text by rank, or empty text if missing."""
    match_df = group_df[group_df["chunk_rank"] == rank]
    if match_df.empty:
        return ""
    return str(match_df.iloc[0]["chunk_text"])


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_storage_config, resolve_storage_path

    storage_config = get_storage_config()
    processed_root = resolve_storage_path(storage_config.get("processed_data"))

    if processed_root is None:
        print("The storage config is missing processed_data.")
        return

    summary_path = (
        processed_root
        / "shine_class_reference_matches"
        / "shine_class_consensus_summary.csv"
    )
    interpreted_path = (
        processed_root
        / "shine_class_reference_matches"
        / "shine_class_consensus_interpreted.csv"
    )
    chunks_path = (
        processed_root
        / "shine_class_reference_matches"
        / "shine_class_supporting_chunks.csv"
    )
    output_path = (
        processed_root
        / "shine_class_reference_matches"
        / "shine_class_explanation_report.csv"
    )

    for path in [summary_path, interpreted_path, chunks_path]:
        if not path.exists():
            print(f"Required input file not found: {path}")
            return

    summary_df = pd.read_csv(summary_path)
    interpreted_df = pd.read_csv(interpreted_path)
    chunks_df = pd.read_csv(chunks_path)

    merged_df = summary_df.merge(
        interpreted_df,
        on=["class_label", "subclass_label"],
        how="left",
        suffixes=("_summary", ""),
    )

    report_rows: list[dict] = []
    for row in merged_df.to_dict(orient="records"):
        row_series = pd.Series(row)
        chunk_group_df = chunks_df[
            (chunks_df["class_label"] == row_series["class_label"])
            & (chunks_df["subclass_label"] == row_series["subclass_label"])
        ].copy()

        supporting_chunk_1 = get_supporting_text(chunk_group_df, 1)
        supporting_chunk_2 = get_supporting_text(chunk_group_df, 2)
        supporting_chunk_3 = get_supporting_text(chunk_group_df, 3)

        explanation_text = (
            f"This class shows a {row_series['top_biochemical_class_1']}/"
            f"{row_series['top_biochemical_class_2']} analog pattern with strongest support in "
            f"{row_series['dominant_region_1']} and {row_series['dominant_region_2']} cm^-1. "
            f"Mapped semantic regions are {row_series['region_semantic_label_1']} and "
            f"{row_series['region_semantic_label_2']}. "
        )
        if supporting_chunk_1:
            explanation_text += (
                "Supporting knowledge chunks emphasize "
                f"{supporting_chunk_1} "
            )
        if row_series.get("confounder_warnings"):
            explanation_text += (
                f"Important confounders include {row_series['confounder_warnings']}. "
            )
        explanation_text += (
            "This should not be interpreted as literal molecule identification."
        )

        report_rows.append(
            {
                "class_label": row_series["class_label"],
                "subclass_label": row_series["subclass_label"],
                "top_biochemical_class_1": row_series["top_biochemical_class_1"],
                "top_biochemical_class_2": row_series["top_biochemical_class_2"],
                "dominant_region_1": row_series["dominant_region_1"],
                "dominant_region_2": row_series["dominant_region_2"],
                "region_semantic_label_1": row_series["region_semantic_label_1"],
                "region_semantic_label_2": row_series["region_semantic_label_2"],
                "knowledge_supported_groups": row_series.get("knowledge_supported_groups", ""),
                "confounder_warnings": row_series.get("confounder_warnings", ""),
                "supporting_chunk_1": supporting_chunk_1,
                "supporting_chunk_2": supporting_chunk_2,
                "supporting_chunk_3": supporting_chunk_3,
                "explanation_text": explanation_text,
            }
        )

    report_df = pd.DataFrame(report_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, index=False)

    print(f"SHINE explanation report written to: {output_path}")
    print(f"Rows written: {len(report_df)}")
    print(report_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
