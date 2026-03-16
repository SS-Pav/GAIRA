import sys
from pathlib import Path

import pandas as pd


def get_supporting_row(group_df: pd.DataFrame, rank: int) -> pd.Series | None:
    """Return one supporting chunk row by rank."""
    match_df = group_df[group_df["chunk_rank"] == rank]
    if match_df.empty:
        return None
    return match_df.iloc[0]


def choose_confidence_tier(group_df: pd.DataFrame) -> str:
    """Assign a conservative confidence tier from chunk diversity and specificity."""
    if group_df.empty:
        return "low"

    non_empty_rows = group_df[group_df["chunk_text"].fillna("").str.strip().ne("")]
    if non_empty_rows.empty:
        return "low"

    unique_sections = int(non_empty_rows["section"].nunique())
    max_score = float(non_empty_rows["total_score"].max())
    generic_only = bool((non_empty_rows["generic_penalty"] > 0).all())

    if unique_sections >= 2 and max_score >= 8 and not generic_only:
        return "moderate"
    return "low"


def build_evidence_summary(group_df: pd.DataFrame) -> str:
    """Summarize the main chunk evidence types in one short sentence fragment."""
    if group_df.empty:
        return "Top support is limited and mostly generic."

    sections = group_df["section"].fillna("").astype(str).tolist()
    section_text = []
    if any("protein" in section or "amide" in section for section in sections):
        section_text.append("protein/amide chunks")
    if any("lipid" in section or "ch_regions" in section for section in sections):
        section_text.append("CH deformation lipid-protein overlap notes")
    if any("nucleic" in section for section in sections):
        section_text.append("nucleic-acid region chunks")
    if any("carbohydrate" in section for section in sections):
        section_text.append("carbohydrate overlap chunks")
    if any("confounder" in section or "sers_cautions" in section for section in sections):
        section_text.append("EV/SERS confounder guidance")

    if not section_text:
        section_text = ["general biosample interpretation guidance"]

    unique_items = []
    for item in section_text:
        if item not in unique_items:
            unique_items.append(item)

    return f"Top support comes from {', '.join(unique_items[:3])}."


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_storage_config, resolve_storage_path

    storage_config = get_storage_config()
    processed_root = resolve_storage_path(storage_config.get("processed_data"))

    if processed_root is None:
        print("The storage config is missing processed_data.")
        return

    summary_path = processed_root / "shine_class_reference_matches" / "shine_class_consensus_summary.csv"
    interpreted_path = processed_root / "shine_class_reference_matches" / "shine_class_consensus_interpreted.csv"
    chunks_path = processed_root / "shine_class_reference_matches" / "shine_class_supporting_chunks_v2.csv"
    output_path = processed_root / "shine_class_reference_matches" / "shine_class_explanation_report_v2.csv"

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

        chunk_1 = get_supporting_row(chunk_group_df, 1)
        chunk_2 = get_supporting_row(chunk_group_df, 2)
        chunk_3 = get_supporting_row(chunk_group_df, 3)

        supporting_chunk_1 = "" if chunk_1 is None else str(chunk_1["chunk_text"])
        supporting_chunk_2 = "" if chunk_2 is None else str(chunk_2["chunk_text"])
        supporting_chunk_3 = "" if chunk_3 is None else str(chunk_3["chunk_text"])

        evidence_summary = build_evidence_summary(chunk_group_df)
        confidence_tier = choose_confidence_tier(chunk_group_df)

        explanation_text = (
            f"This class shows a {row_series['top_biochemical_class_1']}/"
            f"{row_series['top_biochemical_class_2']} analog pattern with strongest support in "
            f"{row_series['dominant_region_1']} and {row_series['dominant_region_2']} cm^-1. "
            f"The strongest semantic regions are {row_series['region_semantic_label_1']} and "
            f"{row_series['region_semantic_label_2']}. "
            f"{evidence_summary} "
        )
        if row_series.get("confounder_warnings"):
            explanation_text += (
                f"Important confounders include {row_series['confounder_warnings']}. "
            )
        explanation_text += (
            "This remains a class-level, analog interpretation and should not be treated as literal molecule identification."
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
                "evidence_summary": evidence_summary,
                "explanation_text": explanation_text,
                "confidence_tier": confidence_tier,
            }
        )

    report_df = pd.DataFrame(report_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, index=False)

    print(f"SHINE explanation report v2 written to: {output_path}")
    print(f"Rows written: {len(report_df)}")
    print(report_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
