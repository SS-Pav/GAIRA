import sys
from pathlib import Path

import pandas as pd


def get_supporting_row(group_df: pd.DataFrame, rank: int) -> pd.Series | None:
    """Return one supporting chunk row by rank."""
    match_df = group_df[group_df["chunk_rank"] == rank]
    if match_df.empty:
        return None
    return match_df.iloc[0]


def build_evidence_summary(group_df: pd.DataFrame, row_series: pd.Series) -> tuple[str, str]:
    """Create a more discriminative evidence summary and specificity label."""
    if group_df.empty:
        return "Support is sparse and mostly generic.", "generic"

    avg_region_specificity = float(group_df["region_specificity_score"].mean())
    avg_semantic = float(group_df["semantic_region_score"].mean())
    avg_generic_penalty = float(group_df["generic_penalty"].mean())
    sections = group_df["section"].astype(str).tolist()

    descriptors: list[str] = []
    if "protein_regions" in sections or "amide_regions" in sections:
        descriptors.append("protein-rich / amide-weighted evidence")
    if "lipid_regions" in sections or "ch_regions" in sections:
        descriptors.append("lipid-overlap / CH-deformation evidence")
    if "nucleic_acid_regions" in sections:
        descriptors.append("nucleic-acid-like region support")
    if "carbohydrate_regions" in sections:
        descriptors.append("carbohydrate-overlap support")
    if "confounders" in sections or "sers_cautions" in sections:
        descriptors.append("confounder-heavy guidance")
    if not descriptors:
        descriptors.append("mixed biosample interpretation guidance")

    if avg_region_specificity >= 2 and avg_semantic >= 5:
        evidence_specificity = "region-specific"
    elif avg_generic_penalty >= 1.5:
        evidence_specificity = "generic"
    else:
        evidence_specificity = "mixed"

    summary = (
        f"Dominant evidence reflects {', '.join(descriptors[:3])} across "
        f"{row_series['region_semantic_label_1']} and {row_series['region_semantic_label_2']}."
    )
    return summary, evidence_specificity


def choose_confidence(group_df: pd.DataFrame) -> tuple[str, str]:
    """Assign a conservative confidence tier and explain why."""
    if group_df.empty:
        return "low", "low because no supporting chunks were retrieved"

    non_empty_rows = group_df[group_df["chunk_text"].fillna("").str.strip().ne("")]
    if len(non_empty_rows) < 2:
        return "low", "low because support is sparse and not diverse across sections"

    distinct_sections = int(non_empty_rows["section"].nunique())
    avg_semantic = float(non_empty_rows["semantic_region_score"].mean())
    avg_region_specificity = float(non_empty_rows["region_specificity_score"].mean())
    avg_reuse_penalty = float(non_empty_rows["reuse_penalty"].mean())
    avg_generic_penalty = float(non_empty_rows["generic_penalty"].mean())
    confounder_only = bool(
        non_empty_rows["section"].isin(["confounders", "sers_cautions"]).all()
    )

    if (
        distinct_sections >= 2
        and avg_semantic >= 4.5
        and avg_region_specificity >= 1.0
        and avg_reuse_penalty <= 1.0
        and avg_generic_penalty <= 1.0
        and not confounder_only
    ):
        return (
            "moderate",
            "moderate because region-specific chunks from multiple sections agree across two semantic regions with limited reuse penalty",
        )

    reason_parts = []
    if distinct_sections < 2:
        reason_parts.append("limited section diversity")
    if avg_semantic < 4.5:
        reason_parts.append("weak semantic-region alignment")
    if avg_region_specificity < 1.0:
        reason_parts.append("limited region specificity")
    if avg_reuse_penalty > 1.0:
        reason_parts.append("repeated chunk reuse")
    if avg_generic_penalty > 1.0:
        reason_parts.append("support is broad and generic")
    if confounder_only:
        reason_parts.append("support is dominated by confounder guidance")

    return "low", "low because " + ", ".join(reason_parts)


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
    chunks_path = processed_root / "shine_class_reference_matches" / "shine_class_supporting_chunks_v3.csv"
    output_path = processed_root / "shine_class_reference_matches" / "shine_class_explanation_report_v3.csv"

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

        evidence_summary, evidence_specificity = build_evidence_summary(chunk_group_df, row_series)
        confidence_tier, confidence_reason = choose_confidence(chunk_group_df)

        explanation_text = (
            f"This class shows a {row_series['top_biochemical_class_1']}/"
            f"{row_series['top_biochemical_class_2']} analog pattern with strongest support in "
            f"{row_series['dominant_region_1']} and {row_series['dominant_region_2']} cm^-1. "
            f"The strongest semantic regions are {row_series['region_semantic_label_1']} and "
            f"{row_series['region_semantic_label_2']}. {evidence_summary} "
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
                "evidence_specificity": evidence_specificity,
                "confidence_tier": confidence_tier,
                "confidence_reason": confidence_reason,
            }
        )

    report_df = pd.DataFrame(report_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, index=False)

    print(f"SHINE explanation report v3 written to: {output_path}")
    print(f"Rows written: {len(report_df)}")
    print(report_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
