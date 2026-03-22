import duckdb
import pandas as pd
import sys
from pathlib import Path


REGION_TO_CONTEXT_COLUMN = {
    "450-700": "region_caution_450_700",
    "700-900": "region_caution_700_900",
    "900-1100": "region_caution_900_1100",
    "1100-1300": "region_caution_1100_1300",
    "1300-1500": "region_caution_1300_1500",
    "1500-1700": "region_caution_1500_1700",
}


def get_supporting_row(group_df: pd.DataFrame, rank: int) -> pd.Series | None:
    """Return one selected v5 evidence row by rank."""
    match_df = group_df[group_df["chunk_rank"] == rank]
    if match_df.empty:
        return None
    return match_df.iloc[0]


def build_evidence_summary(row: pd.Series, group_df: pd.DataFrame) -> str:
    """Summarize selected v5 evidence with the existing deterministic logic."""
    if group_df.empty:
        return "Evidence is sparse and only supports a broad analog-level interpretation."

    sections = set(group_df["section"].astype(str))
    roles = set(group_df["chunk_role"].astype(str))
    descriptors: list[str] = []
    if {"protein_regions", "amide_regions"} & sections:
        descriptors.append("protein-rich / amide-weighted support")
    if {"lipid_regions", "ch_regions"} & sections:
        descriptors.append("lipid-overlap / CH-deformation support")
    if "nucleic_acid_regions" in sections:
        descriptors.append("nucleic-acid-like overlap support")
    if "carbohydrate_regions" in sections:
        descriptors.append("carbohydrate-overlap support")
    if "confounder_or_caution" in roles:
        descriptors.append("explicit confounder guidance")
    if not descriptors:
        descriptors.append("mixed biosample interpretation guidance")

    if "region_mechanistic" in roles and len(roles) >= 2:
        tone = "better balanced"
    elif roles == {"confounder_or_caution"}:
        tone = "confounder-heavy"
    else:
        tone = "mixed"

    return (
        f"Dominant evidence is {tone} and emphasizes {', '.join(descriptors[:3])} across "
        f"{row['region_semantic_label_1']} and {row['region_semantic_label_2']}."
    )


def choose_confidence(group_df: pd.DataFrame) -> tuple[str, str]:
    """Reuse the v5 confidence logic so only context wording changes in v6."""
    if group_df.empty:
        return "low", "low because no supporting evidence was selected"

    roles = group_df["chunk_role"].astype(str).tolist()
    distinct_roles = len(set(roles))
    top_row = group_df.sort_values("chunk_rank").iloc[0]
    top_is_confounder = top_row["chunk_role"] == "confounder_or_caution"
    confounder_count = sum(role == "confounder_or_caution" for role in roles)
    mech_df = group_df[group_df["chunk_role"] == "region_mechanistic"]
    strong_mech_count = int((mech_df["semantic_compatibility_score"] >= 3.5).sum())
    distinct_mech_sections = int(mech_df["section"].nunique()) if not mech_df.empty else 0
    avg_semantic = float(group_df["semantic_compatibility_score"].mean())
    avg_reuse = float(group_df["reuse_penalty_global"].mean())

    if (
        ((distinct_roles >= 2) or (strong_mech_count >= 2 and distinct_mech_sections >= 2))
        and not top_is_confounder
        and confounder_count <= 1
        and avg_semantic >= 2.5
        and avg_reuse <= 1.8
    ):
        return (
            "moderate",
            "moderate because evidence spans multiple roles or semantic-region-compatible mechanistic chunks, while confounder dominance and reuse remain limited",
        )

    reasons: list[str] = []
    if distinct_roles < 2 and not (strong_mech_count >= 2 and distinct_mech_sections >= 2):
        reasons.append("evidence lacks role diversity and region-distinct mechanistic support")
    if top_is_confounder:
        reasons.append("top evidence is confounder-heavy")
    if confounder_count > 1:
        reasons.append("confounder-oriented chunks are overrepresented")
    if avg_semantic < 2.5:
        reasons.append("semantic compatibility is limited")
    if avg_reuse > 1.8:
        reasons.append("reused generic chunks still reduce specificity")
    return "low", "low because " + ", ".join(reasons)


def map_region_caution(context_row: pd.Series, region_bucket: str) -> str:
    """Map one dominant region bucket to the matching dataset-context caution field."""
    column_name = REGION_TO_CONTEXT_COLUMN.get(str(region_bucket))
    if column_name is None:
        return ""
    return str(context_row.get(column_name, "") or "")


def build_explanation_text(row: pd.Series) -> str:
    """Build a cautious context-aware SHINE explanation."""
    explanation = (
        f"This class shows a {row['top_biochemical_class_1']}/{row['top_biochemical_class_2']} analog pattern "
        f"with strongest semantic support in {row['region_semantic_label_1']} and {row['region_semantic_label_2']}. "
        f"{row['evidence_summary']} "
        f"Under {row['context_modality']} {row['context_sample_type']} conditions, this dataset should be read as "
        f"{row['context_enhancement_mode']} and substrate-conditioned rather than composition-only. "
    )
    if str(row["context_region_caution_1"]).strip():
        explanation += f"For {row['dominant_region_1']} cm^-1, note: {row['context_region_caution_1']} "
    if str(row["context_region_caution_2"]).strip():
        explanation += f"For {row['dominant_region_2']} cm^-1, note: {row['context_region_caution_2']} "
    if str(row["confounder_warnings"]).strip():
        explanation += f"Important confounders include {row['confounder_warnings']}. "
    if str(row["context_interpretation_note"]).strip():
        explanation += f"{row['context_interpretation_note']} "
    explanation += (
        f"{row['context_do_not_overclaim_note']} "
        "This remains a class-level, region-aware interpretation and should not be treated as literal molecule identification."
    )
    return explanation


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_database_path, get_storage_config, resolve_storage_path

    storage_config = get_storage_config()
    processed_root = resolve_storage_path(storage_config.get("processed_data"))
    db_path = get_database_path()

    if processed_root is None:
        print("The storage config is missing processed_data.")
        return

    base_path = processed_root / "shine_class_reference_matches"
    summary_path = base_path / "shine_class_consensus_summary.csv"
    interpreted_path = base_path / "shine_class_consensus_interpreted.csv"
    chunks_path = base_path / "shine_class_supporting_chunks_v5.csv"
    output_path = base_path / "shine_class_explanation_report_v6.csv"

    for path in [summary_path, interpreted_path, chunks_path]:
        if not path.exists():
            print(f"Required input file not found: {path}")
            return

    summary_df = pd.read_csv(summary_path)
    interpreted_df = pd.read_csv(interpreted_path)
    chunks_df = pd.read_csv(chunks_path)

    with duckdb.connect(str(db_path), read_only=True) as connection:
        context_df = connection.execute(
            """
            SELECT *
            FROM dataset_context
            WHERE target_dataset_id = 'shine_ev_sers'
            ORDER BY context_id
            LIMIT 1
            """
        ).fetchdf()

    if context_df.empty:
        print("No dataset_context row found for target_dataset_id='shine_ev_sers'.")
        return

    context_row = context_df.iloc[0]
    merged_df = summary_df.merge(
        interpreted_df,
        on=["class_label", "subclass_label", "top_biochemical_class_1", "top_biochemical_class_2", "dominant_region_1", "dominant_region_2"],
        how="left",
    )

    report_rows: list[dict] = []
    for row in merged_df.to_dict(orient="records"):
        row_series = pd.Series(row)
        group_df = chunks_df[
            (chunks_df["class_label"] == row_series["class_label"])
            & (chunks_df["subclass_label"] == row_series["subclass_label"])
        ].copy()

        chunk_1 = get_supporting_row(group_df, 1)
        chunk_2 = get_supporting_row(group_df, 2)
        chunk_3 = get_supporting_row(group_df, 3)

        evidence_summary = build_evidence_summary(row_series, group_df)
        confidence_tier, confidence_reason = choose_confidence(group_df)

        report_row = {
            "class_label": row_series["class_label"],
            "subclass_label": row_series["subclass_label"],
            "n_spectra": row_series.get("n_spectra"),
            "top_biochemical_class_1": row_series["top_biochemical_class_1"],
            "top_biochemical_class_2": row_series["top_biochemical_class_2"],
            "dominant_region_1": row_series["dominant_region_1"],
            "dominant_region_2": row_series["dominant_region_2"],
            "region_semantic_label_1": row_series.get("region_semantic_label_1", ""),
            "region_semantic_label_2": row_series.get("region_semantic_label_2", ""),
            "supporting_chunk_1": "" if chunk_1 is None else str(chunk_1["chunk_text"]),
            "supporting_chunk_2": "" if chunk_2 is None else str(chunk_2["chunk_text"]),
            "supporting_chunk_3": "" if chunk_3 is None else str(chunk_3["chunk_text"]),
            "supporting_role_1": "" if chunk_1 is None else str(chunk_1["chunk_role"]),
            "supporting_role_2": "" if chunk_2 is None else str(chunk_2["chunk_role"]),
            "supporting_role_3": "" if chunk_3 is None else str(chunk_3["chunk_role"]),
            "confidence_tier": confidence_tier,
            "confidence_reason": confidence_reason,
            "evidence_summary": evidence_summary,
            "context_modality": str(context_row["modality"]),
            "context_sample_type": str(context_row["sample_type"]),
            "context_substrate_type": str(context_row["substrate_type"]),
            "context_enhancement_mode": str(context_row["enhancement_mode"]),
            "context_known_biases": str(context_row["known_biases"]),
            "context_region_caution_1": map_region_caution(context_row, str(row_series["dominant_region_1"])),
            "context_region_caution_2": map_region_caution(context_row, str(row_series["dominant_region_2"])),
            "context_interpretation_note": str(context_row["interpretation_note"]),
            "context_do_not_overclaim_note": str(context_row["do_not_overclaim_note"]),
            "confounder_warnings": str(row_series.get("confounder_warnings", "")),
        }
        report_row["explanation_text"] = build_explanation_text(pd.Series(report_row))
        report_rows.append(report_row)

    report_df = pd.DataFrame(report_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, index=False)

    print(f"SHINE explanation report v6 written to: {output_path}")
    print(f"Rows written: {len(report_df)}")
    print(report_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
