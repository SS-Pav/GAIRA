import sys
from pathlib import Path

import pandas as pd


def get_supporting_row(group_df: pd.DataFrame, rank: int) -> pd.Series | None:
    """Return one selected v5 evidence row by rank."""
    match_df = group_df[group_df["chunk_rank"] == rank]
    if match_df.empty:
        return None
    return match_df.iloc[0]


def build_evidence_summary(row: pd.Series, group_df: pd.DataFrame) -> str:
    """Summarize the selected v5 evidence in cautious scientific language."""
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
    """Assign a conservative v5 confidence tier with softer criteria than v4."""
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


def build_explanation_text(row: pd.Series, evidence_summary: str) -> str:
    """Build the final deterministic v5 explanation."""
    explanation = (
        f"This class shows a {row['top_biochemical_class_1']}/{row['top_biochemical_class_2']} analog pattern "
        f"with strongest support in {row['region_semantic_label_1']} and {row['region_semantic_label_2']}. "
        f"{evidence_summary} "
    )
    if str(row.get("confounder_warnings", "")).strip():
        explanation += f"Important confounders include {row['confounder_warnings']}. "
    explanation += (
        "This remains a class-level, region-aware interpretation and should not be treated as literal molecule identification."
    )
    return explanation


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_storage_config, resolve_storage_path

    storage_config = get_storage_config()
    processed_root = resolve_storage_path(storage_config.get("processed_data"))

    if processed_root is None:
        print("The storage config is missing processed_data.")
        return

    base_path = processed_root / "shine_class_reference_matches"
    interpreted_path = base_path / "shine_class_consensus_interpreted.csv"
    chunks_path = base_path / "shine_class_supporting_chunks_v5.csv"
    output_path = base_path / "shine_class_explanation_report_v5.csv"

    for path in [interpreted_path, chunks_path]:
        if not path.exists():
            print(f"Required input file not found: {path}")
            return

    interpreted_df = pd.read_csv(interpreted_path)
    chunks_df = pd.read_csv(chunks_path)

    report_rows: list[dict] = []
    for row in interpreted_df.to_dict(orient="records"):
        row_series = pd.Series(row)
        group_df = chunks_df[
            (chunks_df["class_label"] == row_series["class_label"])
            & (chunks_df["subclass_label"] == row_series["subclass_label"])
        ].copy()

        chunk_1 = get_supporting_row(group_df, 1)
        chunk_2 = get_supporting_row(group_df, 2)
        chunk_3 = get_supporting_row(group_df, 3)

        supporting_chunk_1 = "" if chunk_1 is None else str(chunk_1["chunk_text"])
        supporting_chunk_2 = "" if chunk_2 is None else str(chunk_2["chunk_text"])
        supporting_chunk_3 = "" if chunk_3 is None else str(chunk_3["chunk_text"])
        supporting_role_1 = "" if chunk_1 is None else str(chunk_1["chunk_role"])
        supporting_role_2 = "" if chunk_2 is None else str(chunk_2["chunk_role"])
        supporting_role_3 = "" if chunk_3 is None else str(chunk_3["chunk_role"])

        evidence_summary = build_evidence_summary(row_series, group_df)
        confidence_tier, confidence_reason = choose_confidence(group_df)
        explanation_text = build_explanation_text(row_series, evidence_summary)

        report_rows.append(
            {
                "class_label": row_series["class_label"],
                "subclass_label": row_series["subclass_label"],
                "supporting_chunk_1": supporting_chunk_1,
                "supporting_chunk_2": supporting_chunk_2,
                "supporting_chunk_3": supporting_chunk_3,
                "supporting_role_1": supporting_role_1,
                "supporting_role_2": supporting_role_2,
                "supporting_role_3": supporting_role_3,
                "confidence_tier": confidence_tier,
                "confidence_reason": confidence_reason,
                "evidence_summary": evidence_summary,
                "explanation_text": explanation_text,
            }
        )

    report_df = pd.DataFrame(report_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, index=False)

    print(f"SHINE explanation report v5 written to: {output_path}")
    print(f"Rows written: {len(report_df)}")
    print(report_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
