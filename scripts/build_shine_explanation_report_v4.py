import sys
from pathlib import Path

import pandas as pd


def get_supporting_row(group_df: pd.DataFrame, rank: int) -> pd.Series | None:
    """Return one supporting evidence row by its v4 rank."""
    match_df = group_df[group_df["chunk_rank"] == rank]
    if match_df.empty:
        return None
    return match_df.iloc[0]


def build_evidence_summary(row: pd.Series, group_df: pd.DataFrame) -> str:
    """Summarize the evidence pattern in cautious, class-level language."""
    if group_df.empty:
        return "Evidence is sparse and does not support a specific biochemical analog summary."

    roles = set(group_df["chunk_role"].astype(str))
    sections = set(group_df["section"].astype(str))
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

    coherence = "mixed"
    if "region_mechanistic" in roles and len(roles) >= 2:
        coherence = "coherent"
    if roles == {"confounder_or_caution"}:
        coherence = "confounder-heavy"

    return (
        f"Dominant evidence is {coherence} and emphasizes {', '.join(descriptors[:3])} across "
        f"{row['region_semantic_label_1']} and {row['region_semantic_label_2']}."
    )


def choose_confidence(group_df: pd.DataFrame) -> tuple[str, str]:
    """Assign a conservative v4 confidence tier from role balance and compatibility."""
    if group_df.empty:
        return "low", "low because no supporting evidence was selected"

    roles = group_df["chunk_role"].astype(str).tolist()
    distinct_roles = len(set(roles))
    top_row = group_df.sort_values("chunk_rank").iloc[0]
    top_is_confounder = top_row["chunk_role"] == "confounder_or_caution"
    confounder_count = sum(role == "confounder_or_caution" for role in roles)
    has_strong_region = bool(
        (
            (group_df["chunk_role"] == "region_mechanistic")
            & (group_df["semantic_compatibility_score"] >= 4)
        ).any()
    )
    avg_reuse = float(group_df["reuse_penalty_global"].mean())
    avg_conflict = float(group_df["role_conflict_penalty"].mean())
    avg_confounder_penalty = float(group_df["confounder_overweight_penalty"].mean())

    if (
        distinct_roles >= 2
        and has_strong_region
        and not top_is_confounder
        and confounder_count <= 1
        and avg_reuse <= 1.5
        and avg_conflict <= 2.5
        and avg_confounder_penalty < 1.0
    ):
        return (
            "moderate",
            "moderate because evidence roles are diverse, region compatibility is strong, and confounder-heavy or highly reused chunks do not dominate",
        )

    reasons: list[str] = []
    if distinct_roles < 2:
        reasons.append("evidence roles are not diverse")
    if not has_strong_region:
        reasons.append("region-mechanistic compatibility is limited")
    if top_is_confounder:
        reasons.append("top evidence is confounder-heavy")
    if confounder_count > 1:
        reasons.append("multiple confounder-role chunks were needed")
    if avg_reuse > 1.5:
        reasons.append("repeated generic chunks reduced confidence")
    if avg_conflict > 2.5:
        reasons.append("semantic mismatches weakened the evidence")
    if avg_confounder_penalty >= 1.0:
        reasons.append("confounder weighting forced extra caution")

    return "low", "low because " + ", ".join(reasons)


def build_explanation_text(row: pd.Series, evidence_summary: str) -> str:
    """Build a deterministic, cautious v4 explanation."""
    explanation = (
        f"This class shows a {row['top_biochemical_class_1']}/{row['top_biochemical_class_2']} analog pattern, "
        f"with strongest semantic support in {row['region_semantic_label_1']} and {row['region_semantic_label_2']}. "
        f"{evidence_summary} "
    )
    if str(row.get("confounder_warnings", "")).strip():
        explanation += f"Important confounders include {row['confounder_warnings']}. "
    explanation += (
        "These supporting chunks should be interpreted as class-level and matrix-aware evidence, "
        "not as literal molecule identification."
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
    chunks_path = base_path / "shine_class_supporting_chunks_v4.csv"
    output_path = base_path / "shine_class_explanation_report_v4.csv"

    for path in [interpreted_path, chunks_path]:
        if not path.exists():
            print(f"Required input file not found: {path}")
            return

    interpreted_df = pd.read_csv(interpreted_path)
    chunks_df = pd.read_csv(chunks_path)

    report_rows: list[dict] = []
    for row in interpreted_df.to_dict(orient="records"):
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

        supporting_role_1 = "" if chunk_1 is None else str(chunk_1["chunk_role"])
        supporting_role_2 = "" if chunk_2 is None else str(chunk_2["chunk_role"])
        supporting_role_3 = "" if chunk_3 is None else str(chunk_3["chunk_role"])

        evidence_summary = build_evidence_summary(row_series, chunk_group_df)
        confidence_tier, confidence_reason = choose_confidence(chunk_group_df)
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
                "evidence_summary": evidence_summary,
                "confidence_tier": confidence_tier,
                "confidence_reason": confidence_reason,
                "explanation_text": explanation_text,
            }
        )

    report_df = pd.DataFrame(report_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, index=False)

    print(f"SHINE explanation report v4 written to: {output_path}")
    print(f"Rows written: {len(report_df)}")
    print(report_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
