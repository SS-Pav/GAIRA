import sys
from pathlib import Path

import pandas as pd


ROLE_PRIORITY = {
    "region_mechanistic": 3,
    "matrix_context": 2,
    "confounder_or_caution": 1,
    "mixed_context": 0,
}


def normalize_label(text: str) -> str:
    """Collapse a label into a lowercase alphanumeric token."""
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def semantic_fit(row: pd.Series) -> float:
    """Estimate compatibility between a chunk and the row's semantic regions."""
    tags_text = " ".join(
        [
            str(row["section"]).lower(),
            str(row["chunk_text"]).lower(),
            str(row["matched_terms"]).lower(),
        ]
    )
    score = 0.0

    label_1 = normalize_label(row["region_semantic_label_1"])
    label_2 = normalize_label(row["region_semantic_label_2"])
    if "amide" in label_1 and ("amide" in tags_text or "protein" in tags_text):
        score += 3.0
    if "amide" in label_2 and ("amide" in tags_text or "protein" in tags_text):
        score += 2.0
    if "chdeformation" in label_1 and ("ch" in tags_text or "lipid" in tags_text or "membrane" in tags_text):
        score += 3.0
    if "chdeformation" in label_2 and ("ch" in tags_text or "lipid" in tags_text):
        score += 2.0
    if "nucleicacid" in label_1 and ("nucleic" in tags_text or "phosphate" in tags_text or "base" in tags_text):
        score += 3.0
    if "nucleicacid" in label_2 and ("nucleic" in tags_text or "phosphate" in tags_text):
        score += 2.0
    if "carbohydrate" in label_1 and ("carbohydrate" in tags_text or "glycan" in tags_text or "saccharide" in tags_text):
        score += 2.5
    if "carbohydrate" in label_2 and ("carbohydrate" in tags_text or "glycan" in tags_text or "saccharide" in tags_text):
        score += 1.5
    if "lowwavenumber" in label_1 and row["chunk_role"] in {"matrix_context", "confounder_or_caution", "mixed_context"}:
        score += 1.5
    if "aromatic" in label_1 and ("aromatic" in tags_text or "amino" in tags_text):
        score += 2.5

    return score


def compute_global_reuse(pool_df: pd.DataFrame) -> pd.DataFrame:
    """Count how often each chunk appears in the v5 pool across SHINE rows."""
    reuse_df = (
        pool_df.groupby("chunk_id")[["class_label"]]
        .count()
        .rename(columns={"class_label": "global_reuse_count"})
        .reset_index()
    )
    return pool_df.merge(reuse_df, on="chunk_id", how="left")


def select_group_chunks(group_df: pd.DataFrame) -> pd.DataFrame:
    """Select up to three chunks with soft role diversity and lower harshness than v4."""
    selected_rows: list[dict] = []
    used_chunk_ids: set[str] = set()
    used_roles: list[str] = []
    used_region_signatures: set[str] = set()

    working_df = group_df.copy()
    while len(selected_rows) < 3 and not working_df.empty:
        scored_rows: list[dict] = []
        for row in working_df.to_dict(orient="records"):
            series = pd.Series(row)
            role = str(series["chunk_role"])
            semantic_score = semantic_fit(series)
            role_bonus = 1.0 if role not in used_roles else 0.0
            if role == "region_mechanistic" and semantic_score >= 4:
                role_bonus += 1.0

            region_signature = f"{series['region_semantic_label_1']}|{series['region_semantic_label_2']}|{series['section']}"
            duplication_penalty = 0.6 if role in used_roles else 0.0
            if role == "region_mechanistic" and region_signature in used_region_signatures:
                duplication_penalty += 0.8

            reuse_penalty_global = max(int(series["global_reuse_count"]) - 5, 0) * 0.5
            if role in {"matrix_context", "confounder_or_caution"}:
                reuse_penalty_global *= 0.7

            confounder_overweight_penalty = 0.0
            if role == "confounder_or_caution" and used_roles.count("confounder_or_caution") >= 1:
                confounder_overweight_penalty += 1.2
            if role == "confounder_or_caution" and not str(series["confounder_warnings"]).strip():
                confounder_overweight_penalty += 0.6

            final_score = (
                0.7 * float(series["total_score"])
                + semantic_score
                + role_bonus
                - duplication_penalty
                - reuse_penalty_global
                - confounder_overweight_penalty
            )

            scored_rows.append(
                {
                    **row,
                    "pool_score": float(series["total_score"]),
                    "semantic_compatibility_score": semantic_score,
                    "role_bonus": role_bonus,
                    "role_duplication_penalty": duplication_penalty,
                    "reuse_penalty_global": reuse_penalty_global,
                    "confounder_overweight_penalty": confounder_overweight_penalty,
                    "final_arbitration_score": final_score,
                    "region_signature": region_signature,
                }
            )

        scored_df = pd.DataFrame(scored_rows).sort_values(
            [
                "final_arbitration_score",
                "semantic_compatibility_score",
                "pool_score",
                "chunk_id",
            ],
            ascending=[False, False, False, True],
        )
        best_row = scored_df.iloc[0].to_dict()
        selected_rows.append(best_row)
        used_chunk_ids.add(str(best_row["chunk_id"]))
        used_roles.append(str(best_row["chunk_role"]))
        if best_row["chunk_role"] == "region_mechanistic":
            used_region_signatures.add(str(best_row["region_signature"]))

        working_df = working_df[~working_df["chunk_id"].isin(used_chunk_ids)].copy()

        # If the evidence is weak, do not force a third chunk.
        if len(selected_rows) >= 2 and best_row["final_arbitration_score"] < 4:
            break

    selected_df = pd.DataFrame(selected_rows)
    if selected_df.empty:
        return selected_df
    selected_df = selected_df.sort_values(
        ["final_arbitration_score", "semantic_compatibility_score", "pool_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    selected_df["chunk_rank"] = range(1, len(selected_df) + 1)
    return selected_df


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_storage_config, resolve_storage_path

    storage_config = get_storage_config()
    processed_root = resolve_storage_path(storage_config.get("processed_data"))

    if processed_root is None:
        print("The storage config is missing processed_data.")
        return

    input_path = processed_root / "shine_class_reference_matches" / "shine_class_supporting_chunks_v5_pool.csv"
    output_path = processed_root / "shine_class_reference_matches" / "shine_class_supporting_chunks_v5.csv"

    if not input_path.exists():
        print(f"Required v5 pool file not found: {input_path}")
        return

    pool_df = pd.read_csv(input_path)
    pool_df = compute_global_reuse(pool_df)

    selected_groups: list[pd.DataFrame] = []
    for _, group_df in pool_df.groupby(["class_label", "subclass_label"], sort=True):
        selected_df = select_group_chunks(group_df)
        if not selected_df.empty:
            selected_groups.append(selected_df)

    if not selected_groups:
        print("No v5 supporting chunks were selected.")
        return

    final_df = pd.concat(selected_groups, ignore_index=True)
    keep_columns = [
        "class_label",
        "subclass_label",
        "chunk_rank",
        "chunk_id",
        "source_id",
        "section",
        "chunk_role",
        "chunk_text",
        "pool_score",
        "semantic_compatibility_score",
        "role_bonus",
        "role_duplication_penalty",
        "reuse_penalty_global",
        "confounder_overweight_penalty",
        "final_arbitration_score",
        "matched_terms",
        "top_biochemical_class_1",
        "top_biochemical_class_2",
        "dominant_region_1",
        "dominant_region_2",
        "region_semantic_label_1",
        "region_semantic_label_2",
        "confounder_warnings",
    ]
    final_df = final_df[keep_columns].sort_values(
        ["class_label", "subclass_label", "chunk_rank"],
        ascending=[True, True, True],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print(f"SHINE supporting chunks v5 written to: {output_path}")
    print(f"Rows written: {len(final_df)}")
    print(final_df.head(16).to_string(index=False))


if __name__ == "__main__":
    main()
