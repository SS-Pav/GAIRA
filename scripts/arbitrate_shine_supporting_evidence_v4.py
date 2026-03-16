import json
import sys
from pathlib import Path

import duckdb
import pandas as pd


ROLE_PRIORITY = {
    "region_mechanistic": 3,
    "matrix_context": 2,
    "confounder_or_caution": 1,
    "mixed_context": 0,
}

ROLE_KEYWORDS = {
    "region_mechanistic": {
        "protein_regions",
        "amide_regions",
        "lipid_regions",
        "nucleic_acid_regions",
        "carbohydrate_regions",
        "ch_regions",
        "aromatic_regions",
    },
    "matrix_context": {
        "ev_interpretation",
        "serum_interpretation",
        "biofluid_interpretation",
    },
    "confounder_or_caution": {
        "confounders",
        "sers_cautions",
    },
    "mixed_context": {
        "mixed_signature",
    },
}

SEMANTIC_RULES = [
    {
        "match_terms": {"amide i protein rich region"},
        "preferred": {"protein_regions", "amide_regions"},
        "allowed_overlap": {"lipid_regions", "mixed_signature", "biofluid_interpretation"},
        "mismatch": {"nucleic_acid_regions", "carbohydrate_regions"},
    },
    {
        "match_terms": {"amide iii carbohydrate lipid overlap region", "amide iii and unsaturated lipid region"},
        "preferred": {"protein_regions", "amide_regions", "lipid_regions", "carbohydrate_regions"},
        "allowed_overlap": {"ch_regions", "mixed_signature", "biofluid_interpretation"},
        "mismatch": {"nucleic_acid_regions"},
    },
    {
        "match_terms": {"ch deformation lipid protein overlap region", "broad ch deformation biosample region"},
        "preferred": {"lipid_regions", "ch_regions", "ev_interpretation"},
        "allowed_overlap": {"protein_regions", "mixed_signature", "biofluid_interpretation"},
        "mismatch": {"nucleic_acid_regions"},
    },
    {
        "match_terms": {"nucleic acid phosphate and base region", "choline and nucleic acid ring mode region"},
        "preferred": {"nucleic_acid_regions", "aromatic_regions"},
        "allowed_overlap": {"biofluid_interpretation", "mixed_signature", "carbohydrate_regions"},
        "mismatch": {"lipid_regions", "ch_regions"},
    },
    {
        "match_terms": {"aromatic phosphate carbohydrate overlap region"},
        "preferred": {"aromatic_regions", "carbohydrate_regions", "nucleic_acid_regions"},
        "allowed_overlap": {"biofluid_interpretation", "mixed_signature", "protein_regions"},
        "mismatch": set(),
    },
    {
        "match_terms": {"low wavenumber mixed biosample region"},
        "preferred": {"mixed_signature", "biofluid_interpretation"},
        "allowed_overlap": {"protein_regions", "nucleic_acid_regions", "lipid_regions"},
        "mismatch": set(),
    },
    {
        "match_terms": {"aromatic amino acid and base overlap region"},
        "preferred": {"aromatic_regions", "protein_regions", "nucleic_acid_regions"},
        "allowed_overlap": {"biofluid_interpretation", "mixed_signature"},
        "mismatch": {"lipid_regions"},
    },
    {
        "match_terms": {"high wavenumber carbonyl associated tail region"},
        "preferred": {"lipid_regions", "amide_regions"},
        "allowed_overlap": {"biofluid_interpretation", "mixed_signature"},
        "mismatch": {"nucleic_acid_regions"},
    },
]

CLASS_RULES = {
    "proteins": {"protein_regions", "amide_regions", "biofluid_interpretation"},
    "aminoacids": {"aromatic_regions", "protein_regions", "amide_regions"},
    "lipidsfattyacids": {"lipid_regions", "ch_regions", "ev_interpretation"},
    "lipidshormones": {"lipid_regions", "ch_regions", "sers_cautions"},
    "nucleicacids": {"nucleic_acid_regions", "aromatic_regions"},
    "saccharidesmonosaccharides": {"carbohydrate_regions", "biofluid_interpretation"},
    "saccharidespolysaccharides": {"carbohydrate_regions", "biofluid_interpretation"},
}

GENERIC_SECTIONS = {"biofluid_interpretation", "mixed_signature"}
CONF_SECTIONS = {"confounders", "sers_cautions"}
MATRIX_SECTIONS = {"ev_interpretation", "serum_interpretation", "biofluid_interpretation"}


def normalize_label(text: str) -> str:
    """Collapse a label into a retrieval-friendly token."""
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def parse_tags(metadata_json: str) -> set[str]:
    """Read metadata tags from a knowledge chunk row."""
    try:
        payload = json.loads(str(metadata_json))
    except Exception:
        return set()

    tags = payload.get("tags", [])
    if not isinstance(tags, list):
        return set()
    return {str(tag).strip().lower() for tag in tags if str(tag).strip()}


def infer_role(section: str, tags: set[str], chunk_text: str) -> str:
    """Assign one primary evidence role to a chunk."""
    section_lower = str(section).strip().lower()
    lowered_text = str(chunk_text).lower()

    if section_lower in ROLE_KEYWORDS["confounder_or_caution"] or tags & ROLE_KEYWORDS["confounder_or_caution"]:
        return "confounder_or_caution"
    if section_lower in ROLE_KEYWORDS["matrix_context"] or tags & ROLE_KEYWORDS["matrix_context"]:
        return "matrix_context"
    if section_lower in ROLE_KEYWORDS["region_mechanistic"] or tags & ROLE_KEYWORDS["region_mechanistic"]:
        return "region_mechanistic"
    if section_lower in ROLE_KEYWORDS["mixed_context"] or tags & ROLE_KEYWORDS["mixed_context"]:
        return "mixed_context"
    if "confound" in lowered_text or "caution" in lowered_text:
        return "confounder_or_caution"
    if "vesicle" in lowered_text or "biofluid" in lowered_text or "serum" in lowered_text:
        return "matrix_context"
    return "mixed_context"


def choose_semantic_rule(region_label: str) -> dict | None:
    """Pick the closest semantic rule for one explicit region label."""
    normalized = normalize_label(region_label)
    for rule in SEMANTIC_RULES:
        if normalized in {normalize_label(term) for term in rule["match_terms"]}:
            return rule
    return None


def compute_semantic_compatibility(row: pd.Series) -> tuple[float, float]:
    """Reward chunks whose tags align with the row's explicit semantic regions."""
    tags = row["metadata_tags"]
    role = row["chunk_role"]
    score = 0.0
    penalty = 0.0

    for region_label, weight in [
        (row["region_semantic_label_1"], 3.0),
        (row["region_semantic_label_2"], 2.0),
    ]:
        rule = choose_semantic_rule(str(region_label))
        if rule is None:
            continue
        if tags & rule["preferred"]:
            score += weight
        elif tags & rule["allowed_overlap"]:
            score += weight * 0.5
        elif role != "confounder_or_caution" and rule["mismatch"] and tags & rule["mismatch"]:
            penalty += 2.0

    return score, penalty


def compute_class_compatibility(row: pd.Series) -> tuple[float, float]:
    """Reward chunks aligned to the two interpreted biochemical classes."""
    tags = row["metadata_tags"]
    role = row["chunk_role"]
    score = 0.0
    penalty = 0.0

    for class_label, weight in [
        (row["top_biochemical_class_1"], 3.0),
        (row["top_biochemical_class_2"], 2.0),
    ]:
        class_key = normalize_label(class_label)
        preferred_tags = CLASS_RULES.get(class_key, set())
        if not preferred_tags:
            continue
        if tags & preferred_tags:
            score += weight
        elif role == "confounder_or_caution":
            score += 0.5
        elif role == "matrix_context":
            score += 0.5
        else:
            penalty += 1.5

    return score, penalty


def compute_role_bonus(row: pd.Series, semantic_score: float) -> tuple[float, float]:
    """Reward chunks whose role fits the row without overusing confounders."""
    role = row["chunk_role"]
    score = float(ROLE_PRIORITY.get(role, 0))
    penalty = 0.0

    if role == "region_mechanistic" and semantic_score >= 3:
        score += 1.5
    if role == "matrix_context" and (
        "ev" in str(row["knowledge_supported_groups"]).lower()
        or "extracellular" in str(row["possible_biomarker_contexts"]).lower()
    ):
        score += 1.0
    if role == "confounder_or_caution" and not str(row["confounder_warnings"]).strip():
        penalty += 1.0

    return score, penalty


def generic_penalty(row: pd.Series) -> float:
    """Penalize generic chunks that tend to recur everywhere."""
    tags = row["metadata_tags"]
    penalty = 0.0
    if row["section"] in GENERIC_SECTIONS:
        penalty += 1.0
    if "region_" not in " ".join(sorted(tags)):
        penalty += 0.5
    if len(tags) <= 2:
        penalty += 0.5
    return penalty


def build_candidate_pool(
    interpreted_df: pd.DataFrame,
    v3_df: pd.DataFrame,
    chunk_meta_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join interpreted SHINE rows to the v3 candidate pool and chunk metadata."""
    merged_df = v3_df.merge(
        interpreted_df,
        on=["class_label", "subclass_label"],
        how="left",
    ).merge(
        chunk_meta_df,
        on=["chunk_id", "source_id", "section", "chunk_text"],
        how="left",
    )

    merged_df["metadata_tags"] = merged_df["metadata_json"].apply(parse_tags)
    merged_df["chunk_role"] = merged_df.apply(
        lambda row: infer_role(row["section"], row["metadata_tags"], row["chunk_text"]),
        axis=1,
    )
    return merged_df


def score_candidates(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """Compute transparent v4 arbitration scores for each v3 candidate row."""
    if candidate_df.empty:
        return candidate_df

    reuse_counts = (
        candidate_df.groupby("chunk_id")[["class_label"]]
        .count()
        .rename(columns={"class_label": "global_reuse_count"})
        .reset_index()
    )
    candidate_df = candidate_df.merge(reuse_counts, on="chunk_id", how="left")
    candidate_df["global_reuse_count"] = candidate_df["global_reuse_count"].fillna(0).astype(int)

    semantic_scores: list[float] = []
    role_conflict_penalties: list[float] = []
    class_scores: list[float] = []
    class_penalties: list[float] = []
    role_bonuses: list[float] = []
    role_penalties: list[float] = []
    conf_overweight: list[float] = []
    reuse_penalties: list[float] = []
    generic_penalties: list[float] = []

    for row in candidate_df.to_dict(orient="records"):
        series = pd.Series(row)
        semantic_score, semantic_penalty = compute_semantic_compatibility(series)
        class_score, class_penalty = compute_class_compatibility(series)
        role_bonus, role_penalty = compute_role_bonus(series, semantic_score)
        reuse_penalty = max(int(series["global_reuse_count"]) - 2, 0) * 1.2
        if series["section"] in GENERIC_SECTIONS:
            reuse_penalty += max(int(series["global_reuse_count"]) - 1, 0) * 0.5
        generic = generic_penalty(series)
        conf_penalty = 1.5 if series["chunk_role"] == "confounder_or_caution" and series["semantic_region_score"] < 3 else 0.0

        semantic_scores.append(semantic_score)
        role_conflict_penalties.append(semantic_penalty + role_penalty)
        class_scores.append(class_score)
        class_penalties.append(class_penalty)
        role_bonuses.append(role_bonus)
        conf_overweight.append(conf_penalty)
        reuse_penalties.append(reuse_penalty)
        generic_penalties.append(generic)

    candidate_df["semantic_compatibility_score"] = semantic_scores
    candidate_df["class_compatibility_score"] = class_scores
    candidate_df["role_bonus"] = role_bonuses
    candidate_df["role_conflict_penalty"] = [a + b for a, b in zip(role_conflict_penalties, class_penalties)]
    candidate_df["reuse_penalty_global"] = reuse_penalties
    candidate_df["confounder_overweight_penalty"] = conf_overweight
    candidate_df["generic_penalty_v4"] = generic_penalties
    candidate_df["v3_total_score"] = candidate_df["total_score"]
    candidate_df["final_arbitration_score"] = (
        0.45 * candidate_df["v3_total_score"]
        + candidate_df["semantic_compatibility_score"]
        + candidate_df["class_compatibility_score"]
        + candidate_df["role_bonus"]
        - candidate_df["role_conflict_penalty"]
        - candidate_df["reuse_penalty_global"]
        - candidate_df["confounder_overweight_penalty"]
        - candidate_df["generic_penalty_v4"]
    )
    return candidate_df


def select_role_aware_chunks(group_df: pd.DataFrame) -> pd.DataFrame:
    """Select up to three final chunks with explicit role-aware diversity."""
    if group_df.empty:
        return group_df

    selected_rows: list[dict] = []
    used_chunk_ids: set[str] = set()
    confounder_selected = 0

    role_order = ["region_mechanistic", "matrix_context", "confounder_or_caution"]
    for role in role_order:
        role_df = group_df[group_df["chunk_role"] == role].copy()
        if confounder_selected >= 1 and role == "confounder_or_caution":
            role_df["final_arbitration_score"] = role_df["final_arbitration_score"] - 2.5
        role_df = role_df[~role_df["chunk_id"].isin(used_chunk_ids)]
        if role_df.empty:
            continue
        best_row = role_df.sort_values(
            [
                "final_arbitration_score",
                "semantic_compatibility_score",
                "class_compatibility_score",
                "v3_total_score",
                "chunk_id",
            ],
            ascending=[False, False, False, False, True],
        ).iloc[0]
        selected_rows.append(best_row.to_dict())
        used_chunk_ids.add(str(best_row["chunk_id"]))
        if best_row["chunk_role"] == "confounder_or_caution":
            confounder_selected += 1

    remaining_df = group_df[~group_df["chunk_id"].isin(used_chunk_ids)].copy()
    if confounder_selected >= 1:
        conf_mask = remaining_df["chunk_role"] == "confounder_or_caution"
        remaining_df.loc[conf_mask, "final_arbitration_score"] = (
            remaining_df.loc[conf_mask, "final_arbitration_score"] - 2.5
        )

    while len(selected_rows) < 3 and not remaining_df.empty:
        best_row = remaining_df.sort_values(
            [
                "final_arbitration_score",
                "semantic_compatibility_score",
                "role_bonus",
                "v3_total_score",
                "chunk_id",
            ],
            ascending=[False, False, False, False, True],
        ).iloc[0]
        selected_rows.append(best_row.to_dict())
        remaining_df = remaining_df[remaining_df["chunk_id"] != best_row["chunk_id"]]
        if best_row["chunk_role"] == "confounder_or_caution":
            conf_mask = remaining_df["chunk_role"] == "confounder_or_caution"
            remaining_df.loc[conf_mask, "final_arbitration_score"] = (
                remaining_df.loc[conf_mask, "final_arbitration_score"] - 2.5
            )

    selected_df = pd.DataFrame(selected_rows)
    if selected_df.empty:
        return selected_df

    selected_df = selected_df.sort_values(
        ["chunk_role", "final_arbitration_score"],
        key=lambda series: series.map(ROLE_PRIORITY).fillna(-1) if series.name == "chunk_role" else series,
        ascending=[False, False],
    ).reset_index(drop=True)
    selected_df["chunk_rank"] = range(1, len(selected_df) + 1)
    return selected_df


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_storage_config, resolve_storage_path

    storage_config = get_storage_config()
    processed_root = resolve_storage_path(storage_config.get("processed_data"))
    db_path = project_root / "data" / "gaira.duckdb"

    if processed_root is None:
        print("The storage config is missing processed_data.")
        return

    base_path = processed_root / "shine_class_reference_matches"
    interpreted_path = base_path / "shine_class_consensus_interpreted.csv"
    v3_path = base_path / "shine_class_supporting_chunks_v3.csv"
    output_path = base_path / "shine_class_supporting_chunks_v4.csv"

    for path in [interpreted_path, v3_path]:
        if not path.exists():
            print(f"Required input file not found: {path}")
            return

    interpreted_df = pd.read_csv(interpreted_path)
    v3_df = pd.read_csv(v3_path)

    with duckdb.connect(str(db_path), read_only=True) as connection:
        chunk_meta_df = connection.execute(
            """
            SELECT chunk_id, source_id, section, chunk_text, metadata_json
            FROM knowledge_chunks
            WHERE dataset_id = 'raman_knowledge_core'
            """
        ).fetchdf()
        semantic_regions_df = connection.execute(
            """
            SELECT region_id, region_label, dominant_group, secondary_groups
            FROM semantic_regions
            WHERE dataset_id = 'raman_knowledge_core'
            """
        ).fetchdf()
        confounder_df = connection.execute(
            """
            SELECT confounder_name, note_text
            FROM confounder_notes
            WHERE dataset_id = 'raman_knowledge_core'
            """
        ).fetchdf()

    if chunk_meta_df.empty:
        print("No knowledge_chunks rows were found for raman_knowledge_core.")
        return

    candidate_df = build_candidate_pool(interpreted_df, v3_df, chunk_meta_df)
    candidate_df = score_candidates(candidate_df)

    selected_groups: list[pd.DataFrame] = []
    for _, group_df in candidate_df.groupby(["class_label", "subclass_label"], sort=True):
        selected_df = select_role_aware_chunks(group_df.copy())
        if not selected_df.empty:
            selected_groups.append(selected_df)

    if not selected_groups:
        print("No v4 supporting chunks were selected.")
        return

    final_df = pd.concat(selected_groups, ignore_index=True)
    final_df["matched_terms"] = final_df["matched_terms"].fillna("")
    final_df["semantic_region_count"] = final_df.apply(
        lambda row: int(
            str(row["region_semantic_label_1"]).strip().lower() in semantic_regions_df["region_label"].str.lower().tolist()
            or str(row["region_semantic_label_2"]).strip().lower() in semantic_regions_df["region_label"].str.lower().tolist()
        ),
        axis=1,
    )
    final_df["confounder_reference_count"] = final_df["chunk_text"].apply(
        lambda text: sum(
            1
            for note in confounder_df["note_text"].tolist()
            if any(term in str(text).lower() for term in str(note).lower().split()[:3])
        )
    )

    keep_columns = [
        "class_label",
        "subclass_label",
        "chunk_rank",
        "chunk_id",
        "source_id",
        "section",
        "chunk_role",
        "chunk_text",
        "v3_total_score",
        "semantic_compatibility_score",
        "class_compatibility_score",
        "role_bonus",
        "role_conflict_penalty",
        "reuse_penalty_global",
        "confounder_overweight_penalty",
        "final_arbitration_score",
        "matched_terms",
    ]
    final_df = final_df[keep_columns].sort_values(
        ["class_label", "subclass_label", "chunk_rank"],
        ascending=[True, True, True],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print(f"SHINE supporting chunks v4 written to: {output_path}")
    print(f"Rows written: {len(final_df)}")
    print(final_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
