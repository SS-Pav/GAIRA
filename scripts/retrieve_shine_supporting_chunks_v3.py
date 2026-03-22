import json
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd


CLASS_TERM_MAP = {
    "proteins": {"protein", "proteins", "proteinaceous", "amide", "amidei", "amideiii"},
    "aminoacids": {"amino", "aminoacid", "aminoacids", "aromatic", "phenylalanine", "tyrosine", "tryptophan"},
    "lipids": {"lipid", "lipids", "fatty", "membrane", "ch", "deformation", "vesicle"},
    "fattyacids": {"lipid", "lipids", "fatty", "membrane", "ch", "deformation"},
    "hormones": {"hormone", "hormones", "sterol", "lipid", "serum"},
    "nucleicacids": {"nucleic", "acid", "dna", "rna", "phosphate", "base", "choline"},
    "saccharides": {"carbohydrate", "saccharide", "glycan", "sugar", "polysaccharide", "monosaccharide"},
    "monosaccharides": {"carbohydrate", "saccharide", "monosaccharide", "sugar"},
    "polysaccharides": {"carbohydrate", "saccharide", "polysaccharide", "glycan"},
    "triglycerides": {"lipid", "triglyceride", "membrane", "ch"},
}

REGION_CONCEPT_MAP = {
    "450-700": {"low", "wavenumber", "biosample", "mixed", "region_450_700"},
    "700-900": {"nucleic", "choline", "ring", "phosphate", "base", "region_700_900"},
    "900-1100": {"aromatic", "phosphate", "carbohydrate", "phenylalanine", "region_900_1100"},
    "1100-1300": {"amide", "carbohydrate", "lipid", "amideiii", "region_1100_1300"},
    "1300-1500": {"ch", "deformation", "lipid", "membrane", "protein", "region_1300_1500"},
    "1500-1700": {"amide", "amidei", "aromatic", "amino", "base", "region_1500_1700"},
    "1700-1800": {"carbonyl", "tail", "lipid", "hormone", "region_1700_1800"},
}

GENERIC_SECTION_NAMES = {"biofluid_interpretation", "confounders", "mixed_signature"}
MATRIX_TAGS = {"ev_interpretation", "sers_cautions", "serum_interpretation"}
REGION_TAG_PREFIX = "region_"


def tokenize(text: str) -> set[str]:
    """Normalize a text field into lowercase retrieval tokens."""
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) >= 2
    }


def expand_class_terms(label: str) -> set[str]:
    """Expand a biochemical class label into region-aware retrieval terms."""
    terms = tokenize(label)
    for token in list(terms):
        terms |= CLASS_TERM_MAP.get(token, set())
    return terms


def parse_metadata_tags(metadata_json: str) -> set[str]:
    """Extract retrieval tags from metadata_json."""
    try:
        payload = json.loads(str(metadata_json))
    except Exception:
        return set()

    tags = payload.get("tags", [])
    if not isinstance(tags, list):
        return set()
    return {str(tag).strip().lower() for tag in tags if str(tag).strip()}


def build_chunk_features(chunks_df: pd.DataFrame) -> pd.DataFrame:
    """Precompute chunk tokens, tags, and genericity signals."""
    section_counts = chunks_df["section"].value_counts(dropna=False).to_dict()
    feature_rows: list[dict] = []

    for row in chunks_df.to_dict(orient="records"):
        metadata_tags = parse_metadata_tags(row["metadata_json"])
        chunk_tokens = (
            tokenize(row["section"])
            | tokenize(row["chunk_text"])
            | tokenize(row["page_label"])
            | metadata_tags
        )
        region_tags = {tag for tag in metadata_tags if tag.startswith(REGION_TAG_PREFIX)}
        generic_penalty = 0
        if row["section"] in GENERIC_SECTION_NAMES and not region_tags:
            generic_penalty += 1
        if section_counts.get(row["section"], 0) >= 6 and len(metadata_tags) <= 3:
            generic_penalty += 1

        feature_rows.append(
            {
                **row,
                "metadata_tags": metadata_tags,
                "chunk_tokens": chunk_tokens,
                "region_tags": region_tags,
                "base_generic_penalty": generic_penalty,
            }
        )

    return pd.DataFrame(feature_rows)


def score_chunk(shine_row: pd.Series, chunk_row: pd.Series) -> dict:
    """Compute transparent v3 evidence scores for one chunk against one SHINE class."""
    class_terms_1 = expand_class_terms(str(shine_row.get("top_biochemical_class_1", "")))
    class_terms_2 = expand_class_terms(str(shine_row.get("top_biochemical_class_2", "")))
    region_terms_1 = tokenize(str(shine_row.get("region_semantic_label_1", ""))) | REGION_CONCEPT_MAP.get(
        str(shine_row.get("dominant_region_1", "")),
        set(),
    )
    region_terms_2 = tokenize(str(shine_row.get("region_semantic_label_2", ""))) | REGION_CONCEPT_MAP.get(
        str(shine_row.get("dominant_region_2", "")),
        set(),
    )
    group_terms = tokenize(str(shine_row.get("knowledge_supported_groups", "")))

    metadata_tags = chunk_row["metadata_tags"]
    chunk_tokens = chunk_row["chunk_tokens"]
    region_tags = chunk_row["region_tags"]
    section = str(chunk_row["section"]).lower()
    has_confounders = str(shine_row.get("confounder_warnings", "")).strip() != ""

    class_score = 0
    if (class_terms_1 & chunk_tokens) or (class_terms_1 & metadata_tags):
        class_score += 4
    if (class_terms_2 & chunk_tokens) or (class_terms_2 & metadata_tags):
        class_score += 3

    semantic_region_score = 0
    if region_terms_1 & chunk_tokens:
        semantic_region_score += 4
    if region_terms_2 & chunk_tokens:
        semantic_region_score += 3

    matrix_context_score = 0
    if "ev_interpretation" in metadata_tags:
        matrix_context_score += 2
    if "sers_cautions" in metadata_tags:
        matrix_context_score += 2
    if "serum_interpretation" in metadata_tags:
        matrix_context_score += 1

    dominant_region_1_tag = f"region_{str(shine_row.get('dominant_region_1', '')).replace('-', '_')}"
    dominant_region_2_tag = f"region_{str(shine_row.get('dominant_region_2', '')).replace('-', '_')}"
    region_specificity_score = 0
    if dominant_region_1_tag in region_tags:
        region_specificity_score += 3
    if dominant_region_2_tag in region_tags:
        region_specificity_score += 2

    confounder_score = 0
    if has_confounders and ("confounders" in metadata_tags or "sers_cautions" in metadata_tags or section in {"confounders", "sers_cautions"}):
        confounder_score += 2

    generic_penalty = int(chunk_row["base_generic_penalty"])
    if not region_tags and not (metadata_tags & MATRIX_TAGS):
        generic_penalty += 1

    matched_terms = sorted(
        (class_terms_1 & chunk_tokens)
        | (class_terms_2 & chunk_tokens)
        | (region_terms_1 & chunk_tokens)
        | (region_terms_2 & chunk_tokens)
        | (group_terms & metadata_tags)
        | region_tags
    )

    base_total = (
        class_score
        + semantic_region_score
        + matrix_context_score
        + region_specificity_score
        + confounder_score
        - generic_penalty
    )

    return {
        "base_total": base_total,
        "class_score": class_score,
        "semantic_region_score": semantic_region_score,
        "matrix_context_score": matrix_context_score,
        "region_specificity_score": region_specificity_score,
        "confounder_score": confounder_score,
        "generic_penalty": generic_penalty,
        "matched_terms": "; ".join(matched_terms),
    }


def apply_reuse_penalties(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """Penalize chunks that would otherwise dominate too many SHINE classes."""
    if candidate_df.empty:
        return candidate_df

    top5_df = candidate_df.sort_values(
        ["class_label", "subclass_label", "base_total", "semantic_region_score", "class_score"],
        ascending=[True, True, False, False, False],
    ).groupby(["class_label", "subclass_label"], as_index=False).head(5)

    reuse_counts = (
        top5_df.groupby("chunk_id")[["class_label"]]
        .count()
        .rename(columns={"class_label": "reuse_count"})
        .reset_index()
    )
    candidate_df = candidate_df.merge(reuse_counts, on="chunk_id", how="left")
    candidate_df["reuse_count"] = candidate_df["reuse_count"].fillna(0).astype(int)
    candidate_df["reuse_penalty"] = (candidate_df["reuse_count"] - 3).clip(lower=0)
    candidate_df["score_after_reuse"] = candidate_df["base_total"] - candidate_df["reuse_penalty"]
    return candidate_df


def pick_diverse_top_chunks(group_df: pd.DataFrame) -> pd.DataFrame:
    """Pick top chunks with a small bonus for section diversity."""
    selected_rows: list[dict] = []
    used_sections: set[str] = set()
    remaining_df = group_df.sort_values(
        ["score_after_reuse", "semantic_region_score", "region_specificity_score", "class_score", "chunk_id"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)

    while len(selected_rows) < 3 and not remaining_df.empty:
        working_df = remaining_df.copy()
        working_df["diversity_bonus"] = working_df["section"].apply(
            lambda value: 1 if value not in used_sections else 0
        )
        working_df["total_score"] = working_df["score_after_reuse"] + working_df["diversity_bonus"]
        best_row = working_df.sort_values(
            ["total_score", "diversity_bonus", "semantic_region_score", "region_specificity_score", "chunk_id"],
            ascending=[False, False, False, False, True],
        ).iloc[0]

        selected_rows.append(best_row.to_dict())
        used_sections.add(str(best_row["section"]))
        remaining_df = remaining_df[remaining_df["chunk_id"] != best_row["chunk_id"]].reset_index(drop=True)

    return pd.DataFrame(selected_rows)


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

    interpreted_path = processed_root / "shine_class_reference_matches" / "shine_class_consensus_interpreted.csv"
    output_path = processed_root / "shine_class_reference_matches" / "shine_class_supporting_chunks_v3.csv"

    if not interpreted_path.exists():
        print(f"Interpreted SHINE consensus file not found: {interpreted_path}")
        return

    shine_df = pd.read_csv(interpreted_path)

    with duckdb.connect(str(db_path), read_only=True) as connection:
        chunks_df = connection.execute(
            """
            SELECT
                chunk_id,
                source_id,
                dataset_id,
                section,
                chunk_text,
                chunk_order,
                page_label,
                metadata_json
            FROM knowledge_chunks
            WHERE dataset_id = 'raman_knowledge_core'
            ORDER BY source_id, chunk_order, chunk_id
            """
        ).fetchdf()

    if chunks_df.empty:
        print("No knowledge_chunks rows were found for raman_knowledge_core.")
        print("Ingest the local knowledge package with knowledge_chunks.csv first.")
        return

    feature_df = build_chunk_features(chunks_df)
    candidate_rows: list[dict] = []

    for shine_row in shine_df.to_dict(orient="records"):
        shine_series = pd.Series(shine_row)
        for chunk_row in feature_df.to_dict(orient="records"):
            chunk_series = pd.Series(chunk_row)
            score_parts = score_chunk(shine_series, chunk_series)
            if score_parts["base_total"] <= 0:
                continue

            candidate_rows.append(
                {
                    "class_label": shine_series["class_label"],
                    "subclass_label": shine_series["subclass_label"],
                    "chunk_id": chunk_series["chunk_id"],
                    "source_id": chunk_series["source_id"],
                    "section": chunk_series["section"],
                    "chunk_text": chunk_series["chunk_text"],
                    **score_parts,
                }
            )

    candidate_df = pd.DataFrame(candidate_rows)
    if candidate_df.empty:
        print("No v3 supporting chunks were scored for the SHINE interpreted rows.")
        return

    candidate_df = apply_reuse_penalties(candidate_df)

    selected_rows: list[dict] = []
    for (class_label, subclass_label), group_df in candidate_df.groupby(
        ["class_label", "subclass_label"],
        sort=True,
    ):
        top_df = pick_diverse_top_chunks(group_df)
        if top_df.empty:
            continue
        top_df.insert(2, "chunk_rank", range(1, len(top_df) + 1))
        selected_rows.extend(top_df.to_dict(orient="records"))

    output_df = pd.DataFrame(selected_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"Supporting chunk retrieval v3 written to: {output_path}")
    print(f"Rows written: {len(output_df)}")
    print(output_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
