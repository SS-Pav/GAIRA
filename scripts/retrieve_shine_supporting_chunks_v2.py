import json
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd


CLASS_TERM_MAP = {
    "proteins": {"protein", "proteins", "proteinaceous", "amide", "amidei", "amideiii"},
    "aminoacids": {"amino", "aminoacid", "aminoacids", "aromatic"},
    "lipids": {"lipid", "lipids", "fatty", "membrane", "ch", "deformation"},
    "fattyacids": {"lipid", "lipids", "fatty", "membrane", "ch", "deformation"},
    "hormones": {"hormone", "hormones", "sterol", "lipid"},
    "nucleicacids": {"nucleic", "acid", "dna", "rna", "phosphate", "base", "choline"},
    "saccharides": {"carbohydrate", "saccharide", "glycan", "sugar", "polysaccharide", "monosaccharide"},
    "monosaccharides": {"carbohydrate", "saccharide", "monosaccharide", "sugar"},
    "polysaccharides": {"carbohydrate", "saccharide", "polysaccharide", "glycan"},
    "triglycerides": {"lipid", "triglyceride", "membrane", "ch"},
}

REGION_CONCEPT_MAP = {
    "450-700": {"low", "wavenumber", "biosample", "mixed"},
    "700-900": {"nucleic", "choline", "ring", "phosphate", "base"},
    "900-1100": {"aromatic", "phosphate", "carbohydrate", "phenylalanine"},
    "1100-1300": {"amide", "carbohydrate", "lipid", "amideiii"},
    "1300-1500": {"ch", "deformation", "lipid", "membrane", "protein"},
    "1500-1700": {"amide", "amidei", "aromatic", "amino", "base"},
    "1700-1800": {"carbonyl", "tail", "lipid", "hormone"},
}

GENERIC_TOKENS = {"biofluid", "biofluids", "interpretation", "region", "regions", "cautious", "analog"}


def tokenize(text: str) -> set[str]:
    """Convert text into normalized alphanumeric tokens."""
    normalized = str(text).replace("III", "iii").replace("I", "i")
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", normalized.lower())
        if len(token) >= 2
    }
    collapsed: set[str] = set()
    for token in tokens:
        collapsed.add(token.replace("fattyacids", "fattyacids"))
        collapsed.add(token.replace("nucleicacids", "nucleicacids"))
        collapsed.add(token.replace("aminoacids", "aminoacids"))
        collapsed.add(token.replace("lipids", "lipids"))
    return collapsed | tokens


def expand_class_terms(label: str) -> set[str]:
    """Map a broad biochemical class label to region-aware retrieval terms."""
    terms = tokenize(label)
    for token in list(terms):
        terms |= CLASS_TERM_MAP.get(token, set())
    return terms


def parse_metadata_tags(metadata_json: str) -> set[str]:
    """Extract lightweight retrieval tags from chunk metadata_json."""
    try:
        payload = json.loads(str(metadata_json))
    except Exception:
        return set()

    tags = payload.get("tags", [])
    if not isinstance(tags, list):
        return set()
    return {str(tag).strip().lower() for tag in tags if str(tag).strip()}


def build_chunk_features(chunks_df: pd.DataFrame) -> pd.DataFrame:
    """Precompute chunk tokens and genericity for scoring."""
    feature_rows: list[dict] = []
    section_counts = chunks_df["section"].value_counts(dropna=False).to_dict()

    for row in chunks_df.to_dict(orient="records"):
        chunk_tokens = tokenize(row["section"]) | tokenize(row["chunk_text"]) | tokenize(row["page_label"])
        metadata_tags = parse_metadata_tags(row["metadata_json"])
        chunk_tokens |= metadata_tags

        generic_hits = len(chunk_tokens & GENERIC_TOKENS)
        generic_penalty = 1 if section_counts.get(row["section"], 0) >= 4 and generic_hits >= 2 else 0

        feature_rows.append(
            {
                **row,
                "chunk_tokens": chunk_tokens,
                "metadata_tags": metadata_tags,
                "generic_penalty": generic_penalty,
            }
        )

    return pd.DataFrame(feature_rows)


def score_chunk(shine_row: pd.Series, chunk_row: pd.Series) -> dict:
    """Score one chunk against one interpreted SHINE row with transparent components."""
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
    chunk_tokens = chunk_row["chunk_tokens"]
    metadata_tags = chunk_row["metadata_tags"]
    text = str(chunk_row["chunk_text"]).lower()
    section = str(chunk_row["section"]).lower()
    has_confounders = str(shine_row.get("confounder_warnings", "")).strip() != ""

    class_1_match = bool((class_terms_1 & chunk_tokens) or (class_terms_1 & metadata_tags))
    class_2_match = bool((class_terms_2 & chunk_tokens) or (class_terms_2 & metadata_tags))
    class_score = (4 if class_1_match else 0) + (3 if class_2_match else 0)

    region_1_match = bool(region_terms_1 & chunk_tokens)
    region_2_match = bool(region_terms_2 & chunk_tokens)
    semantic_region_score = (4 if region_1_match else 0) + (3 if region_2_match else 0)

    region_keywords = {"amide", "amidei", "amideiii", "ch", "phosphate", "aromatic", "carbohydrate", "ev", "sers", "confounder", "membrane", "nucleic"}
    region_keyword_hits = sorted(region_keywords & chunk_tokens)
    region_keyword_score = min(2, len(region_keyword_hits))

    confounder_score = 0
    if has_confounders and ("confounder" in section or "caution" in section or "sers_cautions" in section or "confounders" in metadata_tags):
        confounder_score = 2

    group_overlap = sorted(group_terms & metadata_tags)
    group_score = 1 if group_overlap else 0

    generic_penalty = int(chunk_row["generic_penalty"])
    total_score = class_score + semantic_region_score + region_keyword_score + confounder_score + group_score - generic_penalty

    matched_terms = sorted(
        (class_terms_1 & chunk_tokens)
        | (class_terms_2 & chunk_tokens)
        | (region_terms_1 & chunk_tokens)
        | (region_terms_2 & chunk_tokens)
        | set(region_keyword_hits)
        | set(group_overlap)
    )

    return {
        "total_score": total_score,
        "class_score": class_score,
        "semantic_region_score": semantic_region_score,
        "region_keyword_score": region_keyword_score,
        "confounder_score": confounder_score,
        "generic_penalty": generic_penalty,
        "matched_terms": "; ".join(sorted(matched_terms)),
    }


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

    interpreted_path = processed_root / "shine_class_reference_matches" / "shine_class_consensus_interpreted.csv"
    output_path = processed_root / "shine_class_reference_matches" / "shine_class_supporting_chunks_v2.csv"

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
    retrieval_rows: list[dict] = []

    for shine_row in shine_df.to_dict(orient="records"):
        shine_series = pd.Series(shine_row)
        scored_rows: list[dict] = []

        for chunk_row in feature_df.to_dict(orient="records"):
            chunk_series = pd.Series(chunk_row)
            score_parts = score_chunk(shine_series, chunk_series)
            if score_parts["total_score"] <= 0:
                continue

            scored_rows.append(
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

        scored_df = pd.DataFrame(scored_rows)
        if scored_df.empty:
            retrieval_rows.append(
                {
                    "class_label": shine_series["class_label"],
                    "subclass_label": shine_series["subclass_label"],
                    "chunk_rank": 1,
                    "chunk_id": "",
                    "source_id": "",
                    "section": "",
                    "chunk_text": "",
                    "total_score": 0,
                    "class_score": 0,
                    "semantic_region_score": 0,
                    "region_keyword_score": 0,
                    "confounder_score": 0,
                    "generic_penalty": 0,
                    "matched_terms": "",
                }
            )
            continue

        scored_df = scored_df.sort_values(
            ["total_score", "semantic_region_score", "class_score", "source_id", "chunk_id"],
            ascending=[False, False, False, True, True],
        ).reset_index(drop=True)
        scored_df = scored_df.head(3).copy()
        scored_df.insert(2, "chunk_rank", range(1, len(scored_df) + 1))
        retrieval_rows.extend(scored_df.to_dict(orient="records"))

    output_df = pd.DataFrame(retrieval_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"Supporting chunk retrieval v2 written to: {output_path}")
    print(f"Rows written: {len(output_df)}")
    print(output_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
