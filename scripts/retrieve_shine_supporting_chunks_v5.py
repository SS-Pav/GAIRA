import json
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd


CLASS_TAG_MAP = {
    "proteins": {"protein_regions", "amide_regions", "biofluid_interpretation"},
    "aminoacids": {"aromatic_regions", "protein_regions", "amide_regions"},
    "lipidsfattyacids": {"lipid_regions", "ch_regions", "ev_interpretation"},
    "lipidshormones": {"lipid_regions", "ch_regions", "sers_cautions"},
    "nucleicacids": {"nucleic_acid_regions", "aromatic_regions"},
    "saccharidesmonosaccharides": {"carbohydrate_regions", "biofluid_interpretation"},
    "saccharidespolysaccharides": {"carbohydrate_regions", "biofluid_interpretation"},
}

SEMANTIC_TAG_RULES = {
    "amideiproteinrichregion": {"protein_regions", "amide_regions"},
    "amideiiicarbohydratelipidoverlapregion": {"protein_regions", "amide_regions", "carbohydrate_regions", "lipid_regions"},
    "amideiiiandunsaturatedlipidregion": {"amide_regions", "lipid_regions"},
    "chdeformationlipidproteinoverlapregion": {"lipid_regions", "ch_regions", "protein_regions", "ev_interpretation"},
    "broadchdeformationbiosampleregion": {"ch_regions", "lipid_regions", "biofluid_interpretation"},
    "cholinenucleicacidringmoderegion": {"nucleic_acid_regions", "aromatic_regions"},
    "nucleicacidphosphateandbaseregion": {"nucleic_acid_regions", "aromatic_regions"},
    "aromaticphosphatecarbohydrateoverlapregion": {"aromatic_regions", "nucleic_acid_regions", "carbohydrate_regions"},
    "aromaticaminoacidandbaseoverlapregion": {"aromatic_regions", "protein_regions", "nucleic_acid_regions"},
    "lowwavenumbermixedbiosampleregion": {"biofluid_interpretation", "mixed_signature", "protein_regions", "lipid_regions"},
    "highwavenumbercarbonylassociatedtailregion": {"lipid_regions", "amide_regions"},
}

REGION_TAGS = {
    "450-700": {"region_450_700"},
    "700-900": {"region_700_900", "region_700_780", "region_780_900"},
    "900-1100": {"region_900_1100"},
    "1100-1300": {"region_1100_1230", "region_1230_1300", "region_1100_1300"},
    "1300-1500": {"region_1300_1450", "region_1450_1500", "region_1300_1500"},
    "1500-1700": {"region_1500_1610", "region_1610_1700", "region_1500_1700"},
    "1700-1800": {"region_1700_1800"},
}

GENERIC_SECTIONS = {"biofluid_interpretation", "mixed_signature"}


def normalize_label(text: str) -> str:
    """Collapse a label into a lowercase alphanumeric token."""
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def tokenize(text: str) -> set[str]:
    """Tokenize free text into lowercase terms."""
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) >= 2
    }


def parse_tags(metadata_json: str) -> set[str]:
    """Read tag strings from chunk metadata."""
    try:
        payload = json.loads(str(metadata_json))
    except Exception:
        return set()

    tags = payload.get("tags", [])
    if not isinstance(tags, list):
        return set()
    return {str(tag).strip().lower() for tag in tags if str(tag).strip()}


def parse_region_range(region_text: str) -> tuple[float, float] | None:
    """Parse a simple spectral region like 1200-1700."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", str(region_text))
    if not match:
        return None
    start = float(match.group(1))
    end = float(match.group(2))
    return min(start, end), max(start, end)


def overlap_size(range_a: tuple[float, float] | None, range_b: tuple[float, float] | None) -> float:
    """Return overlap width between two numeric intervals."""
    if range_a is None or range_b is None:
        return 0.0
    start = max(range_a[0], range_b[0])
    end = min(range_a[1], range_b[1])
    return max(0.0, end - start)


def infer_chunk_role(section: str, tags: set[str], chunk_text: str) -> str:
    """Assign a broad retrieval role to one chunk."""
    section = str(section).lower()
    text = str(chunk_text).lower()
    if section in {"protein_regions", "amide_regions", "lipid_regions", "nucleic_acid_regions", "carbohydrate_regions", "ch_regions", "aromatic_regions"}:
        return "region_mechanistic"
    if section in {"ev_interpretation", "serum_interpretation", "biofluid_interpretation"}:
        return "matrix_context"
    if section in {"confounders", "sers_cautions"}:
        return "confounder_or_caution"
    if section in {"ev_interpretation", "serum_interpretation", "biofluid_interpretation"} or {
        "ev_interpretation",
        "serum_interpretation",
        "biofluid_interpretation",
    } & tags:
        return "matrix_context"
    if section in {"confounders", "sers_cautions"} or {"confounders", "sers_cautions"} & tags:
        return "confounder_or_caution"
    if {
        "protein_regions",
        "amide_regions",
        "lipid_regions",
        "nucleic_acid_regions",
        "carbohydrate_regions",
        "ch_regions",
        "aromatic_regions",
    } & tags:
        return "region_mechanistic"
    if "confound" in text or "caution" in text:
        return "confounder_or_caution"
    if "vesicle" in text or "serum" in text or "biofluid" in text:
        return "matrix_context"
    return "mixed_context"


def build_claim_profiles(claims_df: pd.DataFrame) -> list[dict]:
    """Precompute retrieval-friendly biomarker claim features."""
    profiles: list[dict] = []
    for row in claims_df.to_dict(orient="records"):
        profiles.append(
            {
                "claim_id": row["claim_id"],
                "tokens": tokenize(row["biomarker_name"]) | tokenize(row["claim_text"]) | tokenize(row["sample_type"]) | tokenize(row["disease_context"]),
                "region": parse_region_range(row["spectral_region"]),
            }
        )
    return profiles


def build_confounder_profiles(confounder_df: pd.DataFrame) -> list[dict]:
    """Precompute retrieval-friendly confounder note features."""
    profiles: list[dict] = []
    for row in confounder_df.to_dict(orient="records"):
        profiles.append(
            {
                "confounder_id": row["confounder_id"],
                "tokens": tokenize(row["confounder_name"]) | tokenize(row["note_text"]) | tokenize(row["applies_to"]),
            }
        )
    return profiles


def score_row_chunk(
    shine_row: pd.Series,
    chunk_row: pd.Series,
    claim_profiles: list[dict],
    confounder_profiles: list[dict],
) -> dict:
    """Compute transparent v5 retrieval scores for one SHINE row/chunk pair."""
    tags = chunk_row["metadata_tags"]
    chunk_tokens = chunk_row["chunk_tokens"]

    class_score = 0.0
    for class_label, weight in [
        (shine_row["top_biochemical_class_1"], 4.0),
        (shine_row["top_biochemical_class_2"], 3.0),
    ]:
        rule_tags = CLASS_TAG_MAP.get(normalize_label(class_label), set())
        if tags & rule_tags:
            class_score += weight
        elif chunk_row["chunk_role"] == "matrix_context":
            class_score += 0.5

    semantic_region_score = 0.0
    for label, weight in [
        (shine_row["region_semantic_label_1"], 4.0),
        (shine_row["region_semantic_label_2"], 3.0),
    ]:
        preferred_tags = SEMANTIC_TAG_RULES.get(normalize_label(label), set())
        if tags & preferred_tags:
            semantic_region_score += weight
        elif chunk_row["chunk_role"] == "matrix_context":
            semantic_region_score += 0.5

    matrix_context_score = 0.0
    if "ev_interpretation" in tags:
        matrix_context_score += 2.0
    if "sers_cautions" in tags:
        matrix_context_score += 1.5
    if "biofluid_interpretation" in tags:
        matrix_context_score += 1.5
    if "serum_interpretation" in tags:
        matrix_context_score += 1.0

    region_specificity_score = 0.0
    for region_bucket, weight in [
        (shine_row["dominant_region_1"], 2.5),
        (shine_row["dominant_region_2"], 1.5),
    ]:
        if tags & REGION_TAGS.get(str(region_bucket), set()):
            region_specificity_score += weight

    claim_context_score = 0.0
    row_tokens = (
        tokenize(shine_row["top_biochemical_class_1"])
        | tokenize(shine_row["top_biochemical_class_2"])
        | tokenize(shine_row["possible_biomarker_contexts"])
        | tokenize(shine_row["knowledge_supported_groups"])
        | {"extracellular", "vesicles", "biofluids", "sers", "hepatotoxicity"}
    )
    dominant_regions = [
        parse_region_range(str(shine_row["dominant_region_1"])),
        parse_region_range(str(shine_row["dominant_region_2"])),
    ]
    for profile in claim_profiles:
        if not (profile["tokens"] & (chunk_tokens | tags) and profile["tokens"] & row_tokens):
            continue
        overlap_bonus = 0.0
        for row_region in dominant_regions:
            if overlap_size(profile["region"], row_region) > 0:
                overlap_bonus = max(overlap_bonus, 1.5)
        claim_context_score = max(claim_context_score, 1.0 + overlap_bonus)

    confounder_context_score = 0.0
    row_confounder_tokens = tokenize(shine_row["confounder_warnings"])
    for profile in confounder_profiles:
        overlap = profile["tokens"] & (chunk_tokens | tags | row_confounder_tokens)
        if overlap:
            score = 1.0
            if chunk_row["chunk_role"] == "confounder_or_caution":
                score += 0.5
            confounder_context_score = max(confounder_context_score, score)

    generic_penalty = 0.0
    if chunk_row["section"] in GENERIC_SECTIONS:
        generic_penalty += 0.6
    if len(tags) <= 2:
        generic_penalty += 0.4
    if not any(tag.startswith("region_") for tag in tags) and chunk_row["chunk_role"] != "confounder_or_caution":
        generic_penalty += 0.3

    matched_terms = sorted(
        (
            tokenize(shine_row["top_biochemical_class_1"])
            | tokenize(shine_row["top_biochemical_class_2"])
            | tokenize(shine_row["region_semantic_label_1"])
            | tokenize(shine_row["region_semantic_label_2"])
            | tokenize(shine_row["dominant_region_1"])
            | tokenize(shine_row["dominant_region_2"])
            | tokenize(shine_row["knowledge_supported_groups"])
        )
        & (chunk_tokens | tags)
    )

    total_score = (
        class_score
        + semantic_region_score
        + matrix_context_score
        + region_specificity_score
        + claim_context_score
        + confounder_context_score
        - generic_penalty
    )

    return {
        "total_score": total_score,
        "class_score": class_score,
        "semantic_region_score": semantic_region_score,
        "matrix_context_score": matrix_context_score,
        "region_specificity_score": region_specificity_score,
        "claim_context_score": claim_context_score,
        "confounder_context_score": confounder_context_score,
        "generic_penalty": generic_penalty,
        "matched_terms": "; ".join(matched_terms),
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
    output_path = processed_root / "shine_class_reference_matches" / "shine_class_supporting_chunks_v5_pool.csv"

    if not interpreted_path.exists():
        print(f"Interpreted SHINE consensus file not found: {interpreted_path}")
        return

    shine_df = pd.read_csv(interpreted_path)

    with duckdb.connect(str(db_path), read_only=True) as connection:
        chunks_df = connection.execute(
            """
            SELECT chunk_id, source_id, dataset_id, section, chunk_text, chunk_order, page_label, metadata_json
            FROM knowledge_chunks
            WHERE dataset_id = 'raman_knowledge_core'
            ORDER BY chunk_id
            """
        ).fetchdf()
        semantic_df = connection.execute(
            """
            SELECT region_label, dominant_group, secondary_groups
            FROM semantic_regions
            WHERE dataset_id = 'raman_knowledge_core'
            """
        ).fetchdf()
        claims_df = connection.execute(
            """
            SELECT claim_id, biomarker_name, disease_context, sample_type, spectral_region, claim_text
            FROM biomarker_claims
            WHERE dataset_id = 'raman_knowledge_core'
            ORDER BY claim_id
            """
        ).fetchdf()
        confounder_df = connection.execute(
            """
            SELECT confounder_id, confounder_name, applies_to, note_text
            FROM confounder_notes
            WHERE dataset_id = 'raman_knowledge_core'
            ORDER BY confounder_id
            """
        ).fetchdf()

    if chunks_df.empty:
        print("No knowledge chunks are available for raman_knowledge_core.")
        return

    chunks_df["metadata_tags"] = chunks_df["metadata_json"].apply(parse_tags)
    chunks_df["chunk_tokens"] = chunks_df.apply(
        lambda row: tokenize(row["section"]) | tokenize(row["chunk_text"]) | tokenize(row["page_label"]) | row["metadata_tags"],
        axis=1,
    )
    chunks_df["chunk_role"] = chunks_df.apply(
        lambda row: infer_chunk_role(row["section"], row["metadata_tags"], row["chunk_text"]),
        axis=1,
    )

    claim_profiles = build_claim_profiles(claims_df)
    confounder_profiles = build_confounder_profiles(confounder_df)
    semantic_labels = {normalize_label(value) for value in semantic_df["region_label"].tolist()}

    scored_rows: list[dict] = []
    for shine_row in shine_df.to_dict(orient="records"):
        shine_series = pd.Series(shine_row)
        for chunk_row in chunks_df.to_dict(orient="records"):
            chunk_series = pd.Series(chunk_row)
            score_dict = score_row_chunk(shine_series, chunk_series, claim_profiles, confounder_profiles)
            scored_rows.append(
                {
                    "class_label": shine_series["class_label"],
                    "subclass_label": shine_series["subclass_label"],
                    "top_biochemical_class_1": shine_series["top_biochemical_class_1"],
                    "top_biochemical_class_2": shine_series["top_biochemical_class_2"],
                    "dominant_region_1": shine_series["dominant_region_1"],
                    "dominant_region_2": shine_series["dominant_region_2"],
                    "region_semantic_label_1": shine_series["region_semantic_label_1"],
                    "region_semantic_label_2": shine_series["region_semantic_label_2"],
                    "knowledge_supported_groups": shine_series["knowledge_supported_groups"],
                    "possible_biomarker_contexts": shine_series["possible_biomarker_contexts"],
                    "confounder_warnings": shine_series["confounder_warnings"],
                    "chunk_id": chunk_series["chunk_id"],
                    "source_id": chunk_series["source_id"],
                    "section": chunk_series["section"],
                    "chunk_role": chunk_series["chunk_role"],
                    "chunk_text": chunk_series["chunk_text"],
                    "metadata_json": chunk_series["metadata_json"],
                    "semantic_known": int(normalize_label(shine_series["region_semantic_label_1"]) in semantic_labels),
                    **score_dict,
                }
            )

    score_df = pd.DataFrame(scored_rows)

    # Reuse penalty from a broad provisional shortlist rather than the final top rows.
    provisional_df = (
        score_df.sort_values(
            ["class_label", "subclass_label", "total_score", "semantic_region_score", "region_specificity_score"],
            ascending=[True, True, False, False, False],
        )
        .groupby(["class_label", "subclass_label"], as_index=False)
        .head(12)
        .copy()
    )
    reuse_df = (
        provisional_df.groupby("chunk_id")[["class_label"]]
        .count()
        .rename(columns={"class_label": "reuse_count"})
        .reset_index()
    )
    score_df = score_df.merge(reuse_df, on="chunk_id", how="left")
    score_df["reuse_count"] = score_df["reuse_count"].fillna(0).astype(int)
    score_df["reuse_penalty"] = ((score_df["reuse_count"] - 4).clip(lower=0) * 0.45).astype(float)
    generic_mask = score_df["section"].isin(GENERIC_SECTIONS)
    score_df.loc[generic_mask, "reuse_penalty"] = (
        score_df.loc[generic_mask, "reuse_penalty"] + (score_df.loc[generic_mask, "reuse_count"] - 3).clip(lower=0) * 0.2
    )
    score_df["total_score"] = score_df["total_score"] - score_df["reuse_penalty"]

    final_pool_df = (
        score_df.sort_values(
            ["class_label", "subclass_label", "total_score", "semantic_region_score", "class_score", "chunk_id"],
            ascending=[True, True, False, False, False, True],
        )
        .groupby(["class_label", "subclass_label"], as_index=False)
        .head(8)
        .copy()
    )
    final_pool_df["pool_rank"] = (
        final_pool_df.groupby(["class_label", "subclass_label"]).cumcount() + 1
    )

    keep_columns = [
        "class_label",
        "subclass_label",
        "pool_rank",
        "chunk_id",
        "source_id",
        "section",
        "chunk_role",
        "chunk_text",
        "total_score",
        "class_score",
        "semantic_region_score",
        "matrix_context_score",
        "region_specificity_score",
        "claim_context_score",
        "confounder_context_score",
        "generic_penalty",
        "reuse_penalty",
        "matched_terms",
        "top_biochemical_class_1",
        "top_biochemical_class_2",
        "dominant_region_1",
        "dominant_region_2",
        "region_semantic_label_1",
        "region_semantic_label_2",
        "knowledge_supported_groups",
        "possible_biomarker_contexts",
        "confounder_warnings",
    ]
    final_pool_df = final_pool_df[keep_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_pool_df.to_csv(output_path, index=False)

    print(f"SHINE supporting chunks v5 pool written to: {output_path}")
    print(f"Rows written: {len(final_pool_df)}")
    print(final_pool_df.head(16).to_string(index=False))


if __name__ == "__main__":
    main()
