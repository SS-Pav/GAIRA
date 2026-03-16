import sys
from pathlib import Path

import duckdb
import pandas as pd


def parse_bucket(region_name: str | None) -> tuple[float, float] | None:
    """Parse a coarse bucket label like 900-1100 into numeric bounds."""
    if region_name is None:
        return None

    text = str(region_name).strip()
    if not text or text.lower() == "nan":
        return None

    try:
        lower, upper = [float(part) for part in text.split("-")]
    except ValueError:
        return None

    return lower, upper


def find_overlapping_regions(
    semantic_regions_df: pd.DataFrame,
    bucket_label: str | None,
) -> pd.DataFrame:
    """Find explicit semantic regions that overlap a SHINE coarse bucket."""
    parsed = parse_bucket(bucket_label)
    if parsed is None:
        return semantic_regions_df.iloc[0:0].copy()

    lower, upper = parsed
    overlaps_df = semantic_regions_df[
        (semantic_regions_df["region_min_cm"] <= upper)
        & (semantic_regions_df["region_max_cm"] >= lower)
    ].copy()
    if overlaps_df.empty:
        return overlaps_df

    overlaps_df["overlap_cm"] = overlaps_df.apply(
        lambda row: max(
            0.0,
            min(float(row["region_max_cm"]), upper) - max(float(row["region_min_cm"]), lower),
        ),
        axis=1,
    )
    return overlaps_df.sort_values(
        ["overlap_cm", "region_min_cm"],
        ascending=[False, True],
    ).reset_index(drop=True)


def unique_join(values: list[str], limit: int = 3) -> str:
    """Join unique non-empty values into a short semicolon-separated string."""
    cleaned: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text.lower() == "nan" or text in cleaned:
            continue
        cleaned.append(text)
    return "; ".join(cleaned[:limit])


def filter_claims(
    biomarker_claims_df: pd.DataFrame,
    top_class_1: str,
    top_class_2: str,
) -> pd.DataFrame:
    """Pick cautious biomarker claim rows that loosely match the broad class signal."""
    terms = [term for term in [top_class_1, top_class_2] if term and term.lower() != "nan"]
    if not terms:
        return biomarker_claims_df.iloc[0:0].copy()

    mask = pd.Series(False, index=biomarker_claims_df.index)
    for term in terms:
        token = term.split("/")[0].strip()
        if not token:
            continue
        mask = mask | biomarker_claims_df["claim_text"].fillna("").str.contains(
            token, case=False, na=False
        )
        mask = mask | biomarker_claims_df["biomarker_name"].fillna("").str.contains(
            token, case=False, na=False
        )
    return biomarker_claims_df[mask].copy()


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

    consensus_path = processed_root / "shine_class_reference_matches" / "shine_class_consensus_summary.csv"
    output_path = (
        processed_root
        / "shine_class_reference_matches"
        / "shine_class_consensus_interpreted.csv"
    )

    if not consensus_path.exists():
        print(f"SHINE consensus summary not found: {consensus_path}")
        return

    consensus_df = pd.read_csv(consensus_path)

    with duckdb.connect(str(db_path), read_only=True) as connection:
        semantic_regions_df = connection.execute(
            """
            SELECT
                region_id,
                dataset_id,
                region_label,
                region_min_cm,
                region_max_cm,
                dominant_group,
                secondary_groups,
                typical_examples,
                interpretation_note,
                caution_note
            FROM semantic_regions
            WHERE dataset_id = 'raman_knowledge_core'
            ORDER BY region_min_cm
            """
        ).fetchdf()
        biomarker_claims_df = connection.execute(
            """
            SELECT biomarker_name, disease_context, sample_type, claim_text, evidence_strength
            FROM biomarker_claims
            WHERE dataset_id = 'raman_knowledge_core'
            """
        ).fetchdf()
        confounder_df = connection.execute(
            """
            SELECT confounder_name, applies_to, note_text, mitigation_text
            FROM confounder_notes
            WHERE dataset_id = 'raman_knowledge_core'
            """
        ).fetchdf()

    if semantic_regions_df.empty:
        print("No semantic_regions rows were found. Ingest raman_knowledge_core with semantic_regions.csv first.")
        return

    interpreted_rows: list[dict] = []
    for row in consensus_df.to_dict(orient="records"):
        top_class_1 = str(row.get("top_biochemical_class_1") or "").strip()
        top_class_2 = str(row.get("top_biochemical_class_2") or "").strip()
        dominant_region_1 = str(row.get("dominant_region_1") or "").strip()
        dominant_region_2 = str(row.get("dominant_region_2") or "").strip()

        overlaps_1 = find_overlapping_regions(semantic_regions_df, dominant_region_1)
        overlaps_2 = find_overlapping_regions(semantic_regions_df, dominant_region_2)

        best_region_1 = overlaps_1.iloc[0] if not overlaps_1.empty else None
        best_region_2 = overlaps_2.iloc[0] if not overlaps_2.empty else None

        region_semantic_label_1 = None if best_region_1 is None else str(best_region_1["region_label"])
        region_semantic_label_2 = None if best_region_2 is None else str(best_region_2["region_label"])

        region_groups = []
        caution_notes = []
        if best_region_1 is not None:
            region_groups.extend(
                [
                    best_region_1["dominant_group"],
                    best_region_1["secondary_groups"],
                ]
            )
            caution_notes.append(best_region_1["caution_note"])
        if best_region_2 is not None:
            region_groups.extend(
                [
                    best_region_2["dominant_group"],
                    best_region_2["secondary_groups"],
                ]
            )
            caution_notes.append(best_region_2["caution_note"])

        claims_df = filter_claims(biomarker_claims_df, top_class_1, top_class_2)
        confounder_support_df = confounder_df[
            confounder_df["applies_to"].fillna("").str.contains(
                "ev|sers|serum|biofluids|protein|lipid|nucleic|carbohydrate",
                case=False,
                na=False,
            )
            | confounder_df["note_text"].fillna("").str.contains(
                "ev|sers|serum|biofluids|protein|lipid|nucleic|carbohydrate",
                case=False,
                na=False,
            )
        ].copy()

        cautious_interpretation = (
            f"{top_class_1} / {top_class_2} analog-reference pattern with strongest support in "
            f"{dominant_region_1} and {dominant_region_2} cm^-1."
        )
        if region_semantic_label_1 or region_semantic_label_2:
            cautious_interpretation += (
                f" Mapped semantic regions: {region_semantic_label_1 or 'unknown'}; "
                f"{region_semantic_label_2 or 'unknown'}."
            )

        interpreted_rows.append(
            {
                "class_label": row.get("class_label"),
                "subclass_label": row.get("subclass_label"),
                "top_biochemical_class_1": top_class_1,
                "top_biochemical_class_2": top_class_2,
                "dominant_region_1": dominant_region_1,
                "dominant_region_2": dominant_region_2,
                "region_semantic_label_1": region_semantic_label_1,
                "region_semantic_label_2": region_semantic_label_2,
                "knowledge_supported_groups": unique_join(region_groups, limit=4),
                "possible_biomarker_contexts": unique_join(
                    claims_df["claim_text"].tolist() if not claims_df.empty else [],
                    limit=3,
                ),
                "confounder_warnings": unique_join(
                    caution_notes + confounder_support_df["note_text"].tolist(),
                    limit=4,
                ),
                "cautious_interpretation": cautious_interpretation,
                "do_not_overclaim_note": (
                    "These outputs are structured knowledge-supported analog interpretations. "
                    "They are not literal molecule identifications and should be cross-checked "
                    "against matrix context, confounders, and source evidence."
                ),
            }
        )

    interpreted_df = pd.DataFrame(interpreted_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    interpreted_df.to_csv(output_path, index=False)

    print(f"Interpreted SHINE consensus written to: {output_path}")
    print(f"Rows written: {len(interpreted_df)}")
    print(interpreted_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
