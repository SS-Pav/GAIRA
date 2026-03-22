import json
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd


def tokenize(text: str) -> set[str]:
    """Convert a short text field into lowercase keyword tokens."""
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) >= 3
    }


def build_query_tokens(row: pd.Series) -> set[str]:
    """Collect retrieval keywords from one SHINE interpreted row."""
    tokens: set[str] = set()
    for column_name in [
        "top_biochemical_class_1",
        "top_biochemical_class_2",
        "region_semantic_label_1",
        "region_semantic_label_2",
        "dominant_region_1",
        "dominant_region_2",
        "knowledge_supported_groups",
    ]:
        tokens |= tokenize(row.get(column_name, ""))
    return tokens


def build_chunk_tokens(chunk_row: pd.Series) -> set[str]:
    """Collect retrieval tokens from one knowledge chunk row."""
    tokens: set[str] = set()
    for column_name in ["section", "chunk_text", "page_label", "metadata_json"]:
        tokens |= tokenize(chunk_row.get(column_name, ""))
    return tokens


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

    interpreted_path = (
        processed_root
        / "shine_class_reference_matches"
        / "shine_class_consensus_interpreted.csv"
    )
    output_path = (
        processed_root
        / "shine_class_reference_matches"
        / "shine_class_supporting_chunks.csv"
    )

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

    chunks_df["chunk_tokens"] = chunks_df.apply(build_chunk_tokens, axis=1)
    retrieval_rows: list[dict] = []

    for shine_row in shine_df.to_dict(orient="records"):
        shine_series = pd.Series(shine_row)
        query_tokens = build_query_tokens(shine_series)

        scored_rows: list[dict] = []
        for chunk_row in chunks_df.to_dict(orient="records"):
            chunk_series = pd.Series(chunk_row)
            overlap_tokens = sorted(query_tokens & chunk_series["chunk_tokens"])
            score = len(overlap_tokens)
            if score == 0:
                continue

            scored_rows.append(
                {
                    "class_label": shine_series["class_label"],
                    "subclass_label": shine_series["subclass_label"],
                    "chunk_id": chunk_series["chunk_id"],
                    "source_id": chunk_series["source_id"],
                    "section": chunk_series["section"],
                    "chunk_text": chunk_series["chunk_text"],
                    "score": score,
                    "overlap_tokens": "; ".join(overlap_tokens),
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
                    "score": 0,
                    "overlap_tokens": "",
                }
            )
            continue

        scored_df = scored_df.sort_values(
            ["score", "source_id", "chunk_id"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
        scored_df = scored_df.head(3).copy()
        scored_df.insert(2, "chunk_rank", range(1, len(scored_df) + 1))
        retrieval_rows.extend(scored_df.to_dict(orient="records"))

    output_df = pd.DataFrame(retrieval_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"Supporting chunk retrieval written to: {output_path}")
    print(f"Rows written: {len(output_df)}")
    print(output_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
