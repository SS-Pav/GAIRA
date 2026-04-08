from __future__ import annotations

import json
from pathlib import Path

import duckdb

from gaira.evidence_v1.constants import (
    REFINEMENT_QA_ROOT,
    REFINEMENT_REPORT_ROOT,
    REFINEMENT_TABLES_ROOT,
    ensure_refinement_output_dirs,
)
from gaira.evidence_v1.schema import REFINEMENT_TABLES


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _markdown_table(columns: list[str], rows: list[list[object]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join("" if value is None else str(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def generate_refinement_qa_artifacts(
    db_path: Path,
    before_example_paths: dict[str, Path],
    after_example_paths: dict[str, Path],
) -> dict[str, int]:
    ensure_refinement_output_dirs()
    counts: dict[str, int] = {}
    with duckdb.connect(str(db_path), read_only=True) as connection:
        row_count_lines = ["table_name,row_count"]
        schema_lines = ["# Phase 1 Refinement Schema Summary", ""]
        for table_name in REFINEMENT_TABLES:
            row_count = int(connection.sql(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
            counts[table_name] = row_count
            row_count_lines.append(f"{table_name},{row_count}")
            schema_df = connection.sql(f"DESCRIBE {table_name}").df()
            schema_lines.append(f"## {table_name}")
            schema_lines.append(_markdown_table(schema_df.columns.tolist(), schema_df.astype(str).values.tolist()))
            schema_lines.append("")

        cluster_summary = connection.sql(
            """
            SELECT normalized_family, COUNT(*) AS cluster_count,
                   AVG(confidence_score) AS mean_confidence,
                   AVG(ambiguity_score) AS mean_ambiguity
            FROM evidence.peak_meaning_clusters
            GROUP BY normalized_family
            ORDER BY cluster_count DESC, normalized_family
            """
        ).df()
        cluster_summary.to_csv(REFINEMENT_TABLES_ROOT / "peak_meaning_cluster_summary.csv", index=False)

        mention_summary = connection.sql(
            """
            SELECT alignment_status, included_as_secondary_support, COUNT(*) AS row_count
            FROM evidence.operational_mention_links
            GROUP BY alignment_status, included_as_secondary_support
            ORDER BY alignment_status
            """
        ).df()
        mention_summary.to_csv(REFINEMENT_TABLES_ROOT / "operational_mention_summary.csv", index=False)

        context_edge_summary = connection.sql(
            """
            SELECT edge_type, COUNT(*) AS row_count
            FROM context.context_edges
            GROUP BY edge_type
            ORDER BY edge_type
            """
        ).df()
        context_edge_summary.to_csv(REFINEMENT_TABLES_ROOT / "context_edge_summary.csv", index=False)

        digitization_link_summary = connection.sql(
            """
            SELECT
                SUM(high_priority_digitization_count) AS high_priority_links,
                SUM(medium_priority_digitization_count) AS medium_priority_links,
                SUM(low_priority_digitization_count) AS low_priority_links
            FROM evidence.peak_meaning_clusters
            """
        ).df()
        digitization_link_summary.to_csv(REFINEMENT_TABLES_ROOT / "digitization_cluster_link_summary.csv", index=False)

        retrieval_doc_summary = connection.sql(
            """
            SELECT normalized_family, COUNT(*) AS retrievable_clusters
            FROM retrieval.peak_meaning_documents
            WHERE direct_retrieval_eligible = TRUE
            GROUP BY normalized_family
            ORDER BY retrievable_clusters DESC, normalized_family
            """
        ).df()
        retrieval_doc_summary.to_csv(REFINEMENT_TABLES_ROOT / "peak_meaning_document_summary.csv", index=False)

    _write_text(REFINEMENT_TABLES_ROOT / "table_row_counts.csv", "\n".join(row_count_lines) + "\n")
    _write_text(REFINEMENT_QA_ROOT / "schema_summary.md", "\n".join(schema_lines))

    before_after_lines = ["# Retrieval Structure Comparison", ""]
    for name, before_path in before_example_paths.items():
        after_path = after_example_paths[name]
        before_payload = json.loads(before_path.read_text())
        after_payload = json.loads(after_path.read_text())
        before_titles = [item["title"] for item in before_payload.get("direct_results", [])[:3]]
        after_titles = [item["title"] for item in after_payload.get("support_bundle_results", [])[:3]]
        before_after_lines.extend(
            [
                f"## {name}",
                "",
                f"- Before top-level objects: {before_titles}",
                f"- After top-level bundles: {after_titles}",
                "",
            ]
        )
    _write_text(REFINEMENT_QA_ROOT / "retrieval_before_after.md", "\n".join(before_after_lines))

    _write_text(
        REFINEMENT_REPORT_ROOT / "qa_summary.md",
        "\n".join(
            [
                "# GAIRA Phase 1 Refinement QA Summary",
                "",
                "## Row Counts",
                "",
                *(f"- `{table}`: {count}" for table, count in counts.items()),
            ]
        ),
    )
    return counts


def write_example_retrieval(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
