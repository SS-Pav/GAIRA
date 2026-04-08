from __future__ import annotations

import json
from pathlib import Path

import duckdb

from gaira.evidence_v1.constants import QA_ROOT, REPORT_ROOT, TABLES_ROOT, ensure_output_dirs
from gaira.evidence_v1.schema import V1_TABLES


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


def generate_qa_artifacts(db_path: Path) -> dict[str, int]:
    ensure_output_dirs()
    counts: dict[str, int] = {}
    with duckdb.connect(str(db_path), read_only=True) as connection:
        row_count_lines = ["table_name,row_count"]
        schema_lines = ["# Evidence v1 Schema Summary", ""]
        for table_name in V1_TABLES:
            row_count = int(connection.sql(f"SELECT COUNT(*) AS n FROM {table_name}").fetchone()[0])
            counts[table_name] = row_count
            row_count_lines.append(f"{table_name},{row_count}")
            schema_df = connection.sql(f"DESCRIBE {table_name}").df()
            schema_lines.append(f"## {table_name}")
            schema_lines.append(_markdown_table(schema_df.columns.tolist(), schema_df.astype(str).values.tolist()))
            schema_lines.append("")

        evidence_breakdown = connection.sql(
            """
            SELECT evidence_kind, evidence_tier, COUNT(*) AS row_count
            FROM evidence.evidence_items
            GROUP BY evidence_kind, evidence_tier
            ORDER BY evidence_kind, evidence_tier
            """
        ).df()
        evidence_breakdown.to_csv(TABLES_ROOT / "evidence_type_tier_breakdown.csv", index=False)

        source_summary = connection.sql(
            """
            SELECT s.source_id, s.source_kind, s.evidence_role, COUNT(i.evidence_item_id) AS evidence_item_count
            FROM registry.evidence_sources s
            LEFT JOIN evidence.evidence_items i ON i.source_id = s.source_id
            GROUP BY s.source_id, s.source_kind, s.evidence_role
            ORDER BY evidence_item_count DESC, s.source_id
            """
        ).df()
        source_summary.to_csv(TABLES_ROOT / "source_provenance_summary.csv", index=False)

        integrity_rows = []
        integrity_checks = {
            "all_evidence_items_have_source": """
                SELECT COUNT(*) FROM evidence.evidence_items i
                LEFT JOIN registry.evidence_sources s ON s.source_id = i.source_id
                WHERE s.source_id IS NULL
            """,
            "mentions_in_feature_table": """
                SELECT COUNT(*) FROM features.spectral_features f
                JOIN evidence.evidence_items i ON i.evidence_item_id = f.evidence_item_id
                WHERE i.evidence_kind = 'wavenumber_mention'
            """,
            "mention_docs_directly_retrievable": """
                SELECT COUNT(*) FROM retrieval.retrieval_documents d
                JOIN evidence.evidence_items i ON i.evidence_item_id = d.evidence_item_id
                WHERE i.evidence_kind = 'wavenumber_mention'
                  AND d.direct_retrieval_eligible = TRUE
            """,
            "null_peak_centers_in_direct_features": """
                SELECT COUNT(*) FROM features.spectral_features
                WHERE feature_type = 'peak' AND peak_center_cm IS NULL
            """,
            "trieste_hcc_direct_reference_rows": """
                SELECT COUNT(*) FROM evidence.reference_spectrum_evidence
                WHERE lower(COALESCE(notes, '')) LIKE '%trieste%'
                   OR lower(COALESCE(component, '')) LIKE '%trieste%'
            """,
        }
        for check_name, sql in integrity_checks.items():
            value = int(connection.sql(sql).fetchone()[0])
            integrity_rows.append({"check_name": check_name, "value": value, "status": "pass" if value == 0 else "review"})
        connection.sql(
            """
            SELECT priority, COUNT(*) AS row_count
            FROM evidence.digitized_spectrum_registry
            GROUP BY priority
            ORDER BY priority
            """
        ).df().to_csv(TABLES_ROOT / "digitization_registry_summary.csv", index=False)

        mention_count = int(connection.sql("SELECT COUNT(*) FROM evidence.wavenumber_mentions").fetchone()[0])
        retrieval_mention_count = int(
            connection.sql(
                """
                SELECT COUNT(*)
                FROM retrieval.retrieval_documents d
                JOIN evidence.evidence_items i ON i.evidence_item_id = d.evidence_item_id
                WHERE i.evidence_kind = 'wavenumber_mention'
                  AND d.direct_retrieval_eligible = TRUE
                """
            ).fetchone()[0]
        )

    _write_text(TABLES_ROOT / "table_row_counts.csv", "\n".join(row_count_lines) + "\n")
    _write_text(QA_ROOT / "schema_summary.md", "\n".join(schema_lines))

    integrity_csv = "check_name,value,status\n" + "\n".join(
        f"{row['check_name']},{row['value']},{row['status']}" for row in integrity_rows
    ) + "\n"
    _write_text(TABLES_ROOT / "integrity_checks.csv", integrity_csv)
    _write_text(
        QA_ROOT / "mention_exclusion_summary.md",
        "\n".join(
            [
                "# Mention Exclusion Summary",
                "",
                f"- Mention rows loaded into `evidence.wavenumber_mentions`: {mention_count}",
                f"- Mention rows directly retrievable: {retrieval_mention_count}",
                "- Direct retrieval exclusion rule: mention rows are never inserted into `features.spectral_features` and never marked as `direct_retrieval_eligible`.",
            ]
        ),
    )
    _write_text(
        REPORT_ROOT / "qa_summary.md",
        "\n".join(
            [
                "# GAIRA Evidence Operationalization v1 QA Summary",
                "",
                "## Row Counts",
                "",
                *(f"- `{table}`: {count}" for table, count in counts.items()),
                "",
                "## Integrity Checks",
                "",
                *(f"- `{row['check_name']}`: {row['value']} ({row['status']})" for row in integrity_rows),
            ]
        ),
    )
    return counts


def write_example_retrieval(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
