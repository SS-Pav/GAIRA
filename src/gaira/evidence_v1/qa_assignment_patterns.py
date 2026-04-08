from __future__ import annotations

import json
from pathlib import Path

import duckdb

from gaira.evidence_v1.constants import (
    CLEANUP_QA_ROOT,
    PATTERN_QA_ROOT,
    PATTERN_REPORT_ROOT,
    PATTERN_TABLES_ROOT,
    ensure_pattern_output_dirs,
)


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


def write_example_retrieval(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def generate_assignment_pattern_qa(
    db_path: Path,
    before_example_paths: dict[str, Path],
    after_example_paths: dict[str, Path],
) -> dict[str, int]:
    ensure_pattern_output_dirs()
    counts: dict[str, int] = {}
    with duckdb.connect(str(db_path), read_only=True) as connection:
        for table_name in (
            "evidence.assignment_patterns",
            "evidence.assignment_pattern_members",
            "evidence.peak_meaning_clusters",
            "retrieval.retrieval_runs",
        ):
            counts[table_name] = int(connection.sql(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])

        pattern_summary = connection.sql(
            """
            SELECT
                normalized_family,
                COUNT(*) AS pattern_count,
                ROUND(AVG(total_member_count), 3) AS avg_member_count,
                ROUND(AVG(core_member_count), 3) AS avg_core_member_count,
                SUM(CASE WHEN provisional THEN 1 ELSE 0 END) AS provisional_patterns
            FROM evidence.assignment_patterns
            GROUP BY normalized_family
            ORDER BY normalized_family
            """
        ).df()
        pattern_summary.to_csv(PATTERN_TABLES_ROOT / "pattern_summary.csv", index=False)

        member_distribution = connection.sql(
            """
            SELECT
                member_role,
                COUNT(*) AS member_count
            FROM evidence.assignment_pattern_members
            GROUP BY member_role
            ORDER BY member_role
            """
        ).df()
        member_distribution.to_csv(PATTERN_TABLES_ROOT / "pattern_member_distribution.csv", index=False)

        size_stats = connection.sql(
            """
            SELECT
                ROUND(AVG(total_member_count), 3) AS avg_pattern_size,
                MIN(total_member_count) AS min_pattern_size,
                MAX(total_member_count) AS max_pattern_size,
                ROUND(AVG(core_member_count), 3) AS avg_core_member_count
            FROM evidence.assignment_patterns
            """
        ).df()
        size_stats.to_csv(PATTERN_TABLES_ROOT / "pattern_size_stats.csv", index=False)

        pattern_quality = connection.sql(
            """
            SELECT
                pattern_id,
                pattern_label,
                normalized_family,
                total_member_count,
                core_member_count,
                source_diversity,
                evidence_count,
                confidence_score,
                ambiguity_score,
                provisional,
                loose_constellation
            FROM evidence.assignment_patterns
            ORDER BY confidence_score DESC, ambiguity_score ASC, pattern_id
            """
        ).df()
        pattern_quality.to_csv(PATTERN_TABLES_ROOT / "pattern_quality_audit.csv", index=False)

        mapping_summary = connection.sql(
            """
            WITH used AS (
                SELECT DISTINCT cluster_id
                FROM evidence.assignment_pattern_members
            )
            SELECT
                (SELECT COUNT(*) FROM used) AS used_clusters,
                (SELECT COUNT(*) FROM evidence.peak_meaning_clusters pmc WHERE pmc.cluster_id NOT IN (SELECT cluster_id FROM used)) AS unused_clusters,
                (SELECT COUNT(*) FROM evidence.peak_meaning_clusters) AS total_clusters
            """
        ).df()
        mapping_summary.to_csv(PATTERN_TABLES_ROOT / "pattern_cluster_mapping_summary.csv", index=False)

        mapping_examples = connection.sql(
            """
            SELECT
                ap.pattern_label,
                ap.normalized_family,
                apm.cluster_id,
                ROUND(apm.canonical_peak_cm, 3) AS canonical_peak_cm,
                apm.member_role,
                apm.member_weight
            FROM evidence.assignment_patterns ap
            JOIN evidence.assignment_pattern_members apm
              ON ap.pattern_id = apm.pattern_id
            ORDER BY ap.pattern_id, apm.member_role, apm.canonical_peak_cm
            LIMIT 80
            """
        ).df()
        mapping_examples.to_csv(PATTERN_TABLES_ROOT / "pattern_member_examples.csv", index=False)

    retrieval_lines = ["# Pattern Retrieval Before/After", ""]
    for name, before_path in before_example_paths.items():
        after_path = after_example_paths[name]
        before_payload = json.loads(before_path.read_text())
        after_payload = json.loads(after_path.read_text())
        before_titles = [item["title"] for item in before_payload.get("support_bundle_results", [])[:3]]
        after_titles = [item["pattern_label"] for item in after_payload.get("pattern_results", [])[:3]]
        retrieval_lines.extend(
            [
                f"## {name}",
                "",
                f"- Cluster-level top results: {before_titles}",
                f"- Pattern-aware top results: {after_titles}",
                "",
            ]
        )
    _write_text(PATTERN_QA_ROOT / "retrieval_before_after.md", "\n".join(retrieval_lines))

    schema_lines = [
        "# Assignment Pattern Schema Summary",
        "",
        "## Tables",
        "",
        "- `evidence.assignment_patterns`",
        "- `evidence.assignment_pattern_members`",
        "",
    ]
    _write_text(PATTERN_QA_ROOT / "schema_summary.md", "\n".join(schema_lines))

    quality_rows = []
    with duckdb.connect(str(db_path), read_only=True) as connection:
        strong_df = connection.sql(
            """
            SELECT pattern_label, normalized_family, total_member_count, core_member_count, confidence_score, ambiguity_score
            FROM evidence.assignment_patterns
            ORDER BY confidence_score DESC, ambiguity_score ASC
            LIMIT 8
            """
        ).df()
        weak_df = connection.sql(
            """
            SELECT pattern_label, normalized_family, total_member_count, core_member_count, confidence_score, ambiguity_score
            FROM evidence.assignment_patterns
            WHERE provisional = TRUE OR loose_constellation = TRUE
            ORDER BY confidence_score ASC, ambiguity_score DESC
            LIMIT 8
            """
        ).df()
        quality_lines = ["# Pattern Quality Audit", "", "## Stronger Patterns", ""]
        quality_lines.append(_markdown_table(strong_df.columns.tolist(), strong_df.astype(str).values.tolist()))
        quality_lines.extend(["", "## Weaker / Provisional Patterns", ""])
        quality_lines.append(_markdown_table(weak_df.columns.tolist(), weak_df.astype(str).values.tolist()))
        _write_text(PATTERN_QA_ROOT / "pattern_quality_audit.md", "\n".join(quality_lines))

        comparison_df = connection.sql(
            """
            WITH used AS (
              SELECT DISTINCT cluster_id FROM evidence.assignment_pattern_members
            )
            SELECT
              pmc.normalized_family,
              COUNT(*) AS total_clusters,
              SUM(CASE WHEN pmc.cluster_id IN (SELECT cluster_id FROM used) THEN 1 ELSE 0 END) AS used_clusters,
              SUM(CASE WHEN pmc.cluster_id NOT IN (SELECT cluster_id FROM used) THEN 1 ELSE 0 END) AS unused_clusters
            FROM evidence.peak_meaning_clusters pmc
            GROUP BY pmc.normalized_family
            ORDER BY pmc.normalized_family
            """
        ).df()
        comparison_df.to_csv(PATTERN_TABLES_ROOT / "pattern_vs_cluster_comparison.csv", index=False)
        comparison_lines = ["# Pattern vs Cluster Comparison", ""]
        comparison_lines.append(_markdown_table(comparison_df.columns.tolist(), comparison_df.astype(str).values.tolist()))
        _write_text(PATTERN_QA_ROOT / "pattern_vs_cluster_comparison.md", "\n".join(comparison_lines))

    _write_text(
        PATTERN_REPORT_ROOT / "qa_summary.md",
        "\n".join(
            [
                "# Assignment Pattern QA Summary",
                "",
                *(f"- `{table}`: {count}" for table, count in counts.items()),
            ]
        ),
    )
    return counts
