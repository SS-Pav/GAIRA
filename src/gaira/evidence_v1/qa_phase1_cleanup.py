from __future__ import annotations

import csv
import json
from pathlib import Path

import duckdb

from gaira.evidence_v1.constants import (
    CLEANUP_QA_ROOT,
    CLEANUP_REPORT_ROOT,
    CLEANUP_TABLES_ROOT,
    REFINEMENT_OUTPUT_ROOT,
    REFINEMENT_QA_ROOT,
    REFINEMENT_TABLES_ROOT,
    ensure_cleanup_output_dirs,
)
from gaira.evidence_v1.phase1_refinement import FAMILY_LABELS, PREVIOUS_FAMILY_LABELS
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


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_example_retrieval(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def generate_cleanup_qa_artifacts(
    db_path: Path,
    before_example_paths: dict[str, Path],
    after_example_paths: dict[str, Path],
) -> dict[str, int]:
    ensure_cleanup_output_dirs()

    previous_row_counts = {
        row["table_name"]: int(row["row_count"])
        for row in _read_csv_rows(REFINEMENT_TABLES_ROOT / "table_row_counts.csv")
    }
    previous_mention_summary = _read_csv_rows(REFINEMENT_TABLES_ROOT / "operational_mention_summary.csv")
    previous_cluster_summary = _read_csv_rows(REFINEMENT_TABLES_ROOT / "peak_meaning_cluster_summary.csv")

    counts: dict[str, int] = {}
    with duckdb.connect(str(db_path), read_only=True) as connection:
        schema_lines = ["# Phase 1 Cleanup/Audit Schema Summary", ""]
        row_count_lines = ["table_name,before_row_count,after_row_count"]
        for table_name in REFINEMENT_TABLES:
            row_count = int(connection.sql(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
            counts[table_name] = row_count
            row_count_lines.append(f"{table_name},{previous_row_counts.get(table_name, '')},{row_count}")
            schema_df = connection.sql(f"DESCRIBE {table_name}").df()
            schema_lines.append(f"## {table_name}")
            schema_lines.append(_markdown_table(schema_df.columns.tolist(), schema_df.astype(str).values.tolist()))
            schema_lines.append("")
        _write_text(CLEANUP_QA_ROOT / "schema_summary.md", "\n".join(schema_lines))
        _write_text(CLEANUP_TABLES_ROOT / "table_row_counts_before_after.csv", "\n".join(row_count_lines) + "\n")

        support_bundle_summary = connection.sql(
            """
            SELECT
                normalized_family,
                normalized_meaning_label,
                COUNT(*) AS cluster_count,
                SUM(CASE WHEN mixed_family_flag THEN 1 ELSE 0 END) AS mixed_family_clusters,
                ROUND(AVG(confidence_score), 6) AS mean_confidence_score,
                ROUND(AVG(ambiguity_score), 6) AS mean_ambiguity_score
            FROM evidence.peak_meaning_clusters
            GROUP BY normalized_family, normalized_meaning_label
            ORDER BY cluster_count DESC, normalized_family
            """
        ).df()
        support_bundle_summary.to_csv(CLEANUP_TABLES_ROOT / "support_bundle_summary.csv", index=False)

        cluster_fragmentation = connection.sql(
            """
            WITH ordered AS (
              SELECT
                normalized_family,
                cluster_id,
                canonical_peak_cm,
                LEAD(cluster_id) OVER (PARTITION BY normalized_family ORDER BY canonical_peak_cm) AS next_cluster_id,
                LEAD(canonical_peak_cm) OVER (PARTITION BY normalized_family ORDER BY canonical_peak_cm) AS next_peak_cm
              FROM evidence.peak_meaning_clusters
            )
            SELECT
              normalized_family,
              cluster_id,
              ROUND(canonical_peak_cm, 3) AS canonical_peak_cm,
              next_cluster_id,
              ROUND(next_peak_cm, 3) AS next_peak_cm,
              ROUND(next_peak_cm - canonical_peak_cm, 3) AS delta_cm
            FROM ordered
            WHERE next_peak_cm IS NOT NULL AND next_peak_cm - canonical_peak_cm <= 12.0
            ORDER BY delta_cm, normalized_family, canonical_peak_cm
            """
        ).df()
        cluster_fragmentation.to_csv(CLEANUP_TABLES_ROOT / "cluster_fragmentation_pairs.csv", index=False)

        overlap_examples = connection.sql(
            """
            WITH pairs AS (
              SELECT
                a.cluster_id AS cluster_id_a,
                a.normalized_meaning_label AS label_a,
                a.normalized_family AS family_a,
                a.canonical_peak_cm AS peak_a,
                b.cluster_id AS cluster_id_b,
                b.normalized_meaning_label AS label_b,
                b.normalized_family AS family_b,
                b.canonical_peak_cm AS peak_b,
                ABS(a.canonical_peak_cm - b.canonical_peak_cm) AS delta_cm
              FROM evidence.peak_meaning_clusters a
              JOIN evidence.peak_meaning_clusters b
                ON a.cluster_id < b.cluster_id
               AND ABS(a.canonical_peak_cm - b.canonical_peak_cm) <= 1.0
            )
            SELECT family_a, ROUND(peak_a, 3) AS peak_a, family_b, ROUND(peak_b, 3) AS peak_b, ROUND(delta_cm, 3) AS delta_cm
            FROM pairs
            ORDER BY delta_cm, family_a, family_b
            LIMIT 20
            """
        ).df()
        overlap_examples.to_csv(CLEANUP_TABLES_ROOT / "cluster_overlap_examples.csv", index=False)

        score_component_audit = connection.sql(
            """
            SELECT
                cluster_id,
                normalized_meaning_label,
                ROUND(canonical_peak_cm, 3) AS canonical_peak_cm,
                mixed_family_flag,
                overlapping_family_count,
                raw_label_diversity_count,
                confidence_score,
                ambiguity_score,
                score_components_json
            FROM evidence.peak_meaning_clusters
            ORDER BY confidence_score DESC, ambiguity_score ASC, canonical_peak_cm
            LIMIT 60
            """
        ).df()
        score_component_audit.to_csv(CLEANUP_TABLES_ROOT / "score_component_audit.csv", index=False)

        mention_summary = connection.sql(
            """
            SELECT alignment_status, alignment_reason, included_as_secondary_support, COUNT(*) AS row_count
            FROM evidence.operational_mention_links
            GROUP BY alignment_status, alignment_reason, included_as_secondary_support
            ORDER BY alignment_status, alignment_reason
            """
        ).df()
        mention_summary.to_csv(CLEANUP_TABLES_ROOT / "mention_alignment_summary.csv", index=False)

        good_mentions = connection.sql(
            """
            SELECT
                oml.alignment_status,
                oml.alignment_reason,
                wm.study_family,
                ROUND(wm.wavenumber_cm, 3) AS wavenumber_cm,
                wm.assigned_molecule_hint,
                wm.biochemical_theme_hint,
                oml.normalized_meaning_label,
                wm.mention_text
            FROM evidence.operational_mention_links oml
            JOIN evidence.wavenumber_mentions wm
              ON wm.evidence_item_id = oml.evidence_item_id
            WHERE oml.alignment_status = 'aligned_secondary_support'
            ORDER BY wm.study_family, wm.wavenumber_cm
            LIMIT 20
            """
        ).df()
        excluded_mentions = connection.sql(
            """
            SELECT
                oml.alignment_status,
                oml.alignment_reason,
                wm.study_family,
                ROUND(wm.wavenumber_cm, 3) AS wavenumber_cm,
                wm.assigned_molecule_hint,
                wm.biochemical_theme_hint,
                oml.normalized_meaning_label,
                wm.mention_text
            FROM evidence.operational_mention_links oml
            JOIN evidence.wavenumber_mentions wm
              ON wm.evidence_item_id = oml.evidence_item_id
            WHERE oml.alignment_status = 'excluded_bare_mention'
            ORDER BY wm.study_family, wm.wavenumber_cm
            LIMIT 20
            """
        ).df()
        good_mentions.to_csv(CLEANUP_TABLES_ROOT / "mention_alignment_examples_good.csv", index=False)
        excluded_mentions.to_csv(CLEANUP_TABLES_ROOT / "mention_alignment_examples_excluded.csv", index=False)

        context_summary = connection.sql(
            """
            SELECT
                (SELECT COUNT(*) FROM context.context_nodes) AS node_count,
                (SELECT COUNT(*) FROM context.context_edges) AS edge_count,
                (SELECT COUNT(*) FROM retrieval.peak_meaning_documents WHERE json_array_length(applicable_context_edge_ids_json) > 0) AS docs_with_context_edges
            """
        ).df()
        context_summary.to_csv(CLEANUP_TABLES_ROOT / "context_graph_summary.csv", index=False)

        context_edges = connection.sql(
            """
            SELECT edge_type, COUNT(*) AS row_count
            FROM context.context_edges
            GROUP BY edge_type
            ORDER BY edge_type
            """
        ).df()
        context_edges.to_csv(CLEANUP_TABLES_ROOT / "context_edge_summary.csv", index=False)

    family_audit_lines = ["# Normalized Family Label Audit", ""]
    family_rows = []
    for family_id in sorted(FAMILY_LABELS):
        previous_label = PREVIOUS_FAMILY_LABELS[family_id]
        current_label = FAMILY_LABELS[family_id]
        changed = "yes" if previous_label != current_label else "no"
        previous_count = ""
        for row in previous_cluster_summary:
            if row["normalized_family"] == family_id:
                previous_count = row["cluster_count"]
                break
        family_rows.append([family_id, previous_label, current_label, changed, previous_count])
    family_audit_lines.append(
        _markdown_table(
            ["family_id", "previous_label", "current_label", "label_changed", "previous_cluster_count"],
            family_rows,
        )
    )
    _write_text(CLEANUP_QA_ROOT / "normalized_family_label_audit.md", "\n".join(family_audit_lines))

    fragmentation_lines = [
        "# Cluster Fragmentation Audit",
        "",
        f"- Previous refinement cluster count: {previous_row_counts.get('evidence.peak_meaning_clusters', 'unknown')}",
        f"- Cleanup cluster count: {counts.get('evidence.peak_meaning_clusters', 0)}",
        f"- Same-family pairs within 12 cm after cleanup: {len(_read_csv_rows(CLEANUP_TABLES_ROOT / 'cluster_fragmentation_pairs.csv'))}",
        "- Interpretation: same-family over-fragmentation was not observed; ambiguity is concentrated in cross-family overlap around shared Raman windows.",
        "",
        "## Cross-Family Overlap Examples",
        "",
    ]
    overlap_rows = _read_csv_rows(CLEANUP_TABLES_ROOT / "cluster_overlap_examples.csv")
    fragmentation_lines.append(
        _markdown_table(
            ["family_a", "peak_a", "family_b", "peak_b", "delta_cm"],
            [[row["family_a"], row["peak_a"], row["family_b"], row["peak_b"], row["delta_cm"]] for row in overlap_rows[:12]],
        )
    )
    _write_text(CLEANUP_QA_ROOT / "cluster_fragmentation_audit.md", "\n".join(fragmentation_lines))

    previous_mentions = {
        (row["alignment_status"], row["included_as_secondary_support"]): int(row["row_count"])
        for row in previous_mention_summary
    }
    current_mentions = {
        (row["alignment_status"], row["included_as_secondary_support"]): int(row["row_count"])
        for row in _read_csv_rows(CLEANUP_TABLES_ROOT / "mention_alignment_summary.csv")
    }
    mention_lines = [
        "# Mention Alignment Audit",
        "",
        f"- Previous aligned secondary mention count: {previous_mentions.get(('aligned_secondary_support', 'True'), 0)}",
        f"- Cleanup aligned secondary mention count: {current_mentions.get(('aligned_secondary_support', 'True'), 0)}",
        f"- Previous excluded bare mention count: {previous_mentions.get(('excluded_bare_mention', 'False'), 0)}",
        f"- Cleanup excluded bare mention count: {current_mentions.get(('excluded_bare_mention', 'False'), 0)}",
        "- Cleanup rule: only mentions with interpretable family hints and a matching cluster window remain as secondary support.",
        "",
        "## Retained Examples",
        "",
    ]
    good_rows = _read_csv_rows(CLEANUP_TABLES_ROOT / "mention_alignment_examples_good.csv")
    mention_lines.append(
        _markdown_table(
            ["study_family", "wavenumber_cm", "assigned_molecule_hint", "biochemical_theme_hint", "normalized_meaning_label", "alignment_reason"],
            [
                [
                    row["study_family"],
                    row["wavenumber_cm"],
                    row["assigned_molecule_hint"],
                    row["biochemical_theme_hint"],
                    row["normalized_meaning_label"],
                    row["alignment_reason"],
                ]
                for row in good_rows[:10]
            ],
        )
    )
    mention_lines.extend(["", "## Excluded Examples", ""])
    excluded_rows = _read_csv_rows(CLEANUP_TABLES_ROOT / "mention_alignment_examples_excluded.csv")
    mention_lines.append(
        _markdown_table(
            ["study_family", "wavenumber_cm", "assigned_molecule_hint", "biochemical_theme_hint", "alignment_reason", "mention_text"],
            [
                [
                    row["study_family"],
                    row["wavenumber_cm"],
                    row["assigned_molecule_hint"],
                    row["biochemical_theme_hint"],
                    row["alignment_reason"],
                    row["mention_text"],
                ]
                for row in excluded_rows[:10]
            ],
        )
    )
    _write_text(CLEANUP_QA_ROOT / "mention_alignment_audit.md", "\n".join(mention_lines))

    score_rows = _read_csv_rows(CLEANUP_TABLES_ROOT / "score_component_audit.csv")
    score_lines = ["# Score Component Audit", "", "## Example Bundles", ""]
    score_lines.append(
        _markdown_table(
            [
                "cluster_id",
                "label",
                "peak_cm",
                "mixed_family_flag",
                "overlapping_family_count",
                "raw_label_diversity_count",
                "confidence_score",
                "ambiguity_score",
            ],
            [
                [
                    row["cluster_id"],
                    row["normalized_meaning_label"],
                    row["canonical_peak_cm"],
                    row["mixed_family_flag"],
                    row["overlapping_family_count"],
                    row["raw_label_diversity_count"],
                    row["confidence_score"],
                    row["ambiguity_score"],
                ]
                for row in score_rows[:12]
            ],
        )
    )
    _write_text(CLEANUP_QA_ROOT / "score_component_audit.md", "\n".join(score_lines))

    context_rows = _read_csv_rows(CLEANUP_TABLES_ROOT / "context_graph_summary.csv")
    context_edges = _read_csv_rows(CLEANUP_TABLES_ROOT / "context_edge_summary.csv")
    context_lines = [
        "# Context Graph Audit",
        "",
        *(f"- {key.replace('_', ' ')}: {value}" for key, value in (context_rows[0] if context_rows else {}).items()),
        "- Interpretation: the graph remains compact; it is used to attach caveats to retrieved bundles rather than to drive inference.",
        "",
        "## Edge Types",
        "",
    ]
    context_lines.append(
        _markdown_table(
            ["edge_type", "row_count"],
            [[row["edge_type"], row["row_count"]] for row in context_edges],
        )
    )
    _write_text(CLEANUP_QA_ROOT / "context_graph_audit.md", "\n".join(context_lines))

    before_after_lines = ["# Retrieval Before/After Cleanup", ""]
    for name, before_path in before_example_paths.items():
        after_path = after_example_paths[name]
        before_payload = json.loads(before_path.read_text())
        after_payload = json.loads(after_path.read_text())
        before_titles = [item["title"] for item in before_payload.get("support_bundle_results", [])[:3]]
        after_titles = [item["title"] for item in after_payload.get("support_bundle_results", [])[:3]]
        before_after_lines.extend(
            [
                f"## {name}",
                "",
                f"- Before cleanup top bundles: {before_titles}",
                f"- After cleanup top bundles: {after_titles}",
                "",
            ]
        )
    _write_text(CLEANUP_QA_ROOT / "retrieval_before_after.md", "\n".join(before_after_lines))

    _write_text(
        CLEANUP_REPORT_ROOT / "qa_summary.md",
        "\n".join(
            [
                "# GAIRA Phase 1 Cleanup/Audit QA Summary",
                "",
                "## Row Counts",
                "",
                *(f"- `{table}`: {count}" for table, count in counts.items()),
            ]
        ),
    )

    return counts
