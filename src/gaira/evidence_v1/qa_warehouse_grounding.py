from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from gaira.evidence_v1.constants import (
    WAREHOUSE_QA_ROOT,
    WAREHOUSE_REPORT_ROOT,
    WAREHOUSE_TABLES_ROOT,
    ensure_warehouse_output_dirs,
)
from gaira.evidence_v1.warehouse_grounding import GROUNDING_DATASET_ROUTES


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    columns = df.columns.tolist()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(value) else str(value) for value in row.tolist()) + " |")
    return "\n".join(lines)


def generate_warehouse_grounding_qa(db_path: Path, build_counts: dict[str, int]) -> dict[str, int]:
    ensure_warehouse_output_dirs()
    with duckdb.connect(str(db_path), read_only=True) as connection:
        warehouse_registry = connection.sql(
            """
            SELECT *
            FROM registry.warehouse_sources
            ORDER BY source_family, source_id
            """
        ).df()
        warehouse_registry.to_csv(WAREHOUSE_TABLES_ROOT / "warehouse_source_registry.csv", index=False)

        warehouse_family_summary = connection.sql(
            """
            SELECT
                source_family,
                COUNT(*) AS source_count,
                SUM(CASE WHEN structured_spectral_data_available THEN 1 ELSE 0 END) AS spectral_source_count,
                SUM(CASE WHEN structured_peak_assignments_available THEN 1 ELSE 0 END) AS assignment_source_count,
                SUM(CASE WHEN digitization_required THEN 1 ELSE 0 END) AS digitization_needed_count
            FROM registry.warehouse_sources
            GROUP BY source_family
            ORDER BY source_family
            """
        ).df()
        warehouse_family_summary.to_csv(WAREHOUSE_TABLES_ROOT / "warehouse_family_summary.csv", index=False)

        for family_name, file_name in [
            ("reference_molecule", "reference_molecule_sources.csv"),
            ("serum_grounding", "serum_grounding_sources.csv"),
            ("ev_grounding", "ev_grounding_sources.csv"),
            ("disease_or_stress_paper", "disease_or_stress_paper_sources.csv"),
        ]:
            connection.sql(
                """
                SELECT *
                FROM registry.warehouse_sources
                WHERE source_family = ?
                ORDER BY source_id
                """,
                params=[family_name],
            ).df().to_csv(WAREHOUSE_TABLES_ROOT / file_name, index=False)

        newly_added_grounding_sources = connection.sql(
            """
            SELECT
                ws.source_id,
                ws.source_name,
                ws.source_family,
                ws.sample_scope,
                COUNT(DISTINCT gse.grounding_record_id) AS summary_spectra_added,
                COUNT(DISTINCT pae.assignment_record_id) AS peak_assignments_added
            FROM registry.warehouse_sources ws
            LEFT JOIN evidence.grounding_spectrum_evidence gse
              ON gse.source_id = ws.source_id
            LEFT JOIN evidence.peak_assignment_evidence pae
              ON pae.source_id = ws.source_id
             AND pae.assignment_origin IN ('reference_grounding_peak', 'serum_grounding_peak')
            WHERE ws.source_id IN ({placeholders})
            GROUP BY 1, 2, 3, 4
            ORDER BY ws.source_family, ws.source_id
            """.format(placeholders=", ".join(["?"] * len(GROUNDING_DATASET_ROUTES))),
            params=list(GROUNDING_DATASET_ROUTES),
        ).df()
        newly_added_grounding_sources.to_csv(WAREHOUSE_TABLES_ROOT / "newly_added_grounding_sources.csv", index=False)

        grounding_peak_support_summary = connection.sql(
            """
            SELECT
                gse.source_id,
                ws.source_family,
                gse.modality,
                COUNT(*) AS summary_spectra_count,
                SUM(gse.peak_count) AS detected_peak_count,
                COUNT(DISTINCT gse.class_label) AS class_count,
                ROUND(AVG(gse.peak_count), 3) AS mean_peaks_per_summary
            FROM evidence.grounding_spectrum_evidence gse
            JOIN registry.warehouse_sources ws
              ON ws.source_id = gse.source_id
            GROUP BY 1, 2, 3
            ORDER BY ws.source_family, gse.source_id
            """
        ).df()
        grounding_peak_support_summary.to_csv(WAREHOUSE_TABLES_ROOT / "grounding_peak_support_summary.csv", index=False)

        strengthened_clusters = int(
            connection.sql(
                """
                WITH grounding_clusters AS (
                    SELECT DISTINCT pms.cluster_id
                    FROM evidence.peak_meaning_support pms
                    WHERE pms.source_id IN ({placeholders})
                ),
                mixed_clusters AS (
                    SELECT gc.cluster_id
                    FROM grounding_clusters gc
                    WHERE EXISTS (
                        SELECT 1
                        FROM evidence.peak_meaning_support pms
                        WHERE pms.cluster_id = gc.cluster_id
                          AND pms.source_id NOT IN ({placeholders})
                    )
                )
                SELECT COUNT(*) FROM mixed_clusters
                """.format(placeholders=", ".join(["?"] * len(CANONICAL_IDS := list(GROUNDING_DATASET_ROUTES)))),
                params=CANONICAL_IDS + CANONICAL_IDS,
            ).fetchone()[0]
        )
        strengthened_patterns = int(
            connection.sql(
                """
                WITH grounding_clusters AS (
                    SELECT DISTINCT pms.cluster_id
                    FROM evidence.peak_meaning_support pms
                    WHERE pms.source_id IN ({placeholders})
                ),
                mixed_clusters AS (
                    SELECT gc.cluster_id
                    FROM grounding_clusters gc
                    WHERE EXISTS (
                        SELECT 1
                        FROM evidence.peak_meaning_support pms
                        WHERE pms.cluster_id = gc.cluster_id
                          AND pms.source_id NOT IN ({placeholders})
                    )
                )
                SELECT COUNT(DISTINCT apm.pattern_id)
                FROM evidence.assignment_pattern_members apm
                JOIN mixed_clusters mc
                  ON mc.cluster_id = apm.cluster_id
                """.format(placeholders=", ".join(["?"] * len(CANONICAL_IDS))),
                params=CANONICAL_IDS + CANONICAL_IDS,
            ).fetchone()[0]
        )

        additions_summary = pd.DataFrame(
            [
                {"metric": "grounding_summary_spectra_added", "count": build_counts.get("grounding_summary_spectra_added", 0)},
                {"metric": "grounding_peak_assignments_added", "count": build_counts.get("grounding_peak_assignments_added", 0)},
                {"metric": "grounding_feature_rows_added", "count": build_counts.get("grounding_feature_rows_added", 0)},
                {"metric": "clusters_strengthened_by_grounding", "count": strengthened_clusters},
                {"metric": "patterns_strengthened_by_grounding", "count": strengthened_patterns},
            ]
        )
        additions_summary.to_csv(WAREHOUSE_TABLES_ROOT / "structured_evidence_additions_summary.csv", index=False)

        pattern_family_summary = connection.sql(
            """
            SELECT
                ap.normalized_family,
                COUNT(*) AS pattern_count,
                ROUND(AVG(ap.total_member_count), 3) AS avg_member_count,
                ROUND(AVG(ap.core_member_count), 3) AS avg_core_member_count,
                ROUND(AVG(ap.confidence_score), 6) AS avg_pattern_confidence,
                ROUND(AVG(ap.ambiguity_score), 6) AS avg_pattern_ambiguity
            FROM evidence.assignment_patterns ap
            GROUP BY ap.normalized_family
            ORDER BY ap.normalized_family
            """
        ).df()
        pattern_family_summary.to_csv(WAREHOUSE_TABLES_ROOT / "pattern_family_summary.csv", index=False)

        pattern_source_composition_summary = connection.sql(
            """
            SELECT
                ap.pattern_id,
                ap.pattern_label,
                ap.normalized_family,
                ws.source_family,
                COUNT(*) AS support_rows,
                COUNT(DISTINCT pms.source_id) AS distinct_sources
            FROM evidence.assignment_patterns ap
            JOIN evidence.assignment_pattern_members apm
              ON apm.pattern_id = ap.pattern_id
            JOIN evidence.peak_meaning_support pms
              ON pms.cluster_id = apm.cluster_id
            LEFT JOIN registry.warehouse_sources ws
              ON ws.source_id = pms.source_id
            GROUP BY 1, 2, 3, 4
            ORDER BY ap.pattern_id, ws.source_family
            """
        ).df()
        pattern_source_composition_summary.to_csv(WAREHOUSE_TABLES_ROOT / "pattern_source_composition_summary.csv", index=False)

        same_family_multi_pattern_examples = connection.sql(
            """
            WITH multi AS (
                SELECT normalized_family
                FROM evidence.assignment_patterns
                GROUP BY normalized_family
                HAVING COUNT(*) > 1
            )
            SELECT
                ap.normalized_family,
                ap.pattern_id,
                ap.pattern_label,
                ap.total_member_count,
                ap.core_member_count,
                ap.source_diversity,
                ap.confidence_score,
                ap.ambiguity_score,
                ap.notes
            FROM evidence.assignment_patterns ap
            JOIN multi
              ON multi.normalized_family = ap.normalized_family
            ORDER BY ap.normalized_family, ap.pattern_id
            """
        ).df()
        same_family_multi_pattern_examples.to_csv(WAREHOUSE_TABLES_ROOT / "same_family_multi_pattern_examples.csv", index=False)

        source_family_counts = dict(zip(warehouse_family_summary["source_family"], warehouse_family_summary["source_count"]))
        multi_pattern_cases = int(same_family_multi_pattern_examples["normalized_family"].nunique()) if not same_family_multi_pattern_examples.empty else 0
        ready_for_scaling = (
            source_family_counts.get("reference_molecule", 0) > 0
            and source_family_counts.get("serum_grounding", 0) > 0
            and build_counts.get("grounding_peak_assignments_added", 0) > 0
        )

    implementation_lines = [
        "# Warehouse Backbone Implementation Note",
        "",
        "## Routing",
        "",
        "- Added `registry.warehouse_sources` as the typed warehouse-routing registry.",
        "- Every active evidence source is now routed into one of `reference_molecule`, `serum_grounding`, `ev_grounding`, or `disease_or_stress_paper`.",
        "- Context-only rows remain outside this warehouse registry.",
        "",
        "## Grounding Expansion Beyond RamanBioLib",
        "",
        "- Bridged canonical summary spectra already present in `main.grounding_class_summary` into `evidence.grounding_spectrum_evidence`.",
        "- Added structured peak-assignment rows from detected peaks on canonical summary spectra for `adenine_sers_control`, `amino_acid_raman_grounding`, `metabolite_sers63_support`, and `serum_ag_colloids_grounding`.",
        "- Kept `serum_ag_colloids_literature_grounding` routed as `disease_or_stress_paper` but did not trigger new broad paper parsing.",
        "",
        "## Structured Evidence Behavior",
        "",
        "- New grounding spectra generate new structured evidence rows because they contribute new peak-support objects.",
        "- Existing cluster/pattern meanings were strengthened through additional support rows rather than duplicated blindly.",
        "- Pure reference and serum-grounding sources remain explicitly separated in routing metadata.",
        "",
        "## Pattern Handling",
        "",
        "- Patterns were rebuilt from the expanded scaffold after the grounding bridge.",
        "- Multiple patterns per family are allowed when the co-support graph separates them; otherwise a family remains a single broader pattern.",
        "",
        "## Remaining Work Before Broad Literature Scaling",
        "",
        "- EV-specific grounding remains absent in the current GAIRA layout.",
        "- Disease/stress paper support is still dominated by already structured manuscript regex evidence plus a limited literature-grounding package.",
        "- Broad PDF/SI parsing and figure digitization remain deferred.",
        "",
    ]
    _write_text(WAREHOUSE_REPORT_ROOT / "implementation_note.md", "\n".join(implementation_lines))

    assessment_lines = [
        "# Current State Assessment",
        "",
        f"- Warehouse sources routed: {int(warehouse_registry.shape[0])}",
        f"- Source-family counts: {json.dumps(source_family_counts, sort_keys=True)}",
        f"- Grounding summary spectra added: {build_counts.get('grounding_summary_spectra_added', 0)}",
        f"- Grounding peak assignments added: {build_counts.get('grounding_peak_assignments_added', 0)}",
        f"- Clusters strengthened by mixed old+new support: {strengthened_clusters}",
        f"- Patterns strengthened by mixed old+new support: {strengthened_patterns}",
        f"- Same-family multi-pattern cases: {multi_pattern_cases}",
        "",
        "## Assessment",
        "",
        "- `reference_molecule`, `serum_grounding`, and `disease_or_stress_paper` are now explicitly separated in the warehouse registry.",
        "- `ev_grounding` remains empty in the current GAIRA assets; this was confirmed rather than backfilled artificially.",
        "- Patterns now emerge from a broader scaffold than RamanBioLib alone because new grounding-derived peak assignments are present in the cluster/pattern rebuild.",
        f"- Ready for broad literature scaling: {'not yet' if not ready_for_scaling else 'partially, but still needs disciplined paper-scale routing and QA'}",
        "- Top weaknesses: no EV grounding source, disease/stress paper layer still relatively sparse, and grounding peak assignments are summary-spectrum-derived rather than manually curated assignments.",
        "",
    ]
    _write_text(WAREHOUSE_REPORT_ROOT / "current_state_assessment.md", "\n".join(assessment_lines))

    schema_summary_lines = [
        "# Warehouse Grounding Schema Summary",
        "",
        "- `registry.warehouse_sources`",
        "- `evidence.grounding_spectrum_evidence`",
        "- rebuilt `evidence.peak_meaning_clusters`",
        "- rebuilt `evidence.assignment_patterns`",
        "",
    ]
    _write_text(WAREHOUSE_QA_ROOT / "schema_summary.md", "\n".join(schema_summary_lines))

    return {
        "warehouse_source_count": int(warehouse_registry.shape[0]),
        "grounding_spectrum_evidence_count": int(build_counts.get("grounding_summary_spectra_added", 0)),
        "grounding_peak_assignment_count": int(build_counts.get("grounding_peak_assignments_added", 0)),
        "same_family_multi_pattern_cases": multi_pattern_cases,
        "clusters_strengthened_by_grounding": strengthened_clusters,
        "patterns_strengthened_by_grounding": strengthened_patterns,
    }
