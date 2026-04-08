from __future__ import annotations

import csv
import json
from pathlib import Path

import duckdb

from gaira.evidence_v1.assignment_patterns import build_assignment_patterns
from gaira.evidence_v1.constants import (
    DB_PATH,
    ONTOLOGY_MOTIF_OUTPUT_ROOT,
    ONTOLOGY_MOTIF_REPORT_ROOT,
    ONTOLOGY_MOTIF_TABLES_ROOT,
    ensure_ontology_motif_output_dirs,
)
from gaira.evidence_v1.schema import initialize_schema, reset_pattern_tables


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_ontology_aligned_motif_summary(db_path: str = str(DB_PATH)) -> dict:
    ensure_ontology_motif_output_dirs()
    with duckdb.connect(db_path) as connection:
        initialize_schema(connection)
        before_counts = connection.sql(
            """
            SELECT
                (SELECT COUNT(*) FROM ontology.pattern_ontology_mappings) AS motif_count_before,
                (SELECT COUNT(DISTINCT broader_family) FROM ontology.pattern_ontology_mappings) AS broader_family_count_before
            """
        ).df().to_dict(orient="records")[0]

        reset_pattern_tables(connection)
        initialize_schema(connection)
        build_counts = build_assignment_patterns(connection)

        after_counts = connection.sql(
            """
            SELECT
                COUNT(*) AS motif_count_after,
                COUNT(DISTINCT normalized_subfamily) AS subfamily_count_after,
                COUNT(DISTINCT broader_family) AS broader_family_count_after
            FROM evidence.assignment_patterns
            """
        ).df().to_dict(orient="records")[0]

        motif_count_before_after = [{**before_counts, **after_counts}]
        _write_csv(
            ONTOLOGY_MOTIF_TABLES_ROOT / "motif_count_before_after.csv",
            motif_count_before_after,
            list(motif_count_before_after[0].keys()),
        )

        motifs_per_subfamily = connection.sql(
            """
            SELECT
                normalized_subfamily,
                broader_family,
                meaning_class,
                confounder_subclass,
                spectral_region,
                COUNT(*) AS motif_count,
                AVG(core_member_count) AS avg_core_members,
                AVG(total_member_count) AS avg_total_members,
                AVG(coherence_score) AS avg_coherence,
                AVG(confidence_score) AS avg_confidence
            FROM evidence.assignment_patterns
            GROUP BY 1,2,3,4,5
            ORDER BY motif_count DESC, normalized_subfamily
            """
        ).df().to_dict(orient="records")
        _write_csv(
            ONTOLOGY_MOTIF_TABLES_ROOT / "motifs_per_subfamily.csv",
            motifs_per_subfamily,
            list(motifs_per_subfamily[0].keys()),
        )

        motifs_per_broader_family = connection.sql(
            """
            SELECT
                broader_family,
                meaning_class,
                COUNT(*) AS motif_count,
                COUNT(DISTINCT normalized_subfamily) AS distinct_subfamilies,
                AVG(total_member_count) AS avg_total_members
            FROM evidence.assignment_patterns
            GROUP BY 1,2
            ORDER BY motif_count DESC, broader_family
            """
        ).df().to_dict(orient="records")
        _write_csv(
            ONTOLOGY_MOTIF_TABLES_ROOT / "motifs_per_broader_family.csv",
            motifs_per_broader_family,
            list(motifs_per_broader_family[0].keys()),
        )

        motif_examples = connection.sql(
            """
            SELECT
                ap.pattern_id,
                ap.pattern_label,
                ap.normalized_subfamily,
                ap.broader_family,
                ap.meaning_class,
                ap.confounder_subclass,
                ap.spectral_region,
                ap.core_member_count,
                ap.total_member_count,
                ap.confidence_score,
                ap.coherence_score,
                STRING_AGG(
                    CASE
                        WHEN apm.member_role = 'core' THEN CAST(ROUND(apm.canonical_peak_cm, 1) AS VARCHAR)
                        ELSE NULL
                    END,
                    ', '
                    ORDER BY apm.canonical_peak_cm
                ) AS core_peaks_cm,
                STRING_AGG(
                    CASE
                        WHEN apm.member_role = 'supporting' THEN CAST(ROUND(apm.canonical_peak_cm, 1) AS VARCHAR)
                        ELSE NULL
                    END,
                    ', '
                    ORDER BY apm.canonical_peak_cm
                ) AS supporting_peaks_cm
            FROM evidence.assignment_patterns ap
            JOIN evidence.assignment_pattern_members apm
              ON ap.pattern_id = apm.pattern_id
            GROUP BY 1,2,3,4,5,6,7,8,9,10,11
            ORDER BY ap.meaning_class, ap.normalized_subfamily, ap.pattern_id
            LIMIT 24
            """
        ).df().to_dict(orient="records")
        _write_csv(
            ONTOLOGY_MOTIF_TABLES_ROOT / "motif_examples_refined.csv",
            motif_examples,
            list(motif_examples[0].keys()),
        )

        confounder_motif_summary = connection.sql(
            """
            SELECT
                pattern_id,
                pattern_label,
                normalized_subfamily,
                confounder_subclass,
                spectral_region,
                core_member_count,
                total_member_count,
                confidence_score
            FROM evidence.assignment_patterns
            WHERE meaning_class = 'confounder_signal'
            ORDER BY pattern_id
            """
        ).df().to_dict(orient="records")
        _write_csv(
            ONTOLOGY_MOTIF_TABLES_ROOT / "confounder_motif_summary.csv",
            confounder_motif_summary,
            list(confounder_motif_summary[0].keys()) if confounder_motif_summary else ["pattern_id"],
        )

        spectral_region_motif_summary = connection.sql(
            """
            SELECT
                spectral_region,
                meaning_class,
                COUNT(*) AS motif_count,
                COUNT(DISTINCT normalized_subfamily) AS distinct_subfamilies
            FROM evidence.assignment_patterns
            GROUP BY 1,2
            ORDER BY spectral_region, meaning_class
            """
        ).df().to_dict(orient="records")
        _write_csv(
            ONTOLOGY_MOTIF_TABLES_ROOT / "spectral_region_motif_summary.csv",
            spectral_region_motif_summary,
            list(spectral_region_motif_summary[0].keys()),
        )

        reference_grounding_summary = connection.sql(
            """
            WITH reference_sources AS (
                SELECT DISTINCT source_id, source_name, source_kind, parent_dataset_id
                FROM registry.warehouse_sources
                WHERE source_family = 'reference_molecule'
            ),
            grounding_rows AS (
                SELECT COUNT(*) AS grounding_spectrum_rows
                FROM evidence.grounding_spectrum_evidence
                WHERE source_family = 'reference_molecule'
            ),
            bridge_rows AS (
                SELECT COUNT(*) AS reference_bridge_spectrum_rows
                FROM evidence.reference_spectrum_evidence
            ),
            peak_rows AS (
                SELECT COUNT(*) AS structured_reference_peak_assignments
                FROM evidence.peak_assignment_evidence
                WHERE assignment_origin = 'reference_grounding_peak'
            )
            SELECT
                (SELECT COUNT(*) FROM reference_sources) AS reference_molecule_sources,
                (SELECT grounding_spectrum_rows FROM grounding_rows) AS grounding_spectrum_rows,
                (SELECT reference_bridge_spectrum_rows FROM bridge_rows) AS reference_bridge_spectrum_rows,
                ((SELECT grounding_spectrum_rows FROM grounding_rows) + (SELECT reference_bridge_spectrum_rows FROM bridge_rows)) AS total_reference_spectrum_rows,
                (SELECT structured_reference_peak_assignments FROM peak_rows) AS structured_reference_peak_assignments,
                STRING_AGG(
                    DISTINCT CASE
                        WHEN COALESCE(parent_dataset_id, source_id) = '' THEN NULL
                        ELSE COALESCE(parent_dataset_id, source_id)
                    END,
                    ', '
                    ORDER BY CASE
                        WHEN COALESCE(parent_dataset_id, source_id) = '' THEN NULL
                        ELSE COALESCE(parent_dataset_id, source_id)
                    END
                ) AS major_reference_grounding_datasets
            FROM reference_sources
            """
        ).df().to_dict(orient="records")
        _write_csv(
            ONTOLOGY_MOTIF_TABLES_ROOT / "reference_grounding_summary.csv",
            reference_grounding_summary,
            list(reference_grounding_summary[0].keys()),
        )

        paper_structured_evidence_summary = connection.sql(
            """
            WITH paper_sources AS (
                SELECT DISTINCT source_id, source_name, source_kind
                FROM registry.warehouse_sources
                WHERE source_family = 'disease_or_stress_paper'
            ),
            incorporated_papers AS (
                SELECT DISTINCT source_id
                FROM evidence.peak_assignment_evidence
                WHERE source_id IN (SELECT source_id FROM paper_sources)
            ),
            paper_assignments AS (
                SELECT
                    source_id,
                    COUNT(*) AS structured_assignments,
                    SUM(CASE WHEN extraction_method = 'digitized_figure' THEN 1 ELSE 0 END) AS figure_assignments,
                    SUM(CASE WHEN extraction_method = 'text_assignment' THEN 1 ELSE 0 END) AS text_assignments,
                    SUM(CASE WHEN extraction_method = 'text_regex' THEN 1 ELSE 0 END) AS regex_assignments
                FROM evidence.peak_assignment_evidence
                WHERE source_id IN (SELECT source_id FROM paper_sources)
                GROUP BY source_id
            )
            SELECT
                (SELECT COUNT(*) FROM incorporated_papers) AS processed_papers_in_structured_evidence,
                (SELECT COUNT(*) FROM paper_sources) AS disease_or_stress_paper_sources,
                COALESCE(SUM(structured_assignments), 0) AS extracted_structured_assignments_from_papers,
                COALESCE(SUM(figure_assignments), 0) AS figure_derived_assignments,
                COALESCE(SUM(text_assignments), 0) AS text_derived_assignments,
                COALESCE(SUM(regex_assignments), 0) AS regex_derived_assignments
            FROM paper_assignments
            """
        ).df().to_dict(orient="records")
        _write_csv(
            ONTOLOGY_MOTIF_TABLES_ROOT / "paper_structured_evidence_summary.csv",
            paper_structured_evidence_summary,
            list(paper_structured_evidence_summary[0].keys()),
        )

        supplementary_coverage_summary = connection.sql(
            """
            WITH source_base AS (
                SELECT
                    source_id,
                    MIN(source_name) AS source_name,
                    STRING_AGG(DISTINCT source_kind, ', ' ORDER BY source_kind) AS source_kinds,
                    STRING_AGG(DISTINCT COALESCE(repo_origin, ''), ' || ' ORDER BY COALESCE(repo_origin, '')) AS repo_origins
                FROM registry.warehouse_sources
                WHERE source_family = 'disease_or_stress_paper'
                GROUP BY source_id
            )
            SELECT
                source_id,
                source_name,
                CASE
                    WHEN lower(COALESCE(repo_origins, '')) LIKE '%supp%' OR lower(COALESCE(source_name, '')) LIKE '%supp%' OR lower(COALESCE(source_name, '')) LIKE '%supplement%'
                    THEN 'supplementary_or_si_detected'
                    ELSE 'no_structured_si_detected'
                END AS supplementary_status,
                CASE
                    WHEN EXISTS (
                        SELECT 1
                        FROM evidence.peak_assignment_evidence p
                        WHERE p.source_id = source_base.source_id
                          AND p.extraction_method = 'digitized_figure'
                    )
                    THEN 'yes'
                    ELSE 'no'
                END AS has_figure_derived_evidence,
                CASE
                    WHEN EXISTS (
                        SELECT 1
                        FROM evidence.peak_assignment_evidence p
                        WHERE p.source_id = source_base.source_id
                    )
                    THEN 'yes'
                    ELSE 'no'
                END AS has_structured_assignments
            FROM source_base
            ORDER BY source_id
            """
        ).df().to_dict(orient="records")
        _write_csv(
            ONTOLOGY_MOTIF_TABLES_ROOT / "supplementary_coverage_summary.csv",
            supplementary_coverage_summary,
            list(supplementary_coverage_summary[0].keys()),
        )

        warehouse_processing_coverage = connection.sql(
            """
            WITH source_base AS (
                SELECT DISTINCT
                    ws.source_id,
                    ws.source_name,
                    ws.source_family,
                    ws.source_kind,
                    ws.biosample_type,
                    ws.modality,
                    ws.repo_origin
                FROM registry.warehouse_sources ws
            ),
            assignment_counts AS (
                SELECT
                    source_id,
                    COUNT(*) AS assignment_rows,
                    SUM(CASE WHEN extraction_method = 'digitized_figure' THEN 1 ELSE 0 END) AS figure_rows,
                    SUM(CASE WHEN extraction_method = 'text_assignment' THEN 1 ELSE 0 END) AS text_rows
                FROM evidence.peak_assignment_evidence
                GROUP BY source_id
            ),
            evidence_counts AS (
                SELECT source_id, COUNT(*) AS evidence_item_rows
                FROM evidence.evidence_items
                GROUP BY source_id
            )
            SELECT
                sb.source_id,
                sb.source_name,
                sb.source_family,
                sb.source_kind,
                sb.biosample_type,
                sb.modality,
                COALESCE(ec.evidence_item_rows, 0) AS evidence_item_rows,
                COALESCE(ac.assignment_rows, 0) AS structured_assignment_rows,
                COALESCE(ac.figure_rows, 0) AS figure_assignment_rows,
                COALESCE(ac.text_rows, 0) AS text_assignment_rows,
                CASE
                    WHEN COALESCE(ac.assignment_rows, 0) > 0 THEN 'processed_into_structured_evidence'
                    WHEN sb.source_kind = 'digitization_queue' THEN 'queued_only'
                    ELSE 'registered_without_structured_assignments'
                END AS coverage_status
            FROM source_base sb
            LEFT JOIN assignment_counts ac ON ac.source_id = sb.source_id
            LEFT JOIN evidence_counts ec ON ec.source_id = sb.source_id
            ORDER BY sb.source_family, sb.source_id, sb.source_kind
            """
        ).df().to_dict(orient="records")
        _write_csv(
            ONTOLOGY_MOTIF_TABLES_ROOT / "warehouse_processing_coverage.csv",
            warehouse_processing_coverage,
            list(warehouse_processing_coverage[0].keys()),
        )

    implementation_note = "\n".join(
        [
            "# Implementation Note",
            "",
            "This pass rebuilds motifs from ontology-aware cluster mappings rather than the older broad-family-only grouping.",
            "",
            "## What Changed",
            "",
            "- Pattern grouping key is now `(normalized_subfamily, broader_family, meaning_class, confounder_subclass, spectral_region)`.",
            "- Confounder motifs are kept separate from biological motifs.",
            "- Carbonyl and high-wavenumber motifs are allowed to exist without being forced into fingerprint-region motif thresholds.",
            "- Multiple motifs can now exist within the same broader family because subfamily identity is primary.",
            "",
            "## Warehouse Summary Scope",
            "",
            "- Reference grounding counts come from `warehouse_sources`, `grounding_spectrum_evidence`, and `peak_assignment_evidence`.",
            "- Paper counts come from disease/stress paper sources with actual structured assignment rows in the live DB.",
            "- Supplementary coverage is reported conservatively: only explicitly detectable structured SI/supplementary artifacts are counted.",
            "",
        ]
    )
    (ONTOLOGY_MOTIF_REPORT_ROOT / "implementation_note.md").write_text(implementation_note + "\n", encoding="utf-8")

    current_state_summary = "\n".join(
        [
            "# Current State Summary",
            "",
            f"- Motifs before rebuild: `{motif_count_before_after[0]['motif_count_before']}`",
            f"- Motifs after ontology-aligned rebuild: `{motif_count_before_after[0]['motif_count_after']}`",
            f"- Reference grounding sources in warehouse: `{reference_grounding_summary[0]['reference_molecule_sources']}`",
            f"- Grounding spectrum rows from reference datasets: `{reference_grounding_summary[0]['grounding_spectrum_rows']}`",
            f"- RamanBioLib/reference-bridge spectrum rows: `{reference_grounding_summary[0]['reference_bridge_spectrum_rows']}`",
            f"- Total reference spectrum rows: `{reference_grounding_summary[0]['total_reference_spectrum_rows']}`",
            f"- Structured reference peak assignments: `{reference_grounding_summary[0]['structured_reference_peak_assignments']}`",
            f"- Processed papers currently in structured evidence: `{paper_structured_evidence_summary[0]['processed_papers_in_structured_evidence']}`",
            f"- Disease/stress paper sources registered: `{paper_structured_evidence_summary[0]['disease_or_stress_paper_sources']}`",
            f"- Figure-derived paper assignments: `{paper_structured_evidence_summary[0]['figure_derived_assignments']}`",
            f"- Text-derived paper assignments: `{paper_structured_evidence_summary[0]['text_derived_assignments']}`",
            "",
            "Motifs are now aligned to the ontology at subfamily resolution, but disease-condition scaling still needs more explicit SI integration, more paper-derived figure evidence, and motif rebuilding that uses ontology directly during cluster formation rather than only at the pattern stage.",
            "",
        ]
    )
    (ONTOLOGY_MOTIF_REPORT_ROOT / "current_state_summary.md").write_text(current_state_summary, encoding="utf-8")

    return {
        "output_root": str(ONTOLOGY_MOTIF_OUTPUT_ROOT),
        "tables_root": str(ONTOLOGY_MOTIF_TABLES_ROOT),
        "report_root": str(ONTOLOGY_MOTIF_REPORT_ROOT),
        "build_counts": build_counts,
        "motif_count_before_after": motif_count_before_after[0],
        "reference_grounding_summary": reference_grounding_summary[0],
        "paper_structured_evidence_summary": paper_structured_evidence_summary[0],
        "confounder_motif_count": len(confounder_motif_summary),
        "high_or_carbonyl_motif_count": sum(
            row["motif_count"] for row in spectral_region_motif_summary if row["spectral_region"] in {"carbonyl_1700_1900", "high_wavenumber_2800_3200"}
        ),
    }
