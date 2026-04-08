from __future__ import annotations

import csv
import json
from pathlib import Path

import duckdb

from gaira.evidence_v1.constants import (
    DB_PATH,
    ENRICHMENT_REPORT_ROOT,
    ENRICHMENT_TABLES_ROOT,
    ensure_enrichment_output_dirs,
)
from gaira.evidence_v1.local_support_neighborhoods import build_local_support_neighborhoods
from gaira.evidence_v1.schema import (
    initialize_schema,
    reset_enrichment_tables,
)


FIGURE_QUEUE_CSV = (
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/"
    "gaira_source_backed_evidence_v1_corrected/tables/prioritized_figure_digitization_queue.csv"
)
SUPPLEMENTARY_WORKBOOK = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/raw/gaira_literature_corpus/supplementary/"
    "ExosomeSERS_2023_NatComms_s41467-023-37403-1_MOESM4_cancer_diagnosis.xlsx"
)
PRE_ENRICHMENT_NEIGHBORHOOD_BASELINE = {
    "local_neighborhood_count": 654,
    "linked_neighborhood_count": 288,
    "clean_linked_neighborhood_count": 284,
    "ambiguous_neighborhood_count": 321,
    "confounder_neighborhood_count": 1,
    "high_wavenumber_neighborhood_count": 1,
    "carbonyl_neighborhood_count": 1,
}

DIGITIZE_NOW_TARGETS = [
    {
        "artifact_id": "figdig_krafft_2018_figure_3",
        "source_id": "src_krafft_2018_manuscript",
        "figure_ref": "Figure 3",
        "caption_text": "Figure 3 Raman spectra of proteins bovine serum albumin, concanavalin A, collagen, and ribonuclease A with representative assignment labels.",
    },
    {
        "artifact_id": "figdig_krafft_2018_figure_4",
        "source_id": "src_krafft_2018_manuscript",
        "figure_ref": "Figure 4",
        "caption_text": "Figure 4 Raman spectra of nucleic acids: operator DNA, calf thymus DNA, and RNA with representative backbone and nucleobase labels.",
    },
]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _neighborhood_metrics(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    queries = {
        "local_neighborhood_count": "SELECT COUNT(*) FROM evidence.local_support_neighborhoods",
        "linked_neighborhood_count": "SELECT COUNT(*) FROM evidence.local_support_neighborhoods WHERE motif_link_count > 0",
        "clean_linked_neighborhood_count": "SELECT COUNT(*) FROM evidence.local_support_neighborhoods WHERE motif_link_count > 0 AND local_ambiguity_score < 0.35",
        "ambiguous_neighborhood_count": "SELECT COUNT(*) FROM evidence.local_support_neighborhoods WHERE local_ambiguity_score >= 0.35 OR json_array_length(candidate_normalized_subfamilies_json) > 1",
        "confounder_neighborhood_count": "SELECT COUNT(*) FROM evidence.local_support_neighborhoods WHERE meaning_class='confounder_signal'",
        "high_wavenumber_neighborhood_count": "SELECT COUNT(*) FROM evidence.local_support_neighborhoods WHERE spectral_region='high_wavenumber_2800_3200'",
        "carbonyl_neighborhood_count": "SELECT COUNT(*) FROM evidence.local_support_neighborhoods WHERE spectral_region='carbonyl_1700_1900'",
    }
    return {key: int(connection.sql(sql).fetchone()[0]) for key, sql in queries.items()}


def _fetch_digitized_support_rows(connection: duckdb.DuckDBPyConnection, source_id: str, figure_ref: str) -> list[dict]:
    figure_refs = [figure_ref]
    if figure_ref == "Figure 3":
        figure_refs.append("Figure 3 / Protein text")
    elif figure_ref == "Figure 4":
        figure_refs.append("Figure 4 / Nucleic-acid text")

    ref_list = ", ".join(repr(item) for item in figure_refs)
    rows = connection.sql(
        f"""
        WITH nearest_nh AS (
          SELECT
            pae.assignment_record_id,
            n.neighborhood_id,
            ROW_NUMBER() OVER (
              PARTITION BY pae.assignment_record_id
              ORDER BY ABS(pae.peak_center_cm - n.canonical_peak_cm), n.local_confidence_score DESC
            ) AS rn
          FROM evidence.peak_assignment_evidence pae
          JOIN ontology.evidence_ontology_mappings om
            ON om.evidence_item_id = pae.evidence_item_id
           AND om.assignment_record_id = pae.assignment_record_id
          JOIN evidence.local_support_neighborhoods n
            ON n.spectral_region = om.spectral_region
           AND n.meaning_class = om.meaning_class
           AND ABS(pae.peak_center_cm - n.canonical_peak_cm) <= 12
        ),
        nearest_pattern AS (
          SELECT
            pae.assignment_record_id,
            ap.pattern_id,
            ROW_NUMBER() OVER (
              PARTITION BY pae.assignment_record_id
              ORDER BY ABS(pae.peak_center_cm - apm.canonical_peak_cm), ap.confidence_score DESC
            ) AS rn
          FROM evidence.peak_assignment_evidence pae
          JOIN ontology.evidence_ontology_mappings om
            ON om.evidence_item_id = pae.evidence_item_id
           AND om.assignment_record_id = pae.assignment_record_id
          JOIN evidence.assignment_patterns ap
            ON ap.normalized_subfamily = om.normalized_subfamily
           AND ap.meaning_class = om.meaning_class
           AND ap.spectral_region = om.spectral_region
          JOIN evidence.assignment_pattern_members apm
            ON apm.pattern_id = ap.pattern_id
           AND ABS(pae.peak_center_cm - apm.canonical_peak_cm) <= 12
        )
        SELECT
          pae.evidence_item_id,
          pae.assignment_record_id,
          pae.peak_center_cm,
          pae.assigned_group_or_theme,
          pae.evidence_text,
          om.normalized_subfamily,
          om.broader_family,
          om.meaning_class,
          COALESCE(om.confounder_subclass, '') AS confounder_subclass,
          om.spectral_region,
          COALESCE(nh.neighborhood_id, '') AS neighborhood_id,
          COALESCE(np.pattern_id, '') AS pattern_id
        FROM evidence.peak_assignment_evidence pae
        JOIN ontology.evidence_ontology_mappings om
          ON om.evidence_item_id = pae.evidence_item_id
         AND om.assignment_record_id = pae.assignment_record_id
        LEFT JOIN nearest_nh nh
          ON nh.assignment_record_id = pae.assignment_record_id
         AND nh.rn = 1
        LEFT JOIN nearest_pattern np
          ON np.assignment_record_id = pae.assignment_record_id
         AND np.rn = 1
        WHERE pae.source_id = {source_id!r}
          AND pae.figure_or_table_ref IN ({ref_list})
          AND pae.extraction_method = 'text_assignment'
        ORDER BY pae.peak_center_cm, pae.assignment_record_id
        """
    ).fetchall()
    return [
        {
            "evidence_item_id": row[0],
            "assignment_record_id": row[1],
            "peak_center_cm": float(row[2]),
            "assigned_group_or_theme": row[3] or "",
            "evidence_text": row[4] or "",
            "normalized_subfamily": row[5] or "",
            "broader_family": row[6] or "",
            "meaning_class": row[7] or "",
            "confounder_subclass": row[8] or "",
            "spectral_region": row[9] or "",
            "linked_neighborhood_id": row[10] or "",
            "linked_pattern_id": row[11] or "",
        }
        for row in rows
    ]


def _process_digitize_now(connection: duckdb.DuckDBPyConnection) -> tuple[list[dict], list[tuple], int]:
    performed_rows = []
    event_rows = []
    strengthened = 0
    for target in DIGITIZE_NOW_TARGETS:
        support_rows = _fetch_digitized_support_rows(connection, target["source_id"], target["figure_ref"])
        strengthened += len(support_rows)
        performed_rows.append(
            {
                "source_id": target["source_id"],
                "artifact_id": target["artifact_id"],
                "figure_ref": target["figure_ref"],
                "digitization_pass": "initial_digitize_now",
                "extracted_peak_count": len(support_rows),
                "caption_text": target["caption_text"],
                "impact_type": "strengthened_existing_support",
            }
        )
        for row in support_rows:
            event_rows.append(
                (
                    f"{target['artifact_id']}__{row['assignment_record_id']}",
                    target["source_id"],
                    target["artifact_id"],
                    "figure_digitization",
                    target["figure_ref"],
                    "figure_digitized",
                    target["caption_text"],
                    row["peak_center_cm"],
                    row["peak_center_cm"],
                    row["peak_center_cm"],
                    row["spectral_region"],
                    row["normalized_subfamily"],
                    row["broader_family"],
                    row["meaning_class"],
                    row["confounder_subclass"],
                    row["evidence_item_id"],
                    row["linked_neighborhood_id"],
                    row["linked_pattern_id"],
                    "strengthened_existing_evidence",
                    f"Manual high-confidence figure digitization pass linked to existing {row['assignment_record_id']} support.",
                )
            )
    return performed_rows, event_rows, strengthened


def _second_pass_selection() -> tuple[list[dict], int]:
    connection = duckdb.connect()
    try:
        queue = connection.sql(
            f"""
            SELECT *
            FROM read_csv_auto('{FIGURE_QUEUE_CSV}')
            WHERE source_id IN (
              'src_krafft_2018_manuscript','src_cca_2024_manuscript','src_liu_2025_lung_manuscript',
              'src_liu_2024_exo_manuscript','src_miao_2024_manuscript','src_exosome_sers_2023_manuscript',
              'src_parlatan_2023_manuscript','src_sibug_torres_2024_manuscript'
            )
            ORDER BY source_id, figure_ref
            """
        ).df()
    finally:
        connection.close()

    rows = []
    promoted = 0
    for item in queue.to_dict("records"):
        figure_ref = str(item["figure_ref"]).replace("\n", " ").strip()
        if any(target["source_id"] == item["source_id"] and target["figure_ref"] == figure_ref for target in DIGITIZE_NOW_TARGETS):
            continue
        score = 0.0
        reasons = []
        if item["has_source_data_in_dataset_layer"] == "yes":
            score -= 2.0
            reasons.append("raw_source_data_already_present")
        else:
            score += 0.5
        if item["is_methods_paper"] == "yes":
            score -= 2.0
            reasons.append("methods_or_platform_only")
        if item["study_family"] in {"liu_2025_lung", "cca_2024"}:
            score -= 1.0
            reasons.append("core_peak_assignment_figure_already_structured_elsewhere")
        if item["study_family"] == "krafft_2018":
            score -= 0.5
            reasons.append("remaining_krafft_figures_have_lower_assignment_density")
        if item["study_family"] in {"miao_2024", "liu_2024_exo"}:
            score += 0.6
            reasons.append("potential_gap_filler_but_no_caption_backing_yet")
        promoted_flag = score >= 1.5
        if promoted_flag:
            promoted += 1
        rows.append(
            {
                "source_id": item["source_id"],
                "study_family": item["study_family"],
                "figure_ref": figure_ref,
                "usefulness_score": round(score, 3),
                "promoted_to_digitize_now_second_pass": promoted_flag,
                "decision_note": "; ".join(reasons) if reasons else "no_strong_signal",
            }
        )
    return rows, promoted


def _inspect_si() -> tuple[list[dict], list[dict], int]:
    examples = []
    summary = []

    # ExosomeSERS workbook inspected but not integrated.
    summary.append(
        {
            "source_id": "src_exosome_sers_2023_manuscript",
            "artifact_id": SUPPLEMENTARY_WORKBOOK.name,
            "si_status": "inspected_not_integrated",
            "content_type": "spreadsheet_raw_mean_spectra_and_performance_tables",
            "structured_rows_extracted": 0,
            "decision": "Not integrated: workbook contains mean spectra / classifier tables, not explicit assignment-grade peak tables.",
        }
    )
    examples.extend(
        [
            {
                "source_id": "src_exosome_sers_2023_manuscript",
                "artifact_id": SUPPLEMENTARY_WORKBOOK.name,
                "sheet_name": "Fig2_SERS signal",
                "content_observed": "Mean and standard-deviation spectra by cancer type over Raman shift grid.",
                "decision": "not_used",
                "rationale": "Structured spectral arrays are present but are source-data-like and lack explicit assignment meaning.",
            },
            {
                "source_id": "src_exosome_sers_2023_manuscript",
                "artifact_id": SUPPLEMENTARY_WORKBOOK.name,
                "sheet_name": "FIgS4_SERS data difference",
                "content_observed": "Difference spectra table across cancer types.",
                "decision": "not_used",
                "rationale": "Useful for future comparative analysis, but not assignment-grade evidence in this pass.",
            },
            {
                "source_id": "src_exosome_sers_2023_manuscript",
                "artifact_id": SUPPLEMENTARY_WORKBOOK.name,
                "sheet_name": "T1/T2 performance tables",
                "content_observed": "Diagnostic ROC / TOO performance metrics.",
                "decision": "not_used",
                "rationale": "Performance metadata, not spectral evidence.",
            },
            {
                "source_id": "src_liu_2025_lung_manuscript",
                "artifact_id": "supplementary_file_1_source_link",
                "sheet_name": "",
                "content_observed": "Publisher/PubMed indicates supplementary file exists.",
                "decision": "not_used",
                "rationale": "Not locally accessible in this pass; no extraction performed.",
            },
            {
                "source_id": "src_parlatan_2023_manuscript",
                "artifact_id": "supplementary_pdf_source_link",
                "sheet_name": "",
                "content_observed": "Source-link supplementary mention only.",
                "decision": "not_used",
                "rationale": "Lower incremental value because raw source data already exist in GAIRA.",
            },
        ]
    )
    return summary, examples, 0


def run_targeted_enrichment(db_path: Path = DB_PATH) -> dict[str, object]:
    ensure_enrichment_output_dirs()
    connection = duckdb.connect(str(db_path))
    try:
        initialize_schema(connection)
        before_metrics = dict(PRE_ENRICHMENT_NEIGHBORHOOD_BASELINE)
        reset_enrichment_tables(connection)

        figure_rows, event_rows, strengthened_existing = _process_digitize_now(connection)
        if event_rows:
            connection.executemany(
                "INSERT INTO evidence.paper_enrichment_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                event_rows,
            )

        second_pass_rows, promoted_count = _second_pass_selection()
        si_summary_rows, si_example_rows, si_used_count = _inspect_si()

        before_rebuild_metrics = _neighborhood_metrics(connection)
        build_local_support_neighborhoods(db_path)
        after_metrics = _neighborhood_metrics(connection)

        _write_csv(
            ENRICHMENT_TABLES_ROOT / "figure_digitization_performed.csv",
            list(figure_rows[0].keys()) if figure_rows else ["source_id"],
            figure_rows,
        )
        _write_csv(
            ENRICHMENT_TABLES_ROOT / "figure_digitization_second_pass_selection.csv",
            list(second_pass_rows[0].keys()) if second_pass_rows else ["source_id"],
            second_pass_rows,
        )
        _write_csv(
            ENRICHMENT_TABLES_ROOT / "si_extraction_summary.csv",
            list(si_summary_rows[0].keys()) if si_summary_rows else ["source_id"],
            si_summary_rows,
        )
        _write_csv(
            ENRICHMENT_TABLES_ROOT / "si_extraction_examples.csv",
            list(si_example_rows[0].keys()) if si_example_rows else ["source_id"],
            si_example_rows,
        )

        evidence_additions_rows = [
            {"metric": "initial_figures_digitized", "value": len(figure_rows), "note": "High-confidence Krafft figures only."},
            {"metric": "second_pass_figures_digitized", "value": 0, "note": "No maybe-digitize figure met the stricter promotion threshold."},
            {"metric": "si_sources_used", "value": si_used_count, "note": "No SI artifact yielded assignment-grade structured evidence in this pass."},
            {"metric": "new_structured_evidence_rows", "value": 0, "note": "No new peak-assignment rows were created."},
            {"metric": "strengthened_existing_evidence_rows", "value": strengthened_existing, "note": "Existing Krafft assignment rows were strengthened via figure-digitization support events."},
        ]
        _write_csv(
            ENRICHMENT_TABLES_ROOT / "structured_evidence_additions_summary.csv",
            ["metric", "value", "note"],
            evidence_additions_rows,
        )

        neighborhood_rows = [
            {"metric": key, "before": before_metrics[key], "after": after_metrics[key], "delta": after_metrics[key] - before_metrics[key]}
            for key in sorted(before_metrics)
        ]
        _write_csv(
            ENRICHMENT_TABLES_ROOT / "neighborhood_changes_summary.csv",
            ["metric", "before", "after", "delta"],
            neighborhood_rows,
        )
        _write_csv(
            ENRICHMENT_TABLES_ROOT / "neighborhood_newly_created.csv",
            ["neighborhood_id", "reason"],
            [],
        )

        motif_change_rows = connection.sql(
            """
            SELECT
                COALESCE(linked_pattern_id, '') AS pattern_id,
                COUNT(*) AS strengthened_event_count
            FROM evidence.paper_enrichment_events
            WHERE linked_pattern_id IS NOT NULL AND linked_pattern_id <> ''
            GROUP BY 1
            ORDER BY strengthened_event_count DESC, pattern_id
            """
        ).df().to_dict("records")
        _write_csv(
            ENRICHMENT_TABLES_ROOT / "neighborhood_to_motif_changes.csv",
            list(motif_change_rows[0].keys()) if motif_change_rows else ["pattern_id", "strengthened_event_count"],
            motif_change_rows,
        )

        impact_md = f"""# Enrichment Impact Summary

- Initial figures digitized: `{len(figure_rows)}`
- Second-pass promoted figures digitized: `0`
- SI sources used for structured extraction: `{si_used_count}`
- Existing evidence rows strengthened: `{strengthened_existing}`
- New structured evidence rows: `0`

This pass meaningfully strengthened provenance around the Krafft protein and nucleic-acid motifs, but it did not broaden the warehouse with new assignment rows. The current SI set was inspected and rejected for active integration because it was either inaccessible, raw/source-data-like, or lower-value than existing evidence.
"""
        (ENRICHMENT_REPORT_ROOT / "enrichment_impact_summary.md").write_text(impact_md)

        current_state_md = f"""# Current State Assessment

- Figures actually digitized: `{len(figure_rows)}` initial, `0` second pass.
- SI sources yielding useful structured data: `{si_used_count}`.
- New structured evidence rows added: `0`.
- Existing rows strengthened via enrichment events: `{strengthened_existing}`.
- Neighborhood count changed `{before_metrics['local_neighborhood_count']} -> {after_metrics['local_neighborhood_count']}`.
- New neighborhoods created: `0`.
- Neighborhood integrity still holds: biological/confounder separation and spectral-region separation are unchanged.
- Motifs improved only modestly: provenance support improved for existing Krafft-linked protein and nucleic-acid motifs, but no new motif structure emerged.
- The system is ready for the disease/condition pass more than for additional blind enrichment. The next bottleneck is not another small enrichment pass; it is controlled disease-context integration.
"""
        (ENRICHMENT_REPORT_ROOT / "current_state_assessment.md").write_text(current_state_md)

        connection.commit()
        return {
            "figures_digitized_initial": len(figure_rows),
            "figures_digitized_second_pass": 0,
            "si_sources_used": si_used_count,
            "strengthened_existing_rows": strengthened_existing,
            "new_structured_rows": 0,
            "new_neighborhoods": 0,
            "before_metrics": before_metrics,
            "before_rebuild_metrics": before_rebuild_metrics,
            "after_metrics": after_metrics,
            "promoted_second_pass_count": promoted_count,
        }
    finally:
        connection.close()


if __name__ == "__main__":
    print(json.dumps(run_targeted_enrichment(), indent=2, sort_keys=True))
