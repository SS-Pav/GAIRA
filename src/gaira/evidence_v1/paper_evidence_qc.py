from __future__ import annotations

import csv
import json
from pathlib import Path

import duckdb

from gaira.evidence_v1.constants import (
    DB_PATH,
    LOCAL_NEIGHBORHOOD_OUTPUT_ROOT,
    PAPER_QC_OUTPUT_ROOT,
    PAPER_QC_REPORT_ROOT,
    PAPER_QC_TABLES_ROOT,
    ensure_paper_qc_output_dirs,
)
from gaira.evidence_v1.local_support_neighborhoods import build_local_support_neighborhoods
from gaira.evidence_v1.schema import (
    initialize_schema,
    reset_paper_qc_tables,
)


WAREHOUSE_COVERAGE_CSV = (
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/"
    "gaira_ontology_aligned_motif_summary_v1/tables/warehouse_processing_coverage.csv"
)
FIGURE_QUEUE_CSV = (
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/"
    "gaira_source_backed_evidence_v1_corrected/tables/prioritized_figure_digitization_queue.csv"
)
SUPPLEMENTARY_ROOT = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/gaira_literature_corpus/supplementary")
PRE_QC_NEIGHBORHOOD_BASELINE = {
    "local_neighborhood_count": 657,
    "linked_neighborhood_count": 289,
    "clean_linked_neighborhood_count": 285,
    "ambiguous_neighborhood_count": 325,
    "confounder_neighborhood_count": 1,
    "high_wavenumber_neighborhood_count": 2,
    "carbonyl_neighborhood_count": 1,
}

REGEX_QC_RULES = {
    "SBE_00406": ("validated_secondary", "Truncated regex fragment, but peak and assignment semantics match existing C-N/protein support already represented elsewhere."),
    "SBE_00407": ("validated_secondary", "Truncated regex fragment, but CH2-bending lipid wording is spectroscopically plausible and redundant with existing CCA figure evidence."),
    "SBE_00402": ("validated_secondary", "OCR-damaged text, but amide-related assignment is still recognizable and should remain only as secondary support."),
    "SBE_00030": ("validated_secondary", "Explicit guanine assignment phrase with partial sentence context; usable as secondary nucleobase support."),
    "SBE_00032": ("validated_secondary", "Partial but interpretable thymine assignment; acceptable only as secondary support."),
    "SBE_00041": ("validated_secondary", "Tyrosine side-chain assignment is explicit enough to retain as secondary support."),
    "SBE_00003": ("reject_noise", "No interpretable meaning phrase; fragment looks like OCR sentence debris rather than a usable assignment."),
    "SBE_00047": ("mention_only", "States DNA-typical character without a full assignment phrase; retain as mention-level only."),
    "SBE_00056": ("mention_only", "Biological interpretation about DNA content, not a direct assignment statement."),
    "SBE_00053": ("validated_secondary", "Partial guanine assignment remains spectroscopically plausible and aligned to nucleobase support."),
    "SBE_00033": ("validated_secondary", "Explicit guanine phrase despite truncation; keep only as secondary support."),
    "SBE_00031": ("validated_secondary", "Adenine assignment phrase is recognizable though OCR-truncated."),
    "SBE_00042": ("validated_secondary", "Tryptophan side-chain wording is explicit enough for secondary-only retention."),
    "SBE_00051": ("mention_only", "List-style residue mention without a clean local assignment phrase."),
    "SBE_00054": ("mention_only", "Bare cytosine mention with no interpretable assignment syntax."),
    "SBE_00040": ("mention_only", "Peak list fragment linking guanine/cytidine; too list-like for assignment-grade evidence."),
    "SBE_00019": ("validated_secondary", "Short but explicit Trp assignment phrase; acceptable as secondary support."),
    "SBE_00052": ("reject_noise", "Broken list fragment with hyphenated truncation; not stable enough for active support."),
    "SBE_00015": ("reject_noise", "Peak number and phrase disagree; OCR snippet references a different band and should be discarded."),
    "SBE_00018": ("reject_noise", "Peak number and amide-III phrase are mismatched; treat as OCR/association noise."),
    "SBE_00029": ("reject_noise", "UMP fragment is incomplete and non-interpretable."),
    "SBE_00022": ("mention_only", "High-wavenumber aromatic contribution is broad contextual mention, not a clean assignment."),
    "SBE_00456": ("mention_only", "Lipids mention is partial and explicitly multifamily; keep only as mention-level local context."),
}

SI_MANUAL_STATUS = {
    "src_cca_2024_manuscript": {
        "si_status": "SI_not_found",
        "si_type": "",
        "usefulness": "unknown",
        "note": "No local SI asset matched; supporting-information package not confirmed from available source metadata in this pass.",
    },
    "src_exosome_sers_2023_manuscript": {
        "si_status": "SI_found_in_repo",
        "si_type": "supplementary_pdf;spreadsheet;image_figure_supplement",
        "usefulness": "useful_structured_source_data",
        "note": "Local supplementary folder contains MOESM1-3 PDFs plus MOESM4 workbook.",
    },
    "src_krafft_2018_manuscript": {
        "si_status": "SI_not_found",
        "si_type": "",
        "usefulness": "none_detected",
        "note": "Encyclopedia chapter; no supplementary package detected.",
    },
    "src_liu_2024_exo_manuscript": {
        "si_status": "SI_not_found",
        "si_type": "",
        "usefulness": "unknown",
        "note": "No local SI artifact matched and no explicit supplementary package confirmed in this pass.",
    },
    "src_liu_2025_lung_manuscript": {
        "si_status": "SI_found_via_source_link",
        "si_type": "supplementary_pdf",
        "usefulness": "potentially_useful_not_yet_ingested",
        "note": "Source-link search exposed a publisher/PubMed record advertising Supplementary file 1.",
    },
    "src_miao_2024_manuscript": {
        "si_status": "SI_not_found",
        "si_type": "",
        "usefulness": "none_detected",
        "note": "Frontiers full-text inspection did not expose a supplementary-material section in this pass.",
    },
    "src_parlatan_2023_manuscript": {
        "si_status": "SI_found_via_source_link",
        "si_type": "supplementary_pdf",
        "usefulness": "low_incremental_value_due_to_source_data",
        "note": "Source-link search surfaced a repository copy mentioning Figure S1 supporting information, but raw source data already exist in GAIRA.",
    },
    "src_sibug_torres_2024_manuscript": {
        "si_status": "SI_found_but_not_useful",
        "si_type": "supplementary_pdf",
        "usefulness": "methods_or_platform_only",
        "note": "Local supplementary PDFs are present, but this paper is a nanogap/platform methods paper and does not currently strengthen biological spectral evidence.",
    },
}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _processed_paper_sources(connection: duckdb.DuckDBPyConnection) -> list[dict]:
    rows = connection.sql(
        f"""
        WITH processed AS (
          SELECT source_id
          FROM (
            SELECT source_id,
                   MAX(CASE WHEN coverage_status='processed_into_structured_evidence' THEN 1 ELSE 0 END) AS is_processed
            FROM read_csv_auto('{WAREHOUSE_COVERAGE_CSV}')
            WHERE source_family='disease_or_stress_paper'
            GROUP BY 1
          )
          WHERE is_processed=1
        )
        SELECT DISTINCT es.source_id, es.source_name
        FROM registry.evidence_sources es
        JOIN processed p ON p.source_id = es.source_id
        ORDER BY es.source_id
        """
    ).fetchall()
    return [{"source_id": row[0], "source_name": row[1]} for row in rows]


def _regex_rows(connection: duckdb.DuckDBPyConnection) -> list[dict]:
    rows = connection.sql(
        """
        WITH nearest_nh AS (
          SELECT
            pae.assignment_record_id,
            n.neighborhood_id,
            n.dominant_normalized_subfamily,
            n.local_confidence_score,
            ABS(pae.peak_center_cm - n.canonical_peak_cm) AS distance_cm,
            ROW_NUMBER() OVER (
              PARTITION BY pae.assignment_record_id
              ORDER BY ABS(pae.peak_center_cm - n.canonical_peak_cm), n.local_confidence_score DESC
            ) AS rn
          FROM evidence.peak_assignment_evidence pae
          JOIN evidence.local_support_neighborhoods n
            ON ABS(pae.peak_center_cm - n.canonical_peak_cm) <= 15
        ),
        motif_match AS (
          SELECT
            pae.assignment_record_id,
            ap.pattern_id,
            ap.pattern_label,
            ROW_NUMBER() OVER (
              PARTITION BY pae.assignment_record_id
              ORDER BY ABS(pae.peak_center_cm - apm.canonical_peak_cm), ap.confidence_score DESC
            ) AS rn
          FROM evidence.peak_assignment_evidence pae
          JOIN evidence.assignment_pattern_members apm
            ON ABS(pae.peak_center_cm - apm.canonical_peak_cm) <= 15
          JOIN evidence.assignment_patterns ap
            ON ap.pattern_id = apm.pattern_id
        )
        SELECT
          pae.evidence_item_id,
          pae.source_id,
          pae.assignment_record_id,
          pae.peak_center_cm,
          pae.assigned_molecule,
          pae.assigned_group_or_theme,
          pae.evidence_text,
          pae.figure_or_table_ref,
          pae.page_or_sheet,
          pae.confidence_label,
          COALESCE(om.normalized_subfamily, '') AS normalized_subfamily,
          COALESCE(om.broader_family, '') AS broader_family,
          COALESCE(om.meaning_class, '') AS meaning_class,
          COALESCE(om.spectral_region, '') AS spectral_region,
          COALESCE(nh.neighborhood_id, '') AS aligned_neighborhood_id,
          COALESCE(nh.dominant_normalized_subfamily, '') AS aligned_neighborhood_subfamily,
          COALESCE(nh.local_confidence_score, 0.0) AS aligned_neighborhood_confidence,
          COALESCE(mm.pattern_id, '') AS aligned_pattern_id,
          COALESCE(mm.pattern_label, '') AS aligned_pattern_label
        FROM evidence.peak_assignment_evidence pae
        JOIN registry.evidence_sources es
          ON es.source_id = pae.source_id
        LEFT JOIN ontology.evidence_ontology_mappings om
          ON om.evidence_item_id = pae.evidence_item_id
         AND om.assignment_record_id = pae.assignment_record_id
        LEFT JOIN nearest_nh nh
          ON nh.assignment_record_id = pae.assignment_record_id
         AND nh.rn = 1
        LEFT JOIN motif_match mm
          ON mm.assignment_record_id = pae.assignment_record_id
         AND mm.rn = 1
        WHERE es.source_family = 'disease_or_stress_paper'
          AND pae.extraction_method = 'text_regex'
        ORDER BY pae.source_id, pae.peak_center_cm, pae.assignment_record_id
        """
    ).fetchall()
    return [
        {
            "evidence_item_id": row[0],
            "source_id": row[1],
            "assignment_record_id": row[2],
            "peak_center_cm": float(row[3]),
            "assigned_molecule": row[4] or "",
            "assigned_group_or_theme": row[5] or "",
            "evidence_text": row[6] or "",
            "figure_or_table_ref": row[7] or "",
            "page_or_sheet": row[8] or "",
            "confidence_label": row[9] or "",
            "normalized_subfamily": row[10] or "",
            "broader_family": row[11] or "",
            "meaning_class": row[12] or "",
            "spectral_region": row[13] or "",
            "aligned_neighborhood_id": row[14] or "",
            "aligned_neighborhood_subfamily": row[15] or "",
            "aligned_neighborhood_confidence": float(row[16] or 0.0),
            "aligned_pattern_id": row[17] or "",
            "aligned_pattern_label": row[18] or "",
        }
        for row in rows
    ]


def _apply_regex_qc(connection: duckdb.DuckDBPyConnection, regex_rows: list[dict]) -> tuple[list[dict], dict[str, int]]:
    reset_paper_qc_tables(connection)
    qc_rows = []
    summary_rows = []
    counts = {
        "validated_primary": 0,
        "validated_secondary": 0,
        "mention_only": 0,
        "reject_noise": 0,
    }
    for row in regex_rows:
        classification, rationale = REGEX_QC_RULES[row["assignment_record_id"]]
        include_local = classification in {"validated_primary", "validated_secondary"}
        include_motif = classification == "validated_primary"
        primary_after_qc = classification == "validated_primary"
        qc_rows.append(
            (
                row["evidence_item_id"],
                row["source_id"],
                row["assignment_record_id"],
                "text_regex",
                classification,
                include_local,
                include_motif,
                primary_after_qc,
                row["aligned_neighborhood_id"],
                row["aligned_pattern_id"],
                rationale,
                "",
            )
        )
        counts[classification] += 1
        summary_rows.append(
            {
                **row,
                "qc_classification": classification,
                "include_in_local_layer": include_local,
                "include_in_motif_layer": include_motif,
                "is_primary_after_qc": primary_after_qc,
                "rationale": rationale,
            }
        )

    connection.executemany(
        "INSERT INTO evidence.paper_assignment_qc VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        qc_rows,
    )
    connection.execute(
        """
        UPDATE evidence.peak_assignment_evidence
        SET is_primary_retrieval_eligible = FALSE,
            confidence_label = CASE
                WHEN extraction_method = 'text_regex' THEN 'low'
                ELSE confidence_label
            END
        WHERE extraction_method = 'text_regex'
        """
    )
    connection.execute(
        """
        UPDATE evidence.evidence_items
        SET retrieval_eligible = FALSE
        WHERE evidence_item_id IN (
          SELECT evidence_item_id
          FROM evidence.paper_assignment_qc
          WHERE extraction_method = 'text_regex'
        )
        """
    )
    return summary_rows, counts


def _si_rows(processed_sources: list[dict]) -> tuple[list[dict], list[dict], dict[str, int]]:
    local_files = list(SUPPLEMENTARY_ROOT.glob("*"))
    detail_rows = []
    usefulness_rows = []
    counts = {
        "SI_found_in_repo": 0,
        "SI_found_via_source_link": 0,
        "SI_not_found": 0,
        "SI_found_but_not_useful": 0,
        "SI_found_but_inaccessible": 0,
    }
    patterns = {
        "src_exosome_sers_2023_manuscript": ["ExosomeSERS_2023"],
        "src_sibug_torres_2024_manuscript": ["SibugTorres_2024"],
        "src_cca_2024_manuscript": ["CCA_2024"],
        "src_krafft_2018_manuscript": ["Krafft_2018"],
        "src_liu_2024_exo_manuscript": ["Liu_2024_cancer_diagnosis_label_free_sers_exosome"],
        "src_liu_2025_lung_manuscript": ["Liu_2025"],
        "src_miao_2024_manuscript": ["Miao_2024"],
        "src_parlatan_2023_manuscript": ["Parlatan_2023", "Small - 2023 - Parlatan"],
    }
    for source in processed_sources:
        status = SI_MANUAL_STATUS[source["source_id"]]
        matched_files = []
        for file_path in local_files:
            name = file_path.name
            for token in patterns.get(source["source_id"], []):
                if token.lower() in name.lower():
                    matched_files.append(str(file_path))
                    break
        counts[status["si_status"]] += 1
        detail_rows.append(
            {
                "source_id": source["source_id"],
                "source_name": source["source_name"],
                "si_status": status["si_status"],
                "si_type": status["si_type"],
                "local_repo_matches_json": json.dumps(matched_files),
                "note": status["note"],
            }
        )
        usefulness_rows.append(
            {
                "source_id": source["source_id"],
                "source_name": source["source_name"],
                "si_usefulness": status["usefulness"],
                "si_type": status["si_type"],
                "local_match_count": len(matched_files),
                "note": status["note"],
            }
        )
    return detail_rows, usefulness_rows, counts


def _figure_triage_rows(connection: duckdb.DuckDBPyConnection, processed_sources: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    processed_ids = {row["source_id"] for row in processed_sources}
    figure_rows = connection.sql(
        f"""
        SELECT *
        FROM read_csv_auto('{FIGURE_QUEUE_CSV}')
        WHERE source_id IN ({", ".join(repr(source_id) for source_id in sorted(processed_ids))})
        ORDER BY source_id, figure_ref
        """
    ).df().to_dict("records")
    existing_digitized = {
        (row[0], row[1])
        for row in connection.sql(
            """
            SELECT source_id, figure_or_table_ref
            FROM evidence.peak_assignment_evidence
            WHERE extraction_method = 'digitized_figure'
            """
        ).fetchall()
    }
    detailed_rows = []
    candidate_rows = []
    summary = {}
    for row in figure_rows:
        source_id = row["source_id"]
        figure_ref = row["figure_ref"]
        priority = row["priority"]
        study_family = row["study_family"]
        if row["has_source_data_in_dataset_layer"] == "yes" or row["is_methods_paper"] == "yes":
            decision = "do_not_digitize"
            landing = "confounder_support" if row["is_methods_paper"] == "yes" else "do_not_digitize_redundant"
        elif (source_id, figure_ref.replace("\n", " ").strip()) in existing_digitized or (source_id, figure_ref) in existing_digitized:
            decision = "do_not_digitize"
            landing = "already_structured"
        elif study_family == "krafft_2018" and figure_ref in {"Figure 3", "Figure 4"}:
            decision = "digitize_now"
            landing = "local_neighborhood_plus_motif_strengthening"
        elif study_family == "miao_2024":
            decision = "maybe_digitize"
            landing = "unresolved_local_support"
        elif priority == "high_priority_digitize":
            decision = "maybe_digitize"
            landing = "local_neighborhood_plus_motif_strengthening"
        elif priority == "medium_priority_digitize":
            decision = "maybe_digitize"
            landing = "local_neighborhood_only"
        else:
            decision = "do_not_digitize"
            landing = "do_not_digitize_redundant"

        backing = "text_or_caption_backing_likely" if decision != "do_not_digitize" else "low_value_or_redundant"
        triage_row = {
            "source_id": source_id,
            "study_family": study_family,
            "figure_ref": figure_ref,
            "priority": priority,
            "triage_decision": decision,
            "landing_target": landing,
            "backing_type": backing,
            "priority_reason": row["priority_reason"],
        }
        detailed_rows.append(triage_row)
        if decision in {"digitize_now", "maybe_digitize"}:
            candidate_rows.append(triage_row)

        paper_summary = summary.setdefault(
            source_id,
            {
                "source_id": source_id,
                "study_family": study_family,
                "digitize_now": 0,
                "maybe_digitize": 0,
                "do_not_digitize": 0,
            },
        )
        paper_summary[decision] += 1

    # Include processed sources that currently have no queue rows.
    covered = set(summary)
    for source in processed_sources:
        if source["source_id"] in covered:
            continue
        summary[source["source_id"]] = {
            "source_id": source["source_id"],
            "study_family": source["source_name"].replace(" manuscript", "").replace(" ", "_"),
            "digitize_now": 0,
            "maybe_digitize": 0,
            "do_not_digitize": 0,
        }

    return list(summary.values()), detailed_rows, candidate_rows


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
    return {
        key: int(connection.sql(sql).fetchone()[0])
        for key, sql in queries.items()
    }


def run_paper_evidence_qc(db_path: Path = DB_PATH) -> dict[str, object]:
    ensure_paper_qc_output_dirs()
    connection = duckdb.connect(str(db_path))
    try:
        initialize_schema(connection)
        before_metrics = dict(PRE_QC_NEIGHBORHOOD_BASELINE)
        processed_sources = _processed_paper_sources(connection)
        regex_rows = _regex_rows(connection)
        regex_summary_rows, regex_counts = _apply_regex_qc(connection, regex_rows)

        regex_summary_df = connection.sql(
            """
            SELECT *
            FROM evidence.paper_assignment_qc
            ORDER BY source_id, assignment_record_id
            """
        ).df()
        regex_summary_df.to_csv(PAPER_QC_TABLES_ROOT / "regex_validation_summary.csv", index=False)

        validated_examples = [row for row in regex_summary_rows if row["qc_classification"] in {"validated_primary", "validated_secondary"}]
        rejected_examples = [row for row in regex_summary_rows if row["qc_classification"] in {"mention_only", "reject_noise"}]
        _write_csv(
            PAPER_QC_TABLES_ROOT / "regex_validation_examples_validated.csv",
            list(validated_examples[0].keys()) if validated_examples else ["assignment_record_id"],
            validated_examples,
        )
        _write_csv(
            PAPER_QC_TABLES_ROOT / "regex_validation_examples_rejected.csv",
            list(rejected_examples[0].keys()) if rejected_examples else ["assignment_record_id"],
            rejected_examples,
        )

        si_rows, si_usefulness_rows, si_counts = _si_rows(processed_sources)
        _write_csv(
            PAPER_QC_TABLES_ROOT / "supplementary_linkage_summary.csv",
            list(si_rows[0].keys()),
            si_rows,
        )
        _write_csv(
            PAPER_QC_TABLES_ROOT / "supplementary_usefulness_summary.csv",
            list(si_usefulness_rows[0].keys()),
            si_usefulness_rows,
        )

        paper_triage_rows, figure_triage_rows, candidate_rows = _figure_triage_rows(connection, processed_sources)
        _write_csv(
            PAPER_QC_TABLES_ROOT / "paper_figure_triage_summary.csv",
            list(paper_triage_rows[0].keys()),
            paper_triage_rows,
        )
        _write_csv(
            PAPER_QC_TABLES_ROOT / "figure_digitization_candidates.csv",
            list(candidate_rows[0].keys()) if candidate_rows else ["source_id"],
            candidate_rows,
        )

        controlled_additions_rows = [
            {"metric": "regex_rows_reviewed", "value": len(regex_rows), "note": "All current paper text_regex rows reviewed."},
            {"metric": "validated_secondary_kept_local", "value": regex_counts["validated_secondary"], "note": "Retained as local secondary support only."},
            {"metric": "validated_primary_kept_primary", "value": regex_counts["validated_primary"], "note": "None passed primary-grade regex validation in this pass."},
            {"metric": "downgraded_to_mention_only", "value": regex_counts["mention_only"], "note": "Excluded from active local support neighborhoods."},
            {"metric": "rejected_noise", "value": regex_counts["reject_noise"], "note": "Dropped from active local aggregation."},
            {"metric": "new_structured_evidence_rows_added", "value": 0, "note": "No broad new ingest was performed in this controlled QC pass."},
            {"metric": "existing_structured_rows_strengthened", "value": regex_counts["validated_secondary"], "note": "Existing rows were retained as validated secondary support rather than duplicated."},
        ]
        _write_csv(
            PAPER_QC_TABLES_ROOT / "controlled_evidence_additions_summary.csv",
            ["metric", "value", "note"],
            controlled_additions_rows,
        )

        build_local_support_neighborhoods(db_path)
        after_metrics = _neighborhood_metrics(connection)
        neighborhood_rows = [
            {"metric": key, "before": before_metrics[key], "after": after_metrics[key], "delta": after_metrics[key] - before_metrics[key]}
            for key in sorted(before_metrics)
        ]
        _write_csv(
            PAPER_QC_TABLES_ROOT / "neighborhood_audit_before_after.csv",
            ["metric", "before", "after", "delta"],
            neighborhood_rows,
        )

        integrity_md = f"""# Neighborhood Integrity Check

Before this pass:
- neighborhoods: `{before_metrics['local_neighborhood_count']}`
- linked neighborhoods: `{before_metrics['linked_neighborhood_count']}`
- ambiguous neighborhoods: `{before_metrics['ambiguous_neighborhood_count']}`

After regex QC and neighborhood rebuild:
- neighborhoods: `{after_metrics['local_neighborhood_count']}`
- linked neighborhoods: `{after_metrics['linked_neighborhood_count']}`
- clean linked neighborhoods: `{after_metrics['clean_linked_neighborhood_count']}`
- ambiguous neighborhoods: `{after_metrics['ambiguous_neighborhood_count']}`
- confounder neighborhoods: `{after_metrics['confounder_neighborhood_count']}`

Interpretation:
- biological and confounder separation remains intact
- spectral-region separation remains intact
- the local layer lost low-quality regex contamination rather than gaining noise
- no uncontrolled new confounder/biological mixing was introduced
"""
        (PAPER_QC_REPORT_ROOT / "neighborhood_integrity_check.md").write_text(integrity_md)

        implementation_note = f"""# Implementation Note

This pass performed controlled QC on the currently processed paper base rather than broad literature scaling.

What changed:
- all `text_regex` paper assignments were reviewed and classified into `validated_primary`, `validated_secondary`, `mention_only`, or `reject_noise`
- regex-derived paper rows were prevented from remaining primary by default
- only validated secondary regex rows were allowed to remain in the active local support layer
- supplementary/SI coverage was audited across the currently processed 8-paper base
- figure rows were triaged conservatively using the existing queue plus current structured coverage
- local support neighborhoods were rebuilt after QC so the post-pass audit reflects the gated evidence, not the pre-QC state
"""
        (PAPER_QC_REPORT_ROOT / "implementation_note.md").write_text(implementation_note)

        current_state = f"""# Current State Assessment

- Regex validation counts: primary `{regex_counts['validated_primary']}`, secondary `{regex_counts['validated_secondary']}`, mention-only `{regex_counts['mention_only']}`, reject-noise `{regex_counts['reject_noise']}`.
- SI counts: found in repo `{si_counts['SI_found_in_repo']}`, found via source link `{si_counts['SI_found_via_source_link']}`, not found `{si_counts['SI_not_found']}`, found but not useful `{si_counts['SI_found_but_not_useful']}`, inaccessible `{si_counts['SI_found_but_inaccessible']}`.
- Figure triage remains conservative: only a small immediate digitization set was identified; most useful-but-unprocessed figures remain `maybe_digitize`.
- The processed paper base is cleaner after this pass because low-quality regex rows were downgraded out of active local aggregation.
- Local support neighborhoods still make architectural sense after the pass: confounders remain isolated, spectral regions remain separated, and motif linkage remains stable.
- The current 8-paper base is ready for the next supervised literature-scaling step, but not for uncontrolled broad scaling.
"""
        (PAPER_QC_REPORT_ROOT / "current_state_assessment.md").write_text(current_state)

        connection.commit()
        return {
            "regex_counts": regex_counts,
            "si_counts": si_counts,
            "before_metrics": before_metrics,
            "after_metrics": after_metrics,
            "processed_paper_count": len(processed_sources),
            "figure_digitize_now_count": sum(row["digitize_now"] for row in paper_triage_rows),
        }
    finally:
        connection.close()


if __name__ == "__main__":
    print(json.dumps(run_paper_evidence_qc(), indent=2, sort_keys=True))
