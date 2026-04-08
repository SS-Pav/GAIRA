from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import duckdb

from gaira.evidence_v1.constants import (
    DIGITIZATION_QUEUE_PATH,
    DIGITIZATION_NOTE_PATH,
    MINIMAL_CONTEXT_DOC_IDS,
    SOURCE_BACKED_MENTIONS_PATH,
    SOURCE_BACKED_NOISE_PATH,
    SOURCE_BACKED_NOTE_PATH,
    SOURCE_BACKED_VALID_PATH,
)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: str | None) -> float | None:
    text = (value or "").strip()
    if not text:
        return None
    return float(text)


def _normalize_text(value: str | None) -> str:
    return (value or "").strip()


def _domain_from_text(*values: str | None) -> str:
    text = " ".join(_normalize_text(value).lower() for value in values)
    if "extracellular vesicle" in text or " exosome" in text or text.startswith("ev"):
        return "ev"
    if "serum" in text:
        return "serum"
    if "plasma" in text:
        return "plasma"
    if "pathogen" in text or "bacteria" in text or "virus" in text:
        return "pathogen"
    if "biofluids" in text or "biofluid" in text:
        return "biofluids"
    if "reference" in text or "pure_biomolecule" in text or "pure biomolecule" in text:
        return "reference"
    return "generic"


def _modality_from_text(*values: str | None) -> str:
    text = " ".join(_normalize_text(value).lower() for value in values)
    if "sers" in text:
        return "sers"
    if "raman" in text:
        return "raman"
    return "unknown"


def _bsv_from_text(value: str | None) -> tuple[str | None, float, str]:
    text = _normalize_text(value).lower()
    if not text or text == "unknown":
        return None, 0.0, "no_resolved_bsv_mapping"
    if "protein" in text or "amide" in text:
        return "bsv_protein", 1.0, f"mapped from {value}"
    if "nucleic" in text or "adenine" in text or "guanine" in text or "cytosine" in text or "thymine" in text:
        return "bsv_nucleic_acid", 1.0, f"mapped from {value}"
    if "amino" in text or "tryptophan" in text or "tyrosine" in text:
        return "bsv_amino_acid", 0.9, f"mapped from {value}"
    if "lipid" in text or "fatty" in text or "phospholipid" in text:
        return "bsv_lipid", 1.0, f"mapped from {value}"
    if "carbohydrate" in text or "saccharide" in text or "glycan" in text:
        return "bsv_carbohydrate", 1.0, f"mapped from {value}"
    if "hormone" in text:
        return "bsv_hormone", 1.0, f"mapped from {value}"
    return None, 0.0, f"unmapped:{value}"


def _insert_rows(connection: duckdb.DuckDBPyConnection, table_name: str, rows: list[tuple], column_count: int) -> None:
    if not rows:
        return
    placeholders = ", ".join(["?"] * column_count)
    connection.executemany(f"INSERT INTO {table_name} VALUES ({placeholders})", rows)


def load_curated_assignments(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    source_rows = connection.sql(
        """
        SELECT source_id, dataset_id, source_type, title, notes
        FROM knowledge_sources
        ORDER BY source_id
        """
    ).fetchall()
    registry_rows = [
        (
            source_id,
            title,
            "raman_knowledge_core",
            "curated_assignment_pack",
            "knowledge_sources + peak_assignments",
            dataset_id,
            "existing_main_tables",
            "structured_peak_assignment",
            "tier0_curated_assignment",
            True,
            notes,
        )
        for source_id, dataset_id, _source_type, title, notes in source_rows
    ]
    _insert_rows(connection, "registry.evidence_sources", registry_rows, 11)

    assignment_rows = connection.sql(
        """
        SELECT p.assignment_id, p.source_id, p.dataset_id, p.peak_cm, p.tolerance_cm,
               p.assigned_molecule, p.assigned_group, p.matrix_context, p.confidence_text,
               p.evidence_text, s.title, s.notes
        FROM peak_assignments p
        LEFT JOIN knowledge_sources s ON s.source_id = p.source_id
        ORDER BY p.assignment_id
        """
    ).fetchall()

    evidence_items = []
    peak_rows = []
    feature_rows = []
    retrieval_rows = []
    link_rows = []
    for (
        assignment_id,
        source_id,
        dataset_id,
        peak_cm,
        tolerance_cm,
        assigned_molecule,
        assigned_group,
        matrix_context,
        confidence_text,
        evidence_text,
        title,
        notes,
    ) in assignment_rows:
        evidence_item_id = f"evi_curated_{assignment_id}"
        label = assigned_molecule or assigned_group or assignment_id
        evidence_items.append(
            (
                evidence_item_id,
                source_id,
                assignment_id,
                "peak_assignment",
                "tier0_curated_assignment",
                confidence_text,
                label,
                "main.peak_assignments",
                assignment_id,
                True,
                "load_curated_assignments",
                notes,
            )
        )
        peak_rows.append(
            (
                evidence_item_id,
                source_id,
                assignment_id,
                "curated_assignment",
                dataset_id,
                peak_cm,
                None,
                None,
                tolerance_cm,
                assigned_molecule,
                assigned_group,
                "biofluids",
                "Raman",
                "",
                matrix_context,
                "curated_note",
                "",
                "",
                "curated_seed",
                confidence_text,
                evidence_text,
                True,
                notes,
            )
        )
        feature_rows.append(
            (
                f"feat_curated_{assignment_id}",
                evidence_item_id,
                source_id,
                "peak",
                peak_cm,
                None,
                None,
                tolerance_cm,
                1.0,
                assigned_molecule or assigned_group,
                "curated_assignment",
                False,
                "",
            )
        )
        retrieval_rows.append(
            (
                f"doc_curated_{assignment_id}",
                evidence_item_id,
                source_id,
                "peak_assignment",
                "tier0_curated_assignment",
                label,
                evidence_text,
                _domain_from_text(matrix_context),
                _modality_from_text(matrix_context, "Raman"),
                True,
                json.dumps(
                    {
                        "assignment_id": assignment_id,
                        "dataset_id": dataset_id,
                        "source_table": "peak_assignments",
                        "source_title": title,
                    },
                    sort_keys=True,
                ),
            )
        )
        bsv_id, link_weight, rationale = _bsv_from_text(assigned_group)
        if bsv_id:
            link_rows.append(
                (
                    f"link_{evidence_item_id}_{bsv_id}",
                    evidence_item_id,
                    bsv_id,
                    "curated_assignment_group",
                    "high",
                    link_weight,
                    rationale,
                )
            )

    _insert_rows(connection, "evidence.evidence_items", evidence_items, 12)
    _insert_rows(connection, "evidence.peak_assignment_evidence", peak_rows, 23)
    _insert_rows(connection, "features.spectral_features", feature_rows, 13)
    _insert_rows(connection, "retrieval.retrieval_documents", retrieval_rows, 11)
    _insert_rows(connection, "interpretation.evidence_bsv_links", link_rows, 7)
    return {
        "curated_sources_loaded": len(registry_rows),
        "curated_assignments_loaded": len(assignment_rows),
        "curated_features_loaded": len(feature_rows),
    }


def load_source_backed_assignments(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    valid_rows = _read_csv_rows(SOURCE_BACKED_VALID_PATH)
    mention_rows = _read_csv_rows(SOURCE_BACKED_MENTIONS_PATH)
    noise_rows = _read_csv_rows(SOURCE_BACKED_NOISE_PATH)

    unique_source_ids = sorted({row["source_id"] for row in valid_rows + mention_rows})
    registry_rows = [
        (
            source_id,
            source_id.replace("src_", "").replace("_", " "),
            _normalize_text(next((row["study_family"] for row in valid_rows + mention_rows if row["source_id"] == source_id), "")),
            "source_backed_regex_extract",
            str(SOURCE_BACKED_VALID_PATH),
            "",
            "processed_gaira_source_backed_evidence_v1_corrected",
            "text_regex_assignment_or_mention",
            "tier2_source_backed_assignment",
            False,
            "Lower-confidence than curated assignments; mention rows remain excluded from direct retrieval.",
        )
        for source_id in unique_source_ids
    ]
    _insert_rows(connection, "registry.evidence_sources", registry_rows, 11)

    evidence_items = []
    peak_rows = []
    feature_rows = []
    retrieval_rows = []
    mention_evidence_items = []
    mention_table_rows = []
    link_rows = []

    for row in valid_rows:
        evidence_item_id = f"evi_source_{row['evidence_id']}"
        peak_center = _safe_float(row["wavenumber_cm"])
        peak_min = _safe_float(row["wavenumber_min_cm"])
        peak_max = _safe_float(row["wavenumber_max_cm"])
        tolerance_cm = 10.0
        assigned_group = row["biochemical_theme"]
        evidence_items.append(
            (
                evidence_item_id,
                row["source_id"],
                row["evidence_id"],
                "peak_assignment",
                "tier2_source_backed_assignment",
                row["confidence"],
                row["assigned_molecule"] or row["biochemical_theme"] or row["study_family"],
                str(SOURCE_BACKED_VALID_PATH),
                row["evidence_id"],
                True,
                "load_source_backed_assignments",
                row["classification_reason"],
            )
        )
        peak_rows.append(
            (
                evidence_item_id,
                row["source_id"],
                row["evidence_id"],
                "source_backed_regex",
                row["study_family"],
                peak_center,
                peak_min,
                peak_max,
                tolerance_cm,
                row["assigned_molecule"],
                assigned_group,
                row["sample_type"],
                row["modality"],
                row["substrate"],
                row["matrix_context"],
                row["manuscript_or_si"],
                row["figure_or_table_ref"],
                row["page_or_sheet"],
                row["extraction_method"],
                row["confidence"],
                row["evidence_text"],
                True,
                row["classification_reason"],
            )
        )
        feature_rows.append(
            (
                f"feat_source_{row['evidence_id']}",
                evidence_item_id,
                row["source_id"],
                "peak",
                peak_center,
                peak_min,
                peak_max,
                tolerance_cm,
                0.8,
                row["assigned_molecule"] or assigned_group,
                "source_backed_regex",
                _normalize_text(row["assigned_molecule"]).lower() in {"", "unknown"},
                row["classification_reason"],
            )
        )
        retrieval_rows.append(
            (
                f"doc_source_{row['evidence_id']}",
                evidence_item_id,
                row["source_id"],
                "peak_assignment",
                "tier2_source_backed_assignment",
                row["assigned_molecule"] or assigned_group or row["study_family"],
                row["evidence_text"],
                _domain_from_text(row["study_family"], row["sample_type"], row["matrix_context"]),
                _modality_from_text(row["modality"]),
                True,
                json.dumps(
                    {
                        "study_family": row["study_family"],
                        "page_or_sheet": row["page_or_sheet"],
                        "figure_or_table_ref": row["figure_or_table_ref"],
                        "source_file": str(SOURCE_BACKED_VALID_PATH),
                    },
                    sort_keys=True,
                ),
            )
        )
        bsv_id, link_weight, rationale = _bsv_from_text(assigned_group or row["assigned_molecule"])
        if bsv_id:
            link_rows.append(
                (
                    f"link_{evidence_item_id}_{bsv_id}",
                    evidence_item_id,
                    bsv_id,
                    "source_backed_theme",
                    "medium",
                    link_weight * 0.75,
                    rationale,
                )
            )

    for row in mention_rows:
        evidence_item_id = f"evi_mention_{row['evidence_id']}"
        mention_evidence_items.append(
            (
                evidence_item_id,
                row["source_id"],
                row["evidence_id"],
                "wavenumber_mention",
                "tier3_mention_only",
                row["confidence"],
                row["study_family"],
                str(SOURCE_BACKED_MENTIONS_PATH),
                row["evidence_id"],
                False,
                "load_source_backed_assignments",
                "excluded_from_direct_retrieval",
            )
        )
        mention_table_rows.append(
            (
                evidence_item_id,
                row["source_id"],
                row["evidence_id"],
                row["study_family"],
                _safe_float(row["wavenumber_cm"]),
                _safe_float(row["wavenumber_min_cm"]),
                _safe_float(row["wavenumber_max_cm"]),
                row["evidence_text"],
                row["assigned_molecule"],
                row["biochemical_theme"],
                row["sample_type"],
                row["modality"],
                row["manuscript_or_si"],
                row["page_or_sheet"],
                row["extraction_method"],
                row["confidence"],
                row["classification"],
                "wavenumber_mentions_live_in_a_separate_low_confidence_table",
                row["classification_reason"],
            )
        )

    _insert_rows(connection, "evidence.evidence_items", evidence_items + mention_evidence_items, 12)
    _insert_rows(connection, "evidence.peak_assignment_evidence", peak_rows, 23)
    _insert_rows(connection, "features.spectral_features", feature_rows, 13)
    _insert_rows(connection, "retrieval.retrieval_documents", retrieval_rows, 11)
    _insert_rows(connection, "evidence.wavenumber_mentions", mention_table_rows, 19)
    _insert_rows(connection, "interpretation.evidence_bsv_links", link_rows, 7)
    return {
        "source_backed_valid_assignments_loaded": len(valid_rows),
        "wavenumber_mentions_loaded": len(mention_rows),
        "noise_mentions_observed_not_loaded": len(noise_rows),
        "source_backed_features_loaded": len(feature_rows),
    }


def load_ramanbiolib_bridge(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    registry_row = (
        "ramanbiolib_reference_bridge",
        "RamanBioLib reference bridge",
        "ramanbiolib",
        "reference_spectrum_bridge",
        "main.reference_metadata + main.reference_peaks + main.reference_spectra",
        "ramanbiolib",
        "existing_main_tables",
        "reference_evidence",
        "tier1_reference_spectrum",
        True,
        "Bridged from existing reference tables; treated as reference evidence, not biological dataset evidence.",
    )
    _insert_rows(connection, "registry.evidence_sources", [registry_row], 11)

    ref_rows = connection.sql(
        """
        SELECT m.ref_id, m.source_row_id, m.component, m.biochemical_class, m.source, m.reference,
               m.sample_substrate, m.raman_technique, s.preprocessing_summary, s.x_min, s.x_max,
               s.n_points, COALESCE(p.peak_count, 0) AS peak_count
        FROM reference_metadata m
        LEFT JOIN reference_spectra s ON s.ref_id = m.ref_id
        LEFT JOIN (
            SELECT ref_id, COUNT(*) AS peak_count
            FROM reference_peaks
            GROUP BY ref_id
        ) p ON p.ref_id = m.ref_id
        ORDER BY m.ref_id
        """
    ).fetchall()
    peak_rows = connection.sql(
        """
        SELECT ref_id, peak_rank, peak_cm, rel_intensity, component
        FROM reference_peaks
        ORDER BY ref_id, peak_rank
        """
    ).fetchall()

    evidence_items = []
    ref_evidence_rows = []
    feature_rows = []
    retrieval_rows = []
    link_rows = []

    for (
        ref_id,
        source_row_id,
        component,
        biochemical_class,
        source_origin,
        reference,
        sample_substrate,
        modality,
        preprocessing_summary,
        x_min,
        x_max,
        n_points,
        peak_count,
    ) in ref_rows:
        evidence_item_id = f"evi_ref_{ref_id}"
        evidence_items.append(
            (
                evidence_item_id,
                "ramanbiolib_reference_bridge",
                ref_id,
                "reference_spectrum",
                "tier1_reference_spectrum",
                "reference",
                component,
                "main.reference_metadata",
                ref_id,
                True,
                "load_ramanbiolib_bridge",
                "Reference spectrum evidence only.",
            )
        )
        ref_evidence_rows.append(
            (
                evidence_item_id,
                "ramanbiolib_reference_bridge",
                ref_id,
                source_row_id,
                component,
                biochemical_class,
                source_origin,
                reference,
                sample_substrate,
                modality,
                preprocessing_summary,
                x_min,
                x_max,
                n_points,
                peak_count,
                True,
                "",
            )
        )
        retrieval_rows.append(
            (
                f"doc_ref_{ref_id}",
                evidence_item_id,
                "ramanbiolib_reference_bridge",
                "reference_spectrum",
                "tier1_reference_spectrum",
                component,
                f"{component} ({biochemical_class}) reference spectrum from RamanBioLib with {peak_count} extracted peaks.",
                "reference",
                _modality_from_text(modality),
                True,
                json.dumps(
                    {
                        "ref_id": ref_id,
                        "reference": reference,
                        "source_origin": source_origin,
                        "source_table": "reference_metadata/reference_peaks/reference_spectra",
                    },
                    sort_keys=True,
                ),
            )
        )
        bsv_id, link_weight, rationale = _bsv_from_text(biochemical_class)
        if bsv_id:
            link_rows.append(
                (
                    f"link_{evidence_item_id}_{bsv_id}",
                    evidence_item_id,
                    bsv_id,
                    "ramanbiolib_biochemical_class",
                    "medium",
                    link_weight * 0.85,
                    rationale,
                )
            )

    for ref_id, peak_rank, peak_cm, rel_intensity, component in peak_rows:
        evidence_item_id = f"evi_ref_{ref_id}"
        feature_rows.append(
            (
                f"feat_ref_{ref_id}_{peak_rank}",
                evidence_item_id,
                "ramanbiolib_reference_bridge",
                "peak",
                peak_cm,
                None,
                None,
                8.0,
                rel_intensity,
                component,
                "ramanbiolib_peak",
                False,
                "",
            )
        )

    _insert_rows(connection, "evidence.evidence_items", evidence_items, 12)
    _insert_rows(connection, "evidence.reference_spectrum_evidence", ref_evidence_rows, 17)
    _insert_rows(connection, "features.spectral_features", feature_rows, 13)
    _insert_rows(connection, "retrieval.retrieval_documents", retrieval_rows, 11)
    _insert_rows(connection, "interpretation.evidence_bsv_links", link_rows, 7)
    return {
        "ramanbiolib_reference_items_loaded": len(ref_rows),
        "ramanbiolib_reference_features_loaded": len(feature_rows),
    }


def load_digitization_registry(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    rows = _read_csv_rows(DIGITIZATION_QUEUE_PATH)
    source_rows = [
        (
            source_id,
            source_id.replace("src_", "").replace("_", " "),
            next((row["study_family"] for row in rows if row["source_id"] == source_id), ""),
            "digitization_queue",
            str(DIGITIZATION_QUEUE_PATH),
            "",
            "processed_gaira_source_backed_evidence_v1_corrected",
            "future_digitization_candidate",
            "queue_only",
            False,
            "Metadata only; no spectrum arrays loaded.",
        )
        for source_id in sorted({row["source_id"] for row in rows})
    ]
    _insert_rows(connection, "registry.evidence_sources", source_rows, 11)

    evidence_items = []
    table_rows = []
    for row in rows:
        evidence_item_id = f"evi_digitize_{row['queue_id']}"
        evidence_items.append(
            (
                evidence_item_id,
                row["source_id"],
                row["queue_id"],
                "digitization_candidate",
                "queue_only",
                row["priority"],
                f"{row['study_family']} {row['figure_ref']}",
                str(DIGITIZATION_QUEUE_PATH),
                row["queue_id"],
                False,
                "load_digitization_registry",
                row["priority_reason"],
            )
        )
        table_rows.append(
            (
                evidence_item_id,
                row["source_id"],
                row["queue_id"],
                row["study_family"],
                row["figure_ref"],
                row["is_spectral_figure"],
                row["has_source_data_in_dataset_layer"],
                row["is_methods_paper"],
                row["priority"],
                row["priority_reason"],
                row["digitization_status"],
                row["notes"],
            )
        )
    _insert_rows(connection, "evidence.evidence_items", evidence_items, 12)
    _insert_rows(connection, "evidence.digitized_spectrum_registry", table_rows, 12)
    return {
        "digitization_registry_rows_loaded": len(rows),
        "digitization_high_priority": Counter(row["priority"] for row in rows)["high_priority_digitize"],
        "digitization_medium_priority": Counter(row["priority"] for row in rows)["medium_priority_digitize"],
        "digitization_low_priority": Counter(row["priority"] for row in rows)["low_priority_or_redundant"],
    }


def load_context_rules(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    source_rows = [
        (
            "existing_domain_context_pack",
            "Existing GAIRA domain context",
            "domain_context",
            "context_rule_pack",
            "main.dataset_context + main.domain_context_documents + main.domain_context_chunks",
            "",
            "existing_main_tables",
            "context_only",
            "context_only",
            False,
            "Context rules are append-only and never direct spectral evidence.",
        )
    ]
    _insert_rows(connection, "registry.evidence_sources", source_rows, 11)

    evidence_items = []
    context_rows = []
    retrieval_rows = []

    dataset_context_rows = connection.sql(
        "SELECT * FROM dataset_context ORDER BY context_id"
    ).fetchall()
    for (
        context_id,
        dataset_id,
        target_dataset_id,
        modality,
        sample_type,
        measurement_state,
        substrate_type,
        enhancement_mode,
        known_biases,
        caution_450_700,
        caution_700_900,
        caution_900_1100,
        caution_1100_1300,
        caution_1300_1500,
        caution_1500_1700,
        interpretation_note,
        do_not_overclaim_note,
    ) in dataset_context_rows:
        region_rules = [
            ("region_caution", 10, 450.0, 700.0, caution_450_700),
            ("region_caution", 20, 700.0, 900.0, caution_700_900),
            ("region_caution", 30, 900.0, 1100.0, caution_900_1100),
            ("region_caution", 40, 1100.0, 1300.0, caution_1100_1300),
            ("region_caution", 50, 1300.0, 1500.0, caution_1300_1500),
            ("region_caution", 60, 1500.0, 1700.0, caution_1500_1700),
            ("interpretation_note", 70, None, None, interpretation_note),
            ("overclaim_caution", 80, None, None, do_not_overclaim_note),
        ]
        for rule_type, rule_priority, start_cm, end_cm, rule_text in region_rules:
            rule_id = f"{context_id}_{rule_type}_{rule_priority}"
            evidence_item_id = f"evi_context_{rule_id}"
            evidence_items.append(
                (
                    evidence_item_id,
                    "existing_domain_context_pack",
                    rule_id,
                    "context_rule",
                    "context_only",
                    "context_only",
                    f"{dataset_id}:{rule_type}",
                    "main.dataset_context",
                    rule_id,
                    False,
                    "load_context_rules",
                    known_biases,
                )
            )
            context_rows.append(
                (
                    rule_id,
                    evidence_item_id,
                    "existing_domain_context_pack",
                    context_id,
                    _domain_from_text(target_dataset_id, sample_type),
                    _modality_from_text(modality),
                    rule_type,
                    rule_priority,
                    start_cm,
                    end_cm,
                    rule_text,
                    "dataset_context",
                    True,
                    False,
                    f"measurement_state={measurement_state}; substrate_type={substrate_type}; enhancement_mode={enhancement_mode}",
                )
            )
            retrieval_rows.append(
                (
                    f"doc_context_{rule_id}",
                    evidence_item_id,
                    "existing_domain_context_pack",
                    "context_rule",
                    "context_only",
                    f"{dataset_id}:{rule_type}",
                    rule_text,
                    _domain_from_text(target_dataset_id, sample_type),
                    _modality_from_text(modality),
                    False,
                    json.dumps({"context_id": context_id, "source_table": "dataset_context"}, sort_keys=True),
                )
            )

    chunk_rows = connection.sql(
        f"""
        SELECT d.document_id, d.intended_domain, d.context_type, d.evidence_basis, c.section, c.chunk_text
        FROM domain_context_documents d
        JOIN domain_context_chunks c
          ON c.document_id = d.document_id
        WHERE d.document_id IN ({", ".join(["?"] * len(MINIMAL_CONTEXT_DOC_IDS))})
          AND c.chunk_order = 1
        ORDER BY d.document_id
        """,
        params=list(MINIMAL_CONTEXT_DOC_IDS),
    ).fetchall()
    for document_id, intended_domain, context_type, evidence_basis, section, chunk_text in chunk_rows:
        rule_id = f"docrule_{document_id}"
        evidence_item_id = f"evi_context_{rule_id}"
        evidence_items.append(
            (
                evidence_item_id,
                "existing_domain_context_pack",
                document_id,
                "context_rule",
                "context_only",
                "context_only",
                document_id,
                "main.domain_context_documents/domain_context_chunks",
                document_id,
                False,
                "load_context_rules",
                section,
            )
        )
        context_rows.append(
            (
                rule_id,
                evidence_item_id,
                "existing_domain_context_pack",
                document_id,
                intended_domain,
                "unknown",
                context_type,
                200,
                None,
                None,
                chunk_text,
                evidence_basis,
                True,
                False,
                section,
            )
        )
        retrieval_rows.append(
            (
                f"doc_context_{rule_id}",
                evidence_item_id,
                "existing_domain_context_pack",
                "context_rule",
                "context_only",
                document_id,
                chunk_text,
                intended_domain,
                "unknown",
                False,
                json.dumps({"document_id": document_id, "source_table": "domain_context_chunks"}, sort_keys=True),
            )
        )

    _insert_rows(connection, "evidence.evidence_items", evidence_items, 12)
    _insert_rows(connection, "evidence.context_rules", context_rows, 15)
    _insert_rows(connection, "retrieval.retrieval_documents", retrieval_rows, 11)
    return {"context_rules_loaded": len(context_rows)}


def load_bsv_definitions(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    rows = [
        ("bsv_protein", "Protein / Amide", "biopolymer", "Broad proteinaceous or amide-associated evidence band family.", "Do not overclaim single proteins from overlapping peaks."),
        ("bsv_nucleic_acid", "Nucleic Acid", "biopolymer", "DNA/RNA/base-associated broad evidence family.", "Base-specific naming remains provisional unless multi-peak support agrees."),
        ("bsv_amino_acid", "Amino Acid / Aromatic", "molecule_family", "Amino-acid-associated or aromatic residue evidence family.", "Single aromatic peaks are rarely unique in biosamples."),
        ("bsv_lipid", "Lipid / Membrane", "molecule_family", "Broad lipid, fatty-acid, or membrane-associated evidence family.", "CH-rich regions overlap strongly with proteins and carbohydrates."),
        ("bsv_carbohydrate", "Carbohydrate / Saccharide", "molecule_family", "Broad carbohydrate or saccharide-associated evidence family.", "Support should be region-level rather than literal monosaccharide identification."),
        ("bsv_hormone", "Hormone-like Lipid", "molecule_family", "Hormone-associated lipid reference family from curated notes or reference packs.", "Treat as analog evidence only."),
    ]
    _insert_rows(connection, "interpretation.bsv_definitions", rows, 5)
    return {"bsv_definitions_loaded": len(rows)}


def load_all(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    counts = {}
    counts.update(load_curated_assignments(connection))
    counts.update(load_source_backed_assignments(connection))
    counts.update(load_ramanbiolib_bridge(connection))
    counts.update(load_digitization_registry(connection))
    counts.update(load_context_rules(connection))
    counts.update(load_bsv_definitions(connection))
    with SOURCE_BACKED_NOTE_PATH.open("r", encoding="utf-8") as handle:
        source_backed_note = handle.read()
    with DIGITIZATION_NOTE_PATH.open("r", encoding="utf-8") as handle:
        digitization_note = handle.read()
    counts["source_backed_note_chars"] = len(source_backed_note)
    counts["digitization_note_chars"] = len(digitization_note)
    return counts
