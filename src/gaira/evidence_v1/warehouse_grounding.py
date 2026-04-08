from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from gaira.evidence_v1.assignment_patterns import build_assignment_patterns
from gaira.evidence_v1.phase1_refinement import build_phase1_refinement
from gaira.evidence_v1.schema import initialize_schema, reset_phase1_refinement_tables


DATASET_REGISTRY_PATH = Path("/Users/suraj/projects/GAIRA/data/registry/datasets.csv")

CANONICAL_GROUNDING_VERSIONS = {
    "adenine_sers_control": "v1_crop400_1800_interp1_vector",
    "amino_acid_raman_grounding": "v1_crop400_1800_interp1_vector",
    "metabolite_sers63_support": "v1_crop500_1800_interp1_vector",
    "serum_ag_colloids_grounding": "v1_crop400_1800_interp1_vector",
}

GROUNDING_DATASET_ROUTES = {
    "adenine_sers_control": {
        "source_family": "reference_molecule",
        "sample_scope": "controlled_reference_grounding",
        "biosample_type": "none",
        "modality": "sers",
        "is_reference_grounding": True,
        "is_disease_or_stress_evidence": False,
        "disease_class": "",
        "stress_class": "",
        "digitization_required": False,
    },
    "amino_acid_raman_grounding": {
        "source_family": "reference_molecule",
        "sample_scope": "controlled_reference_grounding",
        "biosample_type": "none",
        "modality": "raman",
        "is_reference_grounding": True,
        "is_disease_or_stress_evidence": False,
        "disease_class": "",
        "stress_class": "",
        "digitization_required": False,
    },
    "metabolite_sers63_support": {
        "source_family": "reference_molecule",
        "sample_scope": "controlled_reference_grounding",
        "biosample_type": "none",
        "modality": "sers",
        "is_reference_grounding": True,
        "is_disease_or_stress_evidence": False,
        "disease_class": "",
        "stress_class": "",
        "digitization_required": False,
    },
    "serum_ag_colloids_grounding": {
        "source_family": "serum_grounding",
        "sample_scope": "serum_grounding_support",
        "biosample_type": "serum",
        "modality": "sers",
        "is_reference_grounding": False,
        "is_disease_or_stress_evidence": False,
        "disease_class": "",
        "stress_class": "adsorption_protocol_control",
        "digitization_required": False,
    },
    "serum_ag_colloids_literature_grounding": {
        "source_family": "disease_or_stress_paper",
        "sample_scope": "structured_biological_paper",
        "biosample_type": "serum",
        "modality": "mixed",
        "is_reference_grounding": False,
        "is_disease_or_stress_evidence": True,
        "disease_class": "mixed_serum_disease_context",
        "stress_class": "",
        "digitization_required": False,
    },
}


@dataclass
class GroundingBridgeCounts:
    source_id: str
    source_family: str
    summary_spectra_added: int = 0
    peak_assignment_rows_added: int = 0
    feature_rows_added: int = 0


def _read_dataset_registry() -> dict[str, dict[str, str]]:
    with DATASET_REGISTRY_PATH.open("r", encoding="utf-8", newline="") as handle:
        return {row["dataset_id"]: row for row in csv.DictReader(handle)}


def _normalize_text(value: str | None) -> str:
    return (value or "").strip()


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _delete_where_in(connection: duckdb.DuckDBPyConnection, table_name: str, column_name: str, values: list[str]) -> None:
    if not values:
        return
    placeholders = ", ".join(["?"] * len(values))
    connection.execute(f"DELETE FROM {table_name} WHERE {column_name} IN ({placeholders})", values)


def _detect_peaks(
    wavenumbers: list[float],
    intensities: list[float],
    modality: str,
    dataset_id: str,
) -> list[dict[str, float]]:
    x_values = np.asarray(wavenumbers, dtype=float)
    y_values = np.asarray(intensities, dtype=float)
    if len(x_values) < 25:
        return []
    shifted = y_values - float(np.min(y_values))
    max_value = float(np.max(shifted))
    if max_value <= 0:
        return []
    normalized = shifted / max_value

    smoothing_window = 9
    if dataset_id == "serum_ag_colloids_grounding":
        smoothing_window = 11
    elif dataset_id in {"metabolite_sers63_support", "adenine_sers_control"}:
        smoothing_window = 7
    kernel = np.ones(smoothing_window, dtype=float) / float(smoothing_window)
    smoothed = np.convolve(normalized, kernel, mode="same")

    if dataset_id == "serum_ag_colloids_grounding":
        prominence = 0.14
        height = 0.08
        distance = 10
    elif dataset_id == "metabolite_sers63_support":
        prominence = 0.09
        height = 0.06
        distance = 8
    elif dataset_id == "adenine_sers_control":
        prominence = 0.07
        height = 0.05
        distance = 8
    else:
        prominence = 0.06 if modality.lower() == "sers" else 0.05
        height = 0.05 if modality.lower() == "sers" else 0.04
        distance = 8 if modality.lower() == "sers" else 10
    peak_indices, properties = find_peaks(
        smoothed,
        prominence=prominence,
        height=height,
        distance=distance,
    )
    rows = []
    for rank, peak_index in enumerate(peak_indices, start=1):
        peak_cm = float(x_values[peak_index])
        if peak_cm < 400.0 or peak_cm > 1800.0:
            continue
        rows.append(
            {
                "peak_rank": rank,
                "peak_cm": peak_cm,
                "peak_height": float(properties["peak_heights"][rank - 1]),
                "prominence": float(properties["prominences"][rank - 1]),
            }
        )
    return rows


def _infer_biosample_type(text: str) -> str:
    lowered = text.lower()
    if "serum" in lowered:
        return "serum"
    if "extracellular vesicle" in lowered or " exosome" in lowered or "_ev_" in lowered or lowered.startswith("ev_"):
        return "ev"
    if "plasma" in lowered:
        return "plasma"
    if "pathogen" in lowered or "mycoplasma" in lowered:
        return "pathogen"
    return "none"


def _infer_disease_class(text: str) -> str:
    lowered = text.lower()
    if "lung" in lowered:
        return "lung_disease_or_cancer"
    if "breast" in lowered:
        return "breast_disease_or_cancer"
    if "cca" in lowered or "cholangio" in lowered:
        return "cholangiocarcinoma_context"
    if "hcc" in lowered:
        return "hepatocellular_carcinoma_context"
    if "covid" in lowered or "hbv" in lowered or "virus" in lowered:
        return "infectious_disease_context"
    if "diabetes" in lowered:
        return "metabolic_disease_context"
    if "ovarian" in lowered:
        return "ovarian_cancer_context"
    if "stroke" in lowered:
        return "stroke_context"
    if "coeliac" in lowered:
        return "coeliac_context"
    return ""


def _infer_stress_class(text: str) -> str:
    lowered = text.lower()
    if "shine" in lowered or "hepatotox" in lowered or "apap" in lowered:
        return "drug_stress"
    if "interlab" in lowered or "protocol" in lowered:
        return "protocol_variation"
    if "adsorption" in lowered:
        return "adsorption_bias"
    return ""


def _route_existing_source(
    source_id: str,
    source_name: str,
    parent_dataset_id: str,
    source_path: str,
    source_kind: str,
    notes: str,
    digitization_required: bool,
) -> tuple[str, str, str, bool, bool, str, str]:
    if source_id in GROUNDING_DATASET_ROUTES:
        route = GROUNDING_DATASET_ROUTES[source_id]
        return (
            route["source_family"],
            route["sample_scope"],
            route["biosample_type"],
            route["is_reference_grounding"],
            route["is_disease_or_stress_evidence"],
            route["disease_class"],
            route["stress_class"],
        )
    text = " ".join(
        [
            _normalize_text(source_id),
            _normalize_text(source_name),
            _normalize_text(parent_dataset_id),
            _normalize_text(source_path),
            _normalize_text(notes),
        ]
    ).lower()
    if any(term in text for term in ("ramanbiolib", "ref_db_biomolecules", "raman_ir_handbook", "curated seed", "raman_knowledge_core")):
        return "reference_molecule", "interpretive_reference", _infer_biosample_type(text), True, False, "", _infer_stress_class(text)
    return "disease_or_stress_paper", "structured_biological_paper", _infer_biosample_type(text), False, True, _infer_disease_class(text), _infer_stress_class(text)


def ensure_warehouse_source_registry(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    dataset_registry = _read_dataset_registry()

    current_sources = connection.sql(
        """
        SELECT source_id, source_name, source_kind, source_path, parent_dataset_id, notes
        FROM registry.evidence_sources
        ORDER BY source_id
        """
    ).fetchall()

    digitization_required_ids = {
        row[0]
        for row in connection.sql(
            """
            SELECT DISTINCT source_id
            FROM evidence.evidence_items
            WHERE evidence_kind = 'digitization_candidate'
            """
        ).fetchall()
    }

    warehouse_rows: list[tuple] = []
    seen_source_ids: set[str] = set()
    for source_id, source_name, source_kind, source_path, parent_dataset_id, notes in current_sources:
        if source_id == "existing_domain_context_pack":
            continue
        family, sample_scope, biosample_type, is_reference, is_disease, disease_class, stress_class = _route_existing_source(
            source_id=source_id,
            source_name=source_name,
            parent_dataset_id=parent_dataset_id,
            source_path=source_path,
            source_kind=source_kind,
            notes=notes,
            digitization_required=source_id in digitization_required_ids,
        )
        structured_peak_assignments = source_kind in {"curated_assignment_pack", "source_backed_regex_extract"} or source_id == "ramanbiolib_reference_bridge"
        structured_spectral_data = source_id == "ramanbiolib_reference_bridge"
        modality = "mixed" if "handbook" in source_id or source_kind == "source_backed_regex_extract" else "raman"
        if "sers" in source_id:
            modality = "sers"
        warehouse_rows.append(
            (
                source_id,
                source_name,
                family,
                sample_scope,
                biosample_type or "none",
                modality,
                is_reference,
                is_disease,
                disease_class,
                stress_class,
                structured_spectral_data,
                structured_peak_assignments,
                source_id in digitization_required_ids,
                source_path or "existing_evidence_tables",
                source_kind,
                parent_dataset_id,
                notes,
            )
        )
        seen_source_ids.add(source_id)

    for dataset_id, route in GROUNDING_DATASET_ROUTES.items():
        if dataset_id in seen_source_ids:
            continue
        metadata = dataset_registry.get(dataset_id, {})
        source_name = metadata.get("name") or dataset_id
        notes = metadata.get("notes") or ""
        structured_spectral = dataset_id != "serum_ag_colloids_literature_grounding"
        structured_assignments = dataset_id != "serum_ag_colloids_literature_grounding"
        warehouse_rows.append(
            (
                dataset_id,
                source_name,
                route["source_family"],
                route["sample_scope"],
                route["biosample_type"],
                route["modality"],
                route["is_reference_grounding"],
                route["is_disease_or_stress_evidence"],
                route["disease_class"],
                route["stress_class"],
                structured_spectral or dataset_id == "serum_ag_colloids_literature_grounding",
                structured_assignments,
                route["digitization_required"],
                "main.grounding_* tables",
                "grounding_dataset_bridge",
                dataset_id,
                notes,
            )
        )

    connection.execute("DELETE FROM registry.warehouse_sources")
    connection.executemany(
        "INSERT INTO registry.warehouse_sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        warehouse_rows,
    )
    family_counts = Counter(row[2] for row in warehouse_rows)
    return {
        "warehouse_sources_total": len(warehouse_rows),
        **{f"warehouse_sources_{family}": count for family, count in sorted(family_counts.items())},
    }


def ensure_grounding_source_rows(connection: duckdb.DuckDBPyConnection) -> None:
    dataset_registry = _read_dataset_registry()
    for dataset_id, route in GROUNDING_DATASET_ROUTES.items():
        if connection.sql(
            "SELECT COUNT(*) FROM registry.evidence_sources WHERE source_id = ?",
            params=[dataset_id],
        ).fetchone()[0]:
            continue
        metadata = dataset_registry.get(dataset_id, {})
        source_name = metadata.get("name") or dataset_id
        direct_allowed = dataset_id != "serum_ag_colloids_literature_grounding"
        default_tier = {
            "reference_molecule": "tier1_grounding_reference",
            "serum_grounding": "tier1_grounding_support",
            "disease_or_stress_paper": "tier2_grounding_support",
        }[route["source_family"]]
        connection.execute(
            """
            INSERT INTO registry.evidence_sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                dataset_id,
                source_name,
                route["source_family"],
                "grounding_dataset_bridge",
                "main.grounding_class_summary + main.grounding_metadata",
                dataset_id,
                "existing_main_grounding_tables",
                "structured_grounding_peak_support",
                default_tier,
                direct_allowed,
                metadata.get("notes") or "",
            ],
        )


def _fetch_grounding_summary_rows(connection: duckdb.DuckDBPyConnection) -> list[dict]:
    rows = []
    for dataset_id, version in CANONICAL_GROUNDING_VERSIONS.items():
        query = connection.sql(
            """
            WITH meta AS (
                SELECT
                    dataset_id,
                    experiment_family,
                    class_label,
                    ANY_VALUE(source_dataset_id) AS source_dataset_id,
                    ANY_VALUE(grounding_role) AS grounding_role,
                    ANY_VALUE(modality) AS modality,
                    ANY_VALUE(compound_label) AS compound_label,
                    ANY_VALUE(biosample_context) AS biosample_context
                FROM grounding_metadata
                WHERE dataset_id = ?
                GROUP BY dataset_id, experiment_family, class_label
            )
            SELECT
                gcs.summary_id,
                gcs.dataset_id,
                meta.source_dataset_id,
                gcs.experiment_family,
                gcs.class_label,
                meta.compound_label,
                meta.grounding_role,
                meta.modality,
                meta.biosample_context,
                gcs.processing_version,
                gcs.n_spectra,
                gcs.crop_min_cm,
                gcs.crop_max_cm,
                gcs.mean_wavenumbers_json,
                gcs.mean_intensity_json,
                gcs.notes
            FROM grounding_class_summary gcs
            LEFT JOIN meta
              ON meta.dataset_id = gcs.dataset_id
             AND meta.experiment_family = gcs.experiment_family
             AND meta.class_label = gcs.class_label
            WHERE gcs.dataset_id = ?
              AND gcs.processing_version = ?
            ORDER BY gcs.experiment_family, gcs.class_label, gcs.summary_id
            """,
            params=[dataset_id, dataset_id, version],
        ).fetchall()
        for row in query:
            rows.append(
                {
                    "summary_id": row[0],
                    "dataset_id": row[1],
                    "source_dataset_id": row[2] or row[1],
                    "experiment_family": row[3],
                    "class_label": row[4],
                    "compound_label": row[5] or row[4],
                    "grounding_role": row[6] or "grounding_reference",
                    "modality": _normalize_text(row[7]) or GROUNDING_DATASET_ROUTES[dataset_id]["modality"],
                    "biosample_context": row[8] or "",
                    "processing_version": row[9],
                    "n_spectra": int(row[10]),
                    "crop_min_cm": float(row[11]),
                    "crop_max_cm": float(row[12]),
                    "mean_wavenumbers": json.loads(row[13]),
                    "mean_intensity": json.loads(row[14]),
                    "notes": row[15] or "",
                }
            )
    return rows


def bridge_existing_grounding_datasets(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    ensure_grounding_source_rows(connection)
    source_ids = sorted(CANONICAL_GROUNDING_VERSIONS)
    _delete_where_in(connection, "features.spectral_features", "source_id", source_ids)
    _delete_where_in(connection, "evidence.peak_assignment_evidence", "source_id", source_ids)
    _delete_where_in(connection, "evidence.grounding_spectrum_evidence", "source_id", source_ids)
    connection.execute(
        """
        DELETE FROM evidence.evidence_items
        WHERE source_id IN ({placeholders})
          AND created_by = 'bridge_existing_grounding_datasets'
        """.format(placeholders=", ".join(["?"] * len(source_ids))),
        source_ids,
    )

    spectrum_evidence_items = []
    peak_evidence_items = []
    grounding_spectrum_rows = []
    peak_rows = []
    feature_rows = []
    bridge_counts: dict[str, GroundingBridgeCounts] = {
        dataset_id: GroundingBridgeCounts(source_id=dataset_id, source_family=GROUNDING_DATASET_ROUTES[dataset_id]["source_family"])
        for dataset_id in source_ids
    }

    for summary in _fetch_grounding_summary_rows(connection):
        route = GROUNDING_DATASET_ROUTES[summary["dataset_id"]]
        biosample_type = route["biosample_type"]
        spectrum_item_id = f"evi_grounding_spectrum_{summary['summary_id']}"
        spectrum_label = f"{summary['class_label']} summary spectrum"
        spectrum_note = (
            f"Structured grounding summary from {summary['dataset_id']} / {summary['experiment_family']} / "
            f"{summary['class_label']} using {summary['processing_version']}."
        )
        peaks = _detect_peaks(
            summary["mean_wavenumbers"],
            summary["mean_intensity"],
            route["modality"],
            summary["dataset_id"],
        )

        spectrum_evidence_items.append(
            (
                spectrum_item_id,
                summary["dataset_id"],
                summary["summary_id"],
                "grounding_spectrum",
                "tier1_grounding_reference" if route["source_family"] == "reference_molecule" else "tier1_grounding_support",
                "medium",
                spectrum_label,
                "main.grounding_class_summary",
                summary["summary_id"],
                False,
                "bridge_existing_grounding_datasets",
                spectrum_note,
            )
        )
        grounding_spectrum_rows.append(
            (
                spectrum_item_id,
                summary["dataset_id"],
                summary["summary_id"],
                summary["dataset_id"],
                summary["source_dataset_id"],
                route["source_family"],
                route["sample_scope"],
                biosample_type,
                summary["compound_label"],
                summary["class_label"],
                summary["experiment_family"],
                summary["grounding_role"],
                route["modality"],
                summary["biosample_context"],
                summary["processing_version"],
                summary["crop_min_cm"],
                summary["crop_max_cm"],
                len(summary["mean_wavenumbers"]),
                len(peaks),
                True,
                spectrum_note,
            )
        )
        bridge_counts[summary["dataset_id"]].summary_spectra_added += 1

        for peak in peaks:
            peak_item_id = f"evi_grounding_peak_{summary['summary_id']}_{peak['peak_rank']:03d}"
            assignment_origin = (
                "reference_grounding_peak"
                if route["source_family"] == "reference_molecule"
                else "serum_grounding_peak"
            )
            context_key = f"{summary['dataset_id']}::{summary['experiment_family']}::{summary['class_label']}"
            peak_note = (
                f"Detected on canonical mean summary spectrum for {summary['class_label']} in {summary['dataset_id']}. "
                f"Peak detection is structured-spectrum-derived, not a verbatim paper assignment."
            )
            peak_evidence_items.append(
                (
                    peak_item_id,
                    summary["dataset_id"],
                    f"{summary['summary_id']}::{peak['peak_rank']}",
                    "peak_assignment",
                    "tier1_grounding_reference" if route["source_family"] == "reference_molecule" else "tier1_grounding_support",
                    "medium",
                    f"{summary['class_label']} peak {peak['peak_cm']:.1f} cm^-1",
                    "main.grounding_class_summary",
                    summary["summary_id"],
                    True,
                    "bridge_existing_grounding_datasets",
                    peak_note,
                )
            )
            peak_rows.append(
                (
                    peak_item_id,
                    summary["dataset_id"],
                    f"{summary['summary_id']}::{peak['peak_rank']}",
                    assignment_origin,
                    context_key,
                    peak["peak_cm"],
                    max(summary["crop_min_cm"], peak["peak_cm"] - 6.0),
                    min(summary["crop_max_cm"], peak["peak_cm"] + 6.0),
                    8.0,
                    summary["class_label"],
                    summary["class_label"],
                    biosample_type,
                    route["modality"],
                    "",
                    summary["biosample_context"],
                    "structured_grounding_summary",
                    summary["experiment_family"],
                    summary["processing_version"],
                    "grounding_summary_peak_detection",
                    "medium",
                    (
                        f"{summary['class_label']} support from canonical grounding summary spectrum in "
                        f"{summary['dataset_id']} ({summary['experiment_family']})."
                    ),
                    True,
                    peak_note,
                )
            )
            feature_rows.append(
                (
                    f"feat_grounding_{summary['summary_id']}_{peak['peak_rank']:03d}",
                    peak_item_id,
                    summary["dataset_id"],
                    "peak",
                    peak["peak_cm"],
                    max(summary["crop_min_cm"], peak["peak_cm"] - 6.0),
                    min(summary["crop_max_cm"], peak["peak_cm"] + 6.0),
                    8.0,
                    peak["prominence"],
                    summary["class_label"],
                    assignment_origin,
                    False,
                    peak_note,
                )
            )
            bridge_counts[summary["dataset_id"]].peak_assignment_rows_added += 1
            bridge_counts[summary["dataset_id"]].feature_rows_added += 1

    if spectrum_evidence_items:
        connection.executemany("INSERT INTO evidence.evidence_items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", spectrum_evidence_items)
    if peak_evidence_items:
        connection.executemany("INSERT INTO evidence.evidence_items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", peak_evidence_items)
    if grounding_spectrum_rows:
        connection.executemany("INSERT INTO evidence.grounding_spectrum_evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", grounding_spectrum_rows)
    if peak_rows:
        connection.executemany("INSERT INTO evidence.peak_assignment_evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", peak_rows)
    if feature_rows:
        connection.executemany("INSERT INTO features.spectral_features VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", feature_rows)

    return {
        "grounding_summary_spectra_added": len(grounding_spectrum_rows),
        "grounding_peak_assignments_added": len(peak_rows),
        "grounding_feature_rows_added": len(feature_rows),
        **{
            f"{dataset_id}_summary_spectra_added": counts.summary_spectra_added
            for dataset_id, counts in sorted(bridge_counts.items())
        },
        **{
            f"{dataset_id}_peak_rows_added": counts.peak_assignment_rows_added
            for dataset_id, counts in sorted(bridge_counts.items())
        },
    }


def rebuild_phase1_from_expanded_grounding(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    reset_phase1_refinement_tables(connection)
    initialize_schema(connection)
    refinement_counts = build_phase1_refinement(connection)
    pattern_counts = build_assignment_patterns(connection)
    return {**refinement_counts, **pattern_counts}


def build_warehouse_backbone(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    initialize_schema(connection)
    routing_counts = ensure_warehouse_source_registry(connection)
    grounding_counts = bridge_existing_grounding_datasets(connection)
    refinement_counts = rebuild_phase1_from_expanded_grounding(connection)
    return {
        **routing_counts,
        **grounding_counts,
        **refinement_counts,
    }


def load_dataframe(connection: duckdb.DuckDBPyConnection, query: str, params: list | None = None) -> pd.DataFrame:
    return connection.sql(query, params=params or []).df()
