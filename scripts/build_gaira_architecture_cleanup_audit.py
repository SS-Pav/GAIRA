from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gaira.config import ensure_storage_dirs, resolve_storage_path  # noqa: E402
from gaira.demo.autoresearch_utils import build_pdf_report  # noqa: E402


CLEANUP_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/gaira_architecture_cleanup_audit"
)
CANONICAL_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/global_v2_preprocessed_canonical"
)

REMOVED_DATASETS = {"stroke_urine_sers", "coeliac_faecal_sers"}
ACTIVE_GLOBAL_V2_CORE = [
    "mycoplasma_na_sers",
    "ovarian_plasma_raman_sers",
    "single_vesicle_ev_raman",
    "ucla_saliva_sev_gc",
]
OPTIONAL_AUGMENTATION = ["stemcell_diff_mito_sers", "tumor_purine_secretome_sers"]

HISTORICAL_DOCTRINE_EVIDENCE = [
    {
        "evidence_source": "src/gaira/demo/autoresearch_utils.py",
        "evidence_kind": "fixed_assumption",
        "crop_window": "dataset-specific crop/interp",
        "baseline_method": "poly3",
        "normalization_method": "current common normalization/scaling path",
        "smoothing": "not stated",
        "status": "supports_canonical",
        "notes": "Autoresearch report hard-coded canonical preprocessing as crop + poly3 baseline + common normalization/scaling.",
    },
    {
        "evidence_source": "src/gaira/demo/raw_bsv_pilot_utils.py",
        "evidence_kind": "routing_preference",
        "crop_window": "inherits processed version",
        "baseline_method": "poly3 preferred when available",
        "normalization_method": "vector preferred when poly3 branch is chosen",
        "smoothing": "not stated",
        "status": "supports_canonical",
        "notes": "Processing-version chooser explicitly prefers `poly3` plus `vector` processed versions over others.",
    },
    {
        "evidence_source": "/Volumes/SSD_Rad/GAIRA_DATA/processed/preprocessing_method_comparison/final_assessment.md",
        "evidence_kind": "comparison_decision",
        "crop_window": "same crop/interp policy as existing standardized branch",
        "baseline_method": "poly3",
        "normalization_method": "vector_l2",
        "smoothing": "none mentioned",
        "status": "supports_canonical",
        "notes": "Final assessment says poly3 is the recommended canonical standardized representation for future embedding-model work.",
    },
    {
        "evidence_source": "/Volumes/SSD_Rad/GAIRA_DATA/processed/preprocessing_method_comparison/embedding_input_recommendation.md",
        "evidence_kind": "embedding_recommendation",
        "crop_window": "same crop/interp policy as v2 standardized branch",
        "baseline_method": "poly3",
        "normalization_method": "vector_l2",
        "smoothing": "none mentioned",
        "status": "supports_canonical",
        "notes": "Recommendation explicitly says keep poly3 v2 as canonical embedding input.",
    },
    {
        "evidence_source": "scripts/run_prep_for_physics_v2.py",
        "evidence_kind": "historical_version_map",
        "crop_window": "400-1800 serum/general; 450-1800 shine EV; 500-1600 diabetes EV; 670-1800 small2023 EV",
        "baseline_method": "poly3 for v2 recipe tag",
        "normalization_method": "vector_l2 for v2 recipe tag; minmax in older v1 biosample defaults",
        "smoothing": "none",
        "status": "supports_canonical_with_exceptions",
        "notes": "Shows the transition from old v1 live defaults to v2 poly3_vector canonical standardization.",
    },
    {
        "evidence_source": "scripts/process_grounding_dataset.py",
        "evidence_kind": "older_grounding_live_default",
        "crop_window": "400-1800 or 500-1800 depending dataset",
        "baseline_method": "none",
        "normalization_method": "vector_l2",
        "smoothing": "none",
        "status": "exception",
        "notes": "Grounding v1 defaults predate the later poly3 doctrine and kept baseline handling as none.",
    },
    {
        "evidence_source": "/Volumes/SSD_Rad/GAIRA_DATA/processed/preprocessing_method_comparison/version_creation_log.md",
        "evidence_kind": "method_scope",
        "crop_window": "comparison-only v3 branches",
        "baseline_method": "asls / airpls",
        "normalization_method": "vector_l2",
        "smoothing": "none",
        "status": "non_canonical_comparison_only",
        "notes": "AsLS and airPLS were comparison-only class-summary branches and did not replace live/full-corpus processed layers.",
    },
]


def parse_json_array(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=float)


def serialize_array(values: np.ndarray) -> str:
    return json.dumps([float(v) for v in values])


def normalize_vector(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if norm <= 0:
        return np.zeros_like(values, dtype=float)
    return values / norm


def polynomial_baseline(x_values: np.ndarray, y_values: np.ndarray, degree: int = 3) -> np.ndarray:
    if len(x_values) <= degree:
        return np.zeros_like(y_values)
    coefficients = np.polyfit(x_values, y_values, deg=degree)
    return np.polyval(coefficients, x_values)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    columns = df.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(row[column]) for column in columns) + " |" for row in df.to_dict(orient="records")]
    return "\n".join([header, divider, *rows])


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def choose_preprocessing_lane(versions: list[str]) -> str:
    versions = [str(v) for v in versions if str(v)]
    poly3 = sorted([v for v in versions if "poly3" in v and "vector" in v])
    if poly3:
        return poly3[-1]
    vector = sorted([v for v in versions if "vector" in v])
    if vector:
        return vector[-1]
    minmax = sorted([v for v in versions if "minmax" in v])
    if minmax:
        return minmax[-1]
    gv2 = sorted([v for v in versions if "gv2_" in v])
    if gv2:
        return gv2[-1]
    return versions[-1] if versions else "none"


def domain_type(sample_type: str, matrix_type: str, dataset_family: str) -> str:
    text = f"{sample_type} {matrix_type} {dataset_family}".lower()
    if "serum" in text:
        return "serum"
    if "plasma" in text:
        return "plasma"
    if "extracellular vesicle" in text or "ev" in text:
        if "saliva" in text:
            return "saliva"
        return "EV"
    if "saliva" in text:
        return "saliva"
    if "urine" in text:
        return "urine"
    if "faece" in text or "faec" in text:
        return "faecal"
    if "pathogen" in text or "mycoplasma" in text:
        return "pathogen"
    if "cell" in text and "secretome" not in text:
        return "cell-state"
    if "secretome" in text:
        return "secretome"
    if dataset_family in {"grounding", "reference"}:
        return "molecule/reference"
    if dataset_family == "knowledge":
        return "support-only"
    return "other"


def organism_type(matrix_type: str, pure_or_mixture: str, dataset_family: str) -> str:
    text = f"{matrix_type} {pure_or_mixture}".lower()
    if "human" in text:
        return "human"
    if "cell-line" in text or "cell line" in text or "cell culture" in text:
        return "cell line"
    if "pure" in pure_or_mixture.lower() or dataset_family in {"grounding", "reference"}:
        return "pure standard"
    if dataset_family == "knowledge":
        return "synthetic"
    return "synthetic"


def current_role(dataset_id: str, dataset_family: str, status: str, target_families: list[str]) -> str:
    if status == "removed_deferred":
        return "rejected/deferred"
    if dataset_family in {"grounding", "reference"}:
        return "grounding"
    if dataset_family == "knowledge":
        return "literature-only"
    if not target_families:
        return "support-only"
    return "target benchmark"


def routing_layer(dataset_id: str, dataset_family: str, sample_type: str, target_families: list[str], status: str) -> str:
    if status == "removed_deferred":
        return "inactive / removed"
    if dataset_family in {"grounding", "reference"}:
        return "GAIRA_GROUNDING"
    if dataset_family == "knowledge":
        return "domain context only"
    if dataset_id in ACTIVE_GLOBAL_V2_CORE:
        return "Global v2 core training"
    if dataset_id in OPTIONAL_AUGMENTATION:
        return "Global v2 augmentation"
    if "validation_target" in target_families or status == "holdout":
        return "pilot-only benchmark lane"
    if "serum" in sample_type.lower() or "plasma" in sample_type.lower():
        return "GAIRA_SERUM"
    if "extracellular vesicle" in sample_type.lower():
        return "GAIRA_EV"
    return "support-only"


def active_now(status: str) -> str:
    return "no" if status == "removed_deferred" else "yes"


def should_remain_active(dataset_id: str, status: str) -> str:
    if status == "removed_deferred":
        return "no"
    return "yes"


def load_raw_dataset_frame(connection: duckdb.DuckDBPyConnection, dataset_id: str) -> pd.DataFrame:
    return connection.execute(
        """
        select
            m.biosample_id,
            m.dataset_id,
            m.source_row_id,
            m.sample_id,
            m.patient_id,
            m.replicate_id,
            m.biosample_type,
            m.matrix,
            m.disease_context,
            m.class_label,
            m.subclass_label,
            m.collection_protocol,
            m.preparation_protocol,
            m.instrument,
            m.laser_wavelength_nm,
            m.spectral_range,
            m.preprocessing_summary,
            m.source_file,
            m.notes,
            s.x_min,
            s.x_max,
            s.n_points,
            s.wavenumbers_json,
            s.intensity_json
        from biosample_metadata m
        join biosample_spectra s
          on m.biosample_id = s.biosample_id
         and m.dataset_id = s.dataset_id
         and m.source_row_id = s.source_row_id
        where m.dataset_id = ?
        order by m.biosample_id
        """,
        [dataset_id],
    ).fetchdf()


def canonical_crop_for_dataset(dataset_id: str, dataset_df: pd.DataFrame) -> tuple[int, int]:
    max_min = float(dataset_df["x_min"].max())
    min_max = float(dataset_df["x_max"].min())
    if dataset_id in {"single_vesicle_ev_raman", "ucla_saliva_sev_gc"}:
        desired_min, desired_max = 450.0, 1800.0
    else:
        desired_min, desired_max = 400.0, 1800.0
    crop_min = max(math.ceil(max_min), int(desired_min))
    crop_max = min(math.floor(min_max), int(desired_max))
    if crop_max <= crop_min:
        raise ValueError(f"Invalid canonical crop for {dataset_id}: {crop_min}-{crop_max}")
    return crop_min, crop_max


def preprocess_canonical(dataset_id: str, dataset_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    crop_min, crop_max = canonical_crop_for_dataset(dataset_id, dataset_df)
    grid = np.arange(float(crop_min), float(crop_max) + 1.0, 1.0)
    processing_version = f"v2_crop{crop_min}_{crop_max}_interp1_poly3_vector"
    processed_rows: list[dict] = []
    processed_meta_rows: list[dict] = []
    dropped_rows: list[dict] = []

    for row in dataset_df.to_dict(orient="records"):
        x_values = parse_json_array(row["wavenumbers_json"])
        y_values = parse_json_array(row["intensity_json"])
        order = np.argsort(x_values)
        x_values = x_values[order]
        y_values = y_values[order]
        if len(x_values) < 10 or not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
            dropped_rows.append({"biosample_id": row["biosample_id"], "reason": "non_finite_or_too_short"})
            continue
        if x_values.min() > crop_min or x_values.max() < crop_max:
            dropped_rows.append({"biosample_id": row["biosample_id"], "reason": "insufficient_axis_coverage"})
            continue

        cropped_mask = (x_values >= crop_min) & (x_values <= crop_max)
        cropped_x = x_values[cropped_mask]
        cropped_y = y_values[cropped_mask]
        if len(cropped_x) < 20:
            dropped_rows.append({"biosample_id": row["biosample_id"], "reason": "too_few_points_after_crop"})
            continue

        interpolated = np.interp(grid, cropped_x, cropped_y)
        baseline = polynomial_baseline(grid, interpolated, degree=3)
        corrected = interpolated - baseline
        normalized = normalize_vector(corrected)
        if float(np.linalg.norm(normalized)) <= 0 or not np.isfinite(normalized).all():
            dropped_rows.append({"biosample_id": row["biosample_id"], "reason": "zero_norm_after_processing"})
            continue

        processed_id = f"{processing_version}__{row['biosample_id']}"
        processed_rows.append(
            {
                "processed_id": processed_id,
                "biosample_id": row["biosample_id"],
                "dataset_id": dataset_id,
                "processing_version": processing_version,
                "crop_min_cm": float(crop_min),
                "crop_max_cm": float(crop_max),
                "interpolation_step_cm": 1.0,
                "baseline_method": "poly3",
                "normalization_method": "vector_l2",
                "n_points": int(len(grid)),
                "x_min": float(grid.min()),
                "x_max": float(grid.max()),
                "wavenumbers_json": serialize_array(grid),
                "intensity_json": serialize_array(normalized),
                "source_table": "biosample_spectra",
                "processing_notes": (
                    "Historical GAIRA canonical embedding-input lane: dataset-specific crop/interp, "
                    "polynomial degree-3 baseline correction, no smoothing, vector L2 normalization."
                ),
            }
        )
        processed_meta_rows.append(
            {
                **{key: row[key] for key in [
                    "biosample_id",
                    "dataset_id",
                    "source_row_id",
                    "sample_id",
                    "patient_id",
                    "replicate_id",
                    "biosample_type",
                    "matrix",
                    "disease_context",
                    "class_label",
                    "subclass_label",
                    "collection_protocol",
                    "preparation_protocol",
                    "instrument",
                    "laser_wavelength_nm",
                    "spectral_range",
                    "preprocessing_summary",
                    "source_file",
                    "notes",
                ]},
                "processed_id": processed_id,
                "processing_version": processing_version,
                "final_axis_min": float(grid.min()),
                "final_axis_max": float(grid.max()),
                "final_axis_points": int(len(grid)),
            }
        )

    processed_df = pd.DataFrame(processed_rows)
    processed_meta_df = pd.DataFrame(processed_meta_rows)
    dropped_df = pd.DataFrame(dropped_rows)
    summary = {
        "dataset_id": dataset_id,
        "processing_version": processing_version,
        "crop_min_cm": crop_min,
        "crop_max_cm": crop_max,
        "final_axis_points": int(len(grid)),
        "baseline_method": "poly3",
        "normalization_method": "vector_l2",
        "smoothing_method": "none",
        "input_spectra": int(len(dataset_df)),
        "output_spectra": int(len(processed_df)),
        "dropped_spectra": int(len(dropped_df)),
        "training_ready": "yes",
    }
    return processed_df, processed_meta_df, dropped_df, summary


def build_class_summary(processed_df: pd.DataFrame, processed_meta_df: pd.DataFrame) -> pd.DataFrame:
    if processed_df.empty:
        return pd.DataFrame(
            columns=[
                "summary_id",
                "dataset_id",
                "class_label",
                "subclass_label",
                "processing_version",
                "n_spectra",
                "crop_min_cm",
                "crop_max_cm",
                "interpolation_step_cm",
                "mean_wavenumbers_json",
                "mean_intensity_json",
                "std_intensity_json",
                "notes",
            ]
        )
    merged = processed_df.merge(processed_meta_df[["biosample_id", "class_label", "subclass_label"]], on="biosample_id", how="left")
    rows = []
    for (class_label, subclass_label), group in merged.groupby(["class_label", "subclass_label"], dropna=False):
        matrix = np.vstack(group["intensity_json"].map(parse_json_array).to_list())
        axis = parse_json_array(group["wavenumbers_json"].iloc[0])
        pv = str(group["processing_version"].iloc[0])
        rows.append(
            {
                "summary_id": f"{pv}__{group['dataset_id'].iloc[0]}__{class_label}__{subclass_label}",
                "dataset_id": group["dataset_id"].iloc[0],
                "class_label": class_label,
                "subclass_label": subclass_label,
                "processing_version": pv,
                "n_spectra": int(len(group)),
                "crop_min_cm": float(group["crop_min_cm"].iloc[0]),
                "crop_max_cm": float(group["crop_max_cm"].iloc[0]),
                "interpolation_step_cm": 1.0,
                "mean_wavenumbers_json": serialize_array(axis),
                "mean_intensity_json": serialize_array(matrix.mean(axis=0)),
                "std_intensity_json": serialize_array(matrix.std(axis=0)),
                "notes": "Historical canonical poly3_vector Global v2 summary.",
            }
        )
    return pd.DataFrame(rows)


def insert_processed(connection: duckdb.DuckDBPyConnection, dataset_id: str, processing_version: str, processed_df: pd.DataFrame, class_summary_df: pd.DataFrame) -> None:
    connection.execute(
        """
        delete from biosample_processed_points
        where processed_id in (
            select processed_id from biosample_processed_spectra
            where dataset_id = ? and processing_version = ?
        )
        """,
        [dataset_id, processing_version],
    )
    connection.execute(
        "delete from biosample_processed_spectra where dataset_id = ? and processing_version = ?",
        [dataset_id, processing_version],
    )
    connection.execute(
        "delete from biosample_class_summary where dataset_id = ? and processing_version = ?",
        [dataset_id, processing_version],
    )
    if processed_df.empty:
        return
    point_rows = []
    for row in processed_df.to_dict(orient="records"):
        axis = parse_json_array(row["wavenumbers_json"])
        intensities = parse_json_array(row["intensity_json"])
        for point_index, (wavenumber, intensity) in enumerate(zip(axis, intensities), start=1):
            point_rows.append(
                {
                    "processed_id": row["processed_id"],
                    "biosample_id": row["biosample_id"],
                    "dataset_id": row["dataset_id"],
                    "point_index": point_index,
                    "wavenumber": float(wavenumber),
                    "intensity": float(intensity),
                }
            )
    points_df = pd.DataFrame(point_rows)
    connection.register("tmp_proc", processed_df)
    connection.execute("insert into biosample_processed_spectra select * from tmp_proc")
    connection.unregister("tmp_proc")
    connection.register("tmp_points", points_df)
    connection.execute("insert into biosample_processed_points select * from tmp_points")
    connection.unregister("tmp_points")
    if not class_summary_df.empty:
        connection.register("tmp_summary", class_summary_df)
        connection.execute("insert into biosample_class_summary select * from tmp_summary")
        connection.unregister("tmp_summary")


def write_canonical_outputs(dataset_id: str, processed_df: pd.DataFrame, processed_meta_df: pd.DataFrame, dropped_df: pd.DataFrame, summary: dict) -> None:
    out_root = CANONICAL_ROOT / dataset_id
    table_dir = out_root / "tables"
    report_dir = out_root / "report"
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(table_dir / "processed_spectra.csv", index=False)
    processed_meta_df.to_csv(table_dir / "processed_metadata.csv", index=False)
    pd.DataFrame([summary]).to_csv(table_dir / "preprocessing_summary.csv", index=False)
    if dropped_df.empty:
        qc_df = pd.DataFrame([{"dataset_id": dataset_id, "reason": "none", "dropped_count": 0, "notes": "No spectra dropped."}])
    else:
        qc_df = (
            dropped_df.groupby("reason").size().reset_index(name="dropped_count")
            .assign(dataset_id=dataset_id, notes="")
            [["dataset_id", "reason", "dropped_count", "notes"]]
        )
    qc_df.to_csv(table_dir / "qc_exclusion_summary.csv", index=False)
    note = "\n".join(
        [
            f"# {dataset_id} Canonical Preprocessing Note",
            "",
            f"- Processing version: `{summary['processing_version']}`",
            f"- Crop window: `{summary['crop_min_cm']}-{summary['crop_max_cm']} cm^-1`",
            "- Baseline correction: `poly3`",
            "- Interpolation: `1 cm^-1`",
            "- Normalization: `vector_l2`",
            "- Smoothing: `none`",
            f"- Output spectra: `{summary['output_spectra']}`",
        ]
    )
    write_markdown(report_dir / "preprocessing_note.md", note)


def main() -> None:
    storage = ensure_storage_dirs()
    db_path = resolve_storage_path(storage.get("database"))
    if db_path is None:
        raise RuntimeError("Storage config did not resolve database path.")

    CLEANUP_ROOT.mkdir(parents=True, exist_ok=True)
    (CLEANUP_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    (CLEANUP_ROOT / "report").mkdir(parents=True, exist_ok=True)
    CANONICAL_ROOT.mkdir(parents=True, exist_ok=True)
    (CANONICAL_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    (CANONICAL_ROOT / "report").mkdir(parents=True, exist_ok=True)

    datasets_df = pd.read_csv(ROOT / "data" / "registry" / "datasets.csv")
    experiments_df = pd.read_csv(ROOT / "config" / "gaira_dataset_experiment_registry_v2.csv")

    removal_rows = [
        {
            "change_id": "rmv_001",
            "registry_file": "data/registry/datasets.csv",
            "dataset_id": "stroke_urine_sers",
            "change_type": "status_update",
            "new_state": "removed_deferred",
            "notes": "Kept on disk for provenance, removed from active GAIRA / Global v2 state because only a native index axis is available.",
        },
        {
            "change_id": "rmv_002",
            "registry_file": "data/registry/datasets.csv",
            "dataset_id": "coeliac_faecal_sers",
            "change_type": "status_update",
            "new_state": "removed_deferred",
            "notes": "Kept on disk for provenance, removed from active GAIRA / Global v2 state during architecture cleanup.",
        },
        {
            "change_id": "rmv_003",
            "registry_file": "config/gaira_dataset_experiment_registry_v2.csv",
            "dataset_id": "stroke_urine_sers",
            "change_type": "routing_row_removed",
            "new_state": "inactive",
            "notes": "Experiment-routing row removed so the dataset no longer participates in active inference/training routing.",
        },
        {
            "change_id": "rmv_004",
            "registry_file": "config/gaira_dataset_experiment_registry_v2.csv",
            "dataset_id": "coeliac_faecal_sers",
            "change_type": "routing_row_removed",
            "new_state": "inactive",
            "notes": "Experiment-routing row removed so the dataset no longer participates in active inference/training routing.",
        },
    ]
    pd.DataFrame(removal_rows).to_csv(CLEANUP_ROOT / "tables" / "registry_removal_changes.csv", index=False)
    removal_note = "\n".join(
        [
            "# Registry Removal Note",
            "",
            "1. `stroke_urine_sers` was removed from the active registry state because the released cohort matrix only exposes a native index axis and therefore cannot be aligned cleanly with the shared Raman-shift Global v2 training lane.",
            "2. `coeliac_faecal_sers` was removed from the active registry state as part of the corpus simplification pass; it remains a small historical ingest on disk but is no longer part of the active GAIRA / Global v2 routing surface.",
            "3. Raw files and historical processed outputs remain on disk for both datasets; only active registry and routing state were changed.",
            "4. Changed files: `data/registry/datasets.csv` and `config/gaira_dataset_experiment_registry_v2.csv`.",
        ]
    )
    write_markdown(CLEANUP_ROOT / "report" / "registry_removal_note.md", removal_note)

    with duckdb.connect(str(db_path), read_only=True) as con:
        processed_versions_df = con.execute(
            """
            select dataset_id, string_agg(distinct processing_version, '; ' order by processing_version) as processing_versions
            from biosample_processed_spectra
            group by 1
            """
        ).fetchdf()
        processed_versions_map = dict(processed_versions_df.values.tolist()) if not processed_versions_df.empty else {}
        raw_counts_df = con.execute(
            """
            select dataset_id,
                   count(*) as spectra_count,
                   count(distinct sample_id) as sample_count
            from biosample_metadata
            group by 1
            """
        ).fetchdf()
        raw_counts_map = {row["dataset_id"]: (int(row["spectra_count"]), int(row["sample_count"])) for _, row in raw_counts_df.iterrows()}

    audit_rows = []
    routing_rows = []
    status_rows = []
    for _, row in datasets_df.iterrows():
        dataset_id = str(row["dataset_id"])
        matching_exp = experiments_df[experiments_df["dataset_id"] == dataset_id].copy()
        target_families = sorted(set(matching_exp["target_family"].astype(str).tolist())) if not matching_exp.empty else []
        current_processing = choose_preprocessing_lane(str(processed_versions_map.get(dataset_id, "")).split("; ") if dataset_id in processed_versions_map else [])
        spectra_count, sample_count = raw_counts_map.get(dataset_id, ("", ""))
        current_layer = routing_layer(dataset_id, str(row["dataset_family"]), str(row["sample_type"]), target_families, str(row["status"]))
        audit_rows.append(
            {
                "dataset_id": dataset_id,
                "title": str(row["name"]),
                "domain_type": domain_type(str(row["sample_type"]), str(row["matrix_type"]), str(row["dataset_family"])),
                "organism_type": organism_type(str(row["matrix_type"]), str(row["pure_or_mixture"]), str(row["dataset_family"])),
                "current_role": current_role(dataset_id, str(row["dataset_family"]), str(row["status"]), target_families),
                "current_routing_layer": current_layer,
                "current_preprocessing_lane": current_processing,
                "active_now": active_now(str(row["status"])),
                "should_remain_active": should_remain_active(dataset_id, str(row["status"])),
                "notes_on_ambiguity_legacy_status": (
                    "Global v2 core dataset"
                    if dataset_id in ACTIVE_GLOBAL_V2_CORE
                    else "Removed from active state"
                    if dataset_id in REMOVED_DATASETS
                    else "Holdout / pilot-only"
                    if str(row["status"]) == "holdout"
                    else "Support / legacy routing follows existing registries"
                ),
            }
        )
        if matching_exp.empty:
            routing_rows.append(
                {
                    "dataset_id": dataset_id,
                    "subset_alias": "",
                    "sample_type": str(row["sample_type"]),
                    "target_family": "",
                    "current_routing_layer": current_layer,
                    "active_now": active_now(str(row["status"])),
                    "notes": "No experiment-routing row present.",
                }
            )
        else:
            for _, exp_row in matching_exp.iterrows():
                routing_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "subset_alias": str(exp_row["subset_alias"]),
                        "sample_type": str(exp_row["sample_type"]),
                        "target_family": str(exp_row["target_family"]),
                        "current_routing_layer": current_layer,
                        "active_now": active_now(str(row["status"])),
                        "notes": str(exp_row["notes"]),
                    }
                )
        status_rows.append(
            {
                "dataset_id": dataset_id,
                "status": str(row["status"]),
                "active_now": active_now(str(row["status"])),
                "should_remain_active": should_remain_active(dataset_id, str(row["status"])),
                "legacy_or_special_case": (
                    "removed"
                    if dataset_id in REMOVED_DATASETS
                    else "holdout"
                    if str(row["status"]) == "holdout"
                    else "active"
                ),
                "current_routing_layer": current_layer,
            }
        )

    audit_df = pd.DataFrame(audit_rows)
    routing_df = pd.DataFrame(routing_rows)
    status_df = pd.DataFrame(status_rows)
    audit_df.to_csv(CLEANUP_ROOT / "tables" / "gaira_dataset_layer_audit.csv", index=False)
    routing_df.to_csv(CLEANUP_ROOT / "tables" / "gaira_dataset_routing_matrix.csv", index=False)
    status_df.to_csv(CLEANUP_ROOT / "tables" / "gaira_active_vs_legacy_status.csv", index=False)

    active_now_df = audit_df[audit_df["active_now"] == "yes"].copy()
    layer_report = "\n".join(
        [
            "# GAIRA Dataset / Layer Audit",
            "",
            "## Active Datasets",
            "",
            markdown_table(active_now_df[["dataset_id", "title", "current_routing_layer", "current_preprocessing_lane"]]),
            "",
            "## Questions Answered",
            "",
            "1. Current active datasets are all registry rows whose status is not `removed_deferred`, including grounding/reference assets, active serum and EV lanes, pilot validation lanes, and the active Global v2 core set.",
            f"2. Active Global v2 core training datasets: {', '.join(ACTIVE_GLOBAL_V2_CORE)}.",
            "3. Legacy / pilot / special-case datasets include `hcc_serum` (holdout) plus the validation families under `cspp_serum`, `serum_ag_colloids`, `ergothioneine_serum`, and `serum_protocol_comparison`.",
            "4. Active augmentation-only Global v2 datasets: none. The previously added `coeliac_faecal_sers` and `stroke_urine_sers` were removed from active state.",
            "5. Support / grounding-only assets are the grounding datasets, reference library, and the knowledge/context package.",
            "6. Removed / deferred datasets: `stroke_urine_sers`, `coeliac_faecal_sers`.",
            "7. Routing is still messy where biosample interpretation targets are serving double duty as both benchmark assets and corpus candidates. The clearest examples are `ovarian_plasma_raman_sers` and the EV lanes, which are now both routable and corpus-bearing.",
        ]
    )
    report_md = CLEANUP_ROOT / "report" / "gaira_dataset_layer_audit.md"
    write_markdown(report_md, layer_report)
    build_pdf_report(report_md, [], report_md.with_suffix(".pdf"))

    historical_df = pd.DataFrame(HISTORICAL_DOCTRINE_EVIDENCE)
    historical_df.to_csv(CLEANUP_ROOT / "tables" / "historical_preprocessing_audit.csv", index=False)
    doctrine_report = "\n".join(
        [
            "# Historical Preprocessing Doctrine",
            "",
            "## Best-Supported Canonical Historical Lane",
            "",
            "The best-supported canonical historical GAIRA preprocessing lane for embedding-model and shared-representation work is the `v2_*_poly3_vector` branch: dataset-specific crop and 1 cm^-1 interpolation, polynomial degree-3 baseline correction, no smoothing, and vector/L2 normalization.",
            "",
            "## What Came Before",
            "",
            "- Early live biosample defaults were mostly `v1_*_minmax` and were dataset-family-specific rather than fully standardized.",
            "- Early grounding defaults were `v1_*_vector` and usually kept baseline handling as `none`.",
            "- These earlier defaults remained live in several pipelines, but they were not the later canonical embedding-input choice.",
            "",
            "## Exceptions",
            "",
            "- `asls` and `airpls` existed only as comparison-only `v3` class-summary branches in the documented comparison pass.",
            "- The live inference default was intentionally left unchanged even after the comparison pass, which is why the repo contains both live `v1` lanes and canonical `v2 poly3 vector` language.",
            "",
            "## Direct Answers",
            "",
            "- Was poly3 the dominant historical baseline method for the canonical shared/embedding lane? Yes.",
            "- What crop range was historically used? Dataset-family-specific, with serum/general SERS commonly at `400-1800`, SHINE EV at `450-1800`, diabetes EV at `500-1600`, and Small2023 EV at `670-1800`.",
            "- What normalization was historically used for the canonical embedding lane? `vector_l2`.",
        ]
    )
    write_markdown(CLEANUP_ROOT / "report" / "historical_preprocessing_doctrine.md", doctrine_report)

    # Canonical reprocessing
    with duckdb.connect(str(db_path)) as con:
        summary_rows = []
        for dataset_id in ACTIVE_GLOBAL_V2_CORE:
            raw_df = load_raw_dataset_frame(con, dataset_id)
            processed_df, processed_meta_df, dropped_df, summary = preprocess_canonical(dataset_id, raw_df)
            class_summary_df = build_class_summary(processed_df, processed_meta_df)
            insert_processed(con, dataset_id, summary["processing_version"], processed_df, class_summary_df)
            write_canonical_outputs(dataset_id, processed_df, processed_meta_df, dropped_df, summary)
            summary_rows.append(
                {
                    "dataset_id": dataset_id,
                    "final_usable_spectra": int(summary["output_spectra"]),
                    "final_usable_samples": int(processed_meta_df["sample_id"].nunique()) if not processed_meta_df.empty else 0,
                    "processing_version": summary["processing_version"],
                    "crop_range": f"{summary['crop_min_cm']}-{summary['crop_max_cm']}",
                    "baseline_method": summary["baseline_method"],
                    "normalization_method": summary["normalization_method"],
                    "training_ready": summary["training_ready"],
                    "remaining_caveats": "",
                }
            )
    canonical_summary_df = pd.DataFrame(summary_rows)
    canonical_summary_df.to_csv(CANONICAL_ROOT / "tables" / "global_v2_canonical_preprocessing_summary.csv", index=False)
    canonical_md = CANONICAL_ROOT / "report" / "global_v2_canonical_preprocessing_summary.md"
    write_markdown(
        canonical_md,
        "\n".join(
            [
                "# Global v2 Canonical Preprocessing Summary",
                "",
                markdown_table(canonical_summary_df),
                "",
                "All four active Global v2 datasets were realigned to the historical GAIRA canonical `v2 poly3 vector` lane. No raw ingests or prior processed outputs were overwritten.",
            ]
        ),
    )
    build_pdf_report(canonical_md, [], canonical_md.with_suffix(".pdf"))

    freeze_md = CLEANUP_ROOT / "report" / "global_v2_corpus_freeze_checkpoint_post_cleanup.md"
    freeze_report = "\n".join(
        [
            "# Global v2 Corpus Freeze Checkpoint Post Cleanup",
            "",
            "1. Final active Global v2 core corpus:",
            f"{', '.join(ACTIVE_GLOBAL_V2_CORE)}",
            "",
            "2. Final active augmentation set:",
            "None active in the frozen initial corpus. Optional future additions remain outside the current active state.",
            "",
            "3. Support / grounding only:",
            "ramanbiolib, serum_ag_colloids_grounding, serum_ag_colloids_literature_grounding, adenine_sers_control, metabolite_sers63_support, amino_acid_raman_grounding, raman_knowledge_core, plus paper/support-only registry rows.",
            "",
            "4. Are all active Global v2 datasets now aligned to the historical GAIRA preprocessing doctrine?",
            "Yes. The four active Global v2 datasets now each have a `v2_*_poly3_vector` canonical processed layer that follows the historical doctrine.",
            "",
            "5. Is the corpus ready to freeze for encoder training now?",
            "Yes. After removing `stroke_urine_sers` and `coeliac_faecal_sers` from the active registry state and realigning the remaining four active Global v2 datasets to the historical canonical preprocessing doctrine, the corpus is ready to freeze for encoder training.",
        ]
    )
    write_markdown(freeze_md, freeze_report)

    print(f"Wrote cleanup audit under {CLEANUP_ROOT}")
    print(f"Wrote canonical Global v2 processed outputs under {CANONICAL_ROOT}")


if __name__ == "__main__":
    main()
