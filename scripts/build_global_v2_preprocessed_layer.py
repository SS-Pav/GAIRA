from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gaira.config import ensure_storage_dirs, resolve_storage_path  # noqa: E402
from gaira.demo.autoresearch_utils import build_pdf_report  # noqa: E402


AUDIT_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/global_v2_preprocessing_audit"
)
PREPROCESSED_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/global_v2_preprocessed"
)

FROZEN_DATASETS = [
    "mycoplasma_na_sers",
    "ovarian_plasma_raman_sers",
    "coeliac_faecal_sers",
    "single_vesicle_ev_raman",
    "stroke_urine_sers",
    "ucla_saliva_sev_gc",
]

EXPLICIT_AXIS_DATASETS = [
    "mycoplasma_na_sers",
    "ovarian_plasma_raman_sers",
    "coeliac_faecal_sers",
    "single_vesicle_ev_raman",
    "ucla_saliva_sev_gc",
]

COMMON_EXPLICIT_CROP_MIN = 565.0
COMMON_EXPLICIT_CROP_MAX = 1681.0
COMMON_INTERPOLATION_STEP = 1.0
COMMON_EXPLICIT_GRID = np.arange(
    COMMON_EXPLICIT_CROP_MIN,
    COMMON_EXPLICIT_CROP_MAX + COMMON_INTERPOLATION_STEP,
    COMMON_INTERPOLATION_STEP,
)

EXPLICIT_PROCESSING_VERSION = "gv2_explicitaxis_v1_crop565_1681_interp1_asls_vector"
NATIVE_INDEX_PROCESSING_VERSION = "gv2_nativeindex_v1_fullidx_asls_vector"


def parse_json_array(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=float)


def serialize_array(values: np.ndarray) -> str:
    return json.dumps([float(v) for v in values])


def normalize_vector(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if norm <= 0:
        return np.zeros_like(values, dtype=float)
    return values / norm


def asls_baseline(values: np.ndarray, lam: float = 1e6, p: float = 0.01, niter: int = 15) -> np.ndarray:
    length = len(values)
    if length < 3:
        return np.zeros_like(values)
    diff = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(length - 2, length), format="csc")
    weights = np.ones(length, dtype=float)
    y = np.asarray(values, dtype=float)
    for _ in range(niter):
        weight_matrix = sparse.spdiags(weights, 0, length, length)
        system = weight_matrix + lam * (diff.T @ diff)
        baseline = spsolve(system, weights * y)
        weights = p * (y > baseline) + (1.0 - p) * (y < baseline)
    return np.asarray(baseline, dtype=float)


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


def qc_reason_row(dataset_id: str, reason: str, dropped_count: int, notes: str) -> dict:
    return {
        "dataset_id": dataset_id,
        "reason": reason,
        "dropped_count": int(dropped_count),
        "notes": notes,
    }


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


def explicit_axis_type(_: pd.DataFrame) -> str:
    return "explicit_raman_shift"


def native_axis_type(_: pd.DataFrame) -> str:
    return "native_index_only"


def preprocess_explicit_dataset(dataset_df: pd.DataFrame, dataset_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    processed_rows: list[dict] = []
    processed_meta_rows: list[dict] = []
    dropped_rows: list[dict] = []

    for row in dataset_df.to_dict(orient="records"):
        x_values = parse_json_array(row["wavenumbers_json"])
        y_values = parse_json_array(row["intensity_json"])
        ordered_idx = np.argsort(x_values)
        x_values = x_values[ordered_idx]
        y_values = y_values[ordered_idx]

        if len(x_values) < 10 or not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
            dropped_rows.append({"biosample_id": row["biosample_id"], "reason": "non_finite_or_too_short"})
            continue
        if x_values.min() > COMMON_EXPLICIT_CROP_MIN or x_values.max() < COMMON_EXPLICIT_CROP_MAX:
            dropped_rows.append({"biosample_id": row["biosample_id"], "reason": "insufficient_axis_coverage"})
            continue

        crop_mask = (x_values >= COMMON_EXPLICIT_CROP_MIN) & (x_values <= COMMON_EXPLICIT_CROP_MAX)
        cropped_x = x_values[crop_mask]
        cropped_y = y_values[crop_mask]
        if len(cropped_x) < 20:
            dropped_rows.append({"biosample_id": row["biosample_id"], "reason": "too_few_points_after_crop"})
            continue

        interpolated = np.interp(COMMON_EXPLICIT_GRID, cropped_x, cropped_y)
        baseline = asls_baseline(interpolated)
        corrected = interpolated - baseline
        normalized = normalize_vector(corrected)
        if not np.isfinite(normalized).all() or float(np.linalg.norm(normalized)) <= 0:
            dropped_rows.append({"biosample_id": row["biosample_id"], "reason": "zero_norm_after_processing"})
            continue

        processed_id = f"{EXPLICIT_PROCESSING_VERSION}__{row['biosample_id']}"
        processed_rows.append(
            {
                "processed_id": processed_id,
                "biosample_id": row["biosample_id"],
                "dataset_id": dataset_id,
                "processing_version": EXPLICIT_PROCESSING_VERSION,
                "crop_min_cm": COMMON_EXPLICIT_CROP_MIN,
                "crop_max_cm": COMMON_EXPLICIT_CROP_MAX,
                "interpolation_step_cm": COMMON_INTERPOLATION_STEP,
                "baseline_method": "asls",
                "normalization_method": "vector_l2",
                "n_points": int(len(COMMON_EXPLICIT_GRID)),
                "x_min": float(COMMON_EXPLICIT_GRID.min()),
                "x_max": float(COMMON_EXPLICIT_GRID.max()),
                "wavenumbers_json": serialize_array(COMMON_EXPLICIT_GRID),
                "intensity_json": serialize_array(normalized),
                "source_table": "biosample_spectra",
                "processing_notes": (
                    "Global v2 explicit-axis preprocessing: crop 565-1681 cm^-1, interpolate to 1 cm^-1 grid, "
                    "AsLS baseline correction, no smoothing, vector L2 normalization."
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
                "processing_version": EXPLICIT_PROCESSING_VERSION,
                "axis_type": "explicit_raman_shift",
                "final_axis_min": float(COMMON_EXPLICIT_GRID.min()),
                "final_axis_max": float(COMMON_EXPLICIT_GRID.max()),
                "final_axis_points": int(len(COMMON_EXPLICIT_GRID)),
                "training_ready": "yes",
                "training_caveat": "",
            }
        )

    processed_df = pd.DataFrame(processed_rows)
    processed_meta_df = pd.DataFrame(processed_meta_rows)
    dropped_df = pd.DataFrame(dropped_rows)
    summary = {
        "dataset_id": dataset_id,
        "processing_version": EXPLICIT_PROCESSING_VERSION,
        "input_spectra": int(len(dataset_df)),
        "output_spectra": int(len(processed_df)),
        "dropped_spectra": int(len(dropped_df)),
        "final_axis_min": float(COMMON_EXPLICIT_GRID.min()),
        "final_axis_max": float(COMMON_EXPLICIT_GRID.max()),
        "final_axis_points": int(len(COMMON_EXPLICIT_GRID)),
        "axis_type": "explicit_raman_shift",
        "baseline_method": "asls",
        "normalization_method": "vector_l2",
        "smoothing_method": "none",
        "training_ready": "yes",
    }
    return processed_df, processed_meta_df, dropped_df, summary


def preprocess_native_dataset(dataset_df: pd.DataFrame, dataset_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    processed_rows: list[dict] = []
    processed_meta_rows: list[dict] = []
    dropped_rows: list[dict] = []

    for row in dataset_df.to_dict(orient="records"):
        x_values = parse_json_array(row["wavenumbers_json"])
        y_values = parse_json_array(row["intensity_json"])
        ordered_idx = np.argsort(x_values)
        x_values = x_values[ordered_idx]
        y_values = y_values[ordered_idx]

        if len(x_values) < 10 or not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
            dropped_rows.append({"biosample_id": row["biosample_id"], "reason": "non_finite_or_too_short"})
            continue

        baseline = asls_baseline(y_values)
        corrected = y_values - baseline
        normalized = normalize_vector(corrected)
        if not np.isfinite(normalized).all() or float(np.linalg.norm(normalized)) <= 0:
            dropped_rows.append({"biosample_id": row["biosample_id"], "reason": "zero_norm_after_processing"})
            continue

        processed_id = f"{NATIVE_INDEX_PROCESSING_VERSION}__{row['biosample_id']}"
        processed_rows.append(
            {
                "processed_id": processed_id,
                "biosample_id": row["biosample_id"],
                "dataset_id": dataset_id,
                "processing_version": NATIVE_INDEX_PROCESSING_VERSION,
                "crop_min_cm": float(x_values.min()),
                "crop_max_cm": float(x_values.max()),
                "interpolation_step_cm": None,
                "baseline_method": "asls",
                "normalization_method": "vector_l2",
                "n_points": int(len(x_values)),
                "x_min": float(x_values.min()),
                "x_max": float(x_values.max()),
                "wavenumbers_json": serialize_array(x_values),
                "intensity_json": serialize_array(normalized),
                "source_table": "biosample_spectra",
                "processing_notes": (
                    "Global v2 native-index preprocessing: preserve released 0-4095 native index axis, apply AsLS "
                    "baseline correction and vector L2 normalization, but do not map to Raman shift."
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
                "processing_version": NATIVE_INDEX_PROCESSING_VERSION,
                "axis_type": "native_index_only",
                "final_axis_min": float(x_values.min()),
                "final_axis_max": float(x_values.max()),
                "final_axis_points": int(len(x_values)),
                "training_ready": "no",
                "training_caveat": "native index only; hold out from initial shared-axis Global v2 encoder training",
            }
        )

    processed_df = pd.DataFrame(processed_rows)
    processed_meta_df = pd.DataFrame(processed_meta_rows)
    dropped_df = pd.DataFrame(dropped_rows)
    summary = {
        "dataset_id": dataset_id,
        "processing_version": NATIVE_INDEX_PROCESSING_VERSION,
        "input_spectra": int(len(dataset_df)),
        "output_spectra": int(len(processed_df)),
        "dropped_spectra": int(len(dropped_df)),
        "final_axis_min": 0.0,
        "final_axis_max": 4095.0,
        "final_axis_points": 4096,
        "axis_type": "native_index_only",
        "baseline_method": "asls",
        "normalization_method": "vector_l2",
        "smoothing_method": "none",
        "training_ready": "no",
    }
    return processed_df, processed_meta_df, dropped_df, summary


def class_summary_from_processed(processed_meta_df: pd.DataFrame, processed_df: pd.DataFrame) -> pd.DataFrame:
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
    merged_df = processed_df.merge(
        processed_meta_df[["biosample_id", "class_label", "subclass_label"]],
        on="biosample_id",
        how="left",
    )
    rows: list[dict] = []
    for (class_label, subclass_label), group in merged_df.groupby(["class_label", "subclass_label"], dropna=False):
        intensities = np.vstack(group["intensity_json"].map(parse_json_array).to_list())
        axis = parse_json_array(group["wavenumbers_json"].iloc[0])
        mean_values = intensities.mean(axis=0)
        std_values = intensities.std(axis=0)
        processing_version = str(group["processing_version"].iloc[0])
        crop_min = group["crop_min_cm"].iloc[0]
        crop_max = group["crop_max_cm"].iloc[0]
        step = group["interpolation_step_cm"].iloc[0]
        rows.append(
            {
                "summary_id": f"{processing_version}__{group['dataset_id'].iloc[0]}__{class_label}__{subclass_label}",
                "dataset_id": group["dataset_id"].iloc[0],
                "class_label": class_label,
                "subclass_label": subclass_label,
                "processing_version": processing_version,
                "n_spectra": int(len(group)),
                "crop_min_cm": crop_min,
                "crop_max_cm": crop_max,
                "interpolation_step_cm": step,
                "mean_wavenumbers_json": serialize_array(axis),
                "mean_intensity_json": serialize_array(mean_values),
                "std_intensity_json": serialize_array(std_values),
                "notes": "Global v2 preprocessing class summary.",
            }
        )
    return pd.DataFrame(rows)


def write_dataset_outputs(dataset_id: str, processed_df: pd.DataFrame, processed_meta_df: pd.DataFrame, dropped_df: pd.DataFrame, summary: dict) -> None:
    dataset_root = PREPROCESSED_ROOT / dataset_id
    table_dir = dataset_root / "tables"
    report_dir = dataset_root / "report"
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    processed_df.to_csv(table_dir / "processed_spectra.csv", index=False)
    processed_meta_df.to_csv(table_dir / "processed_metadata.csv", index=False)
    pd.DataFrame([summary]).to_csv(table_dir / "preprocessing_summary.csv", index=False)

    if dropped_df.empty:
        qc_df = pd.DataFrame([qc_reason_row(dataset_id, "none", 0, "No spectra dropped during preprocessing.")])
    else:
        qc_df = (
            dropped_df.groupby("reason")
            .size()
            .reset_index(name="dropped_count")
            .assign(dataset_id=dataset_id, notes="")
            [["dataset_id", "reason", "dropped_count", "notes"]]
        )
    qc_df.to_csv(table_dir / "qc_exclusion_summary.csv", index=False)

    note_lines = [
        f"# {dataset_id} Preprocessing Note",
        "",
        f"- Processing version: `{summary['processing_version']}`",
        f"- Output spectra: {summary['output_spectra']} / {summary['input_spectra']}",
        f"- Final axis: {summary['axis_type']} ({summary['final_axis_min']:.1f} to {summary['final_axis_max']:.1f}; {summary['final_axis_points']} points)",
        f"- Baseline correction: `{summary['baseline_method']}`",
        f"- Normalization: `{summary['normalization_method']}`",
        f"- Smoothing: `{summary['smoothing_method']}`",
        f"- Training-ready: `{summary['training_ready']}`",
    ]
    write_markdown(report_dir / "preprocessing_note.md", "\n".join(note_lines))


def insert_processed_layer(
    connection: duckdb.DuckDBPyConnection,
    dataset_id: str,
    processed_df: pd.DataFrame,
    processed_meta_df: pd.DataFrame,
    class_summary_df: pd.DataFrame,
    processing_version: str,
) -> None:
    connection.execute(
        "DELETE FROM biosample_processed_points WHERE processed_id IN (SELECT processed_id FROM biosample_processed_spectra WHERE dataset_id = ? AND processing_version = ?)",
        [dataset_id, processing_version],
    )
    connection.execute(
        "DELETE FROM biosample_processed_spectra WHERE dataset_id = ? AND processing_version = ?",
        [dataset_id, processing_version],
    )
    connection.execute(
        "DELETE FROM biosample_class_summary WHERE dataset_id = ? AND processing_version = ?",
        [dataset_id, processing_version],
    )

    if not processed_df.empty:
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
        connection.register("tmp_processed_spectra", processed_df)
        connection.execute("INSERT INTO biosample_processed_spectra SELECT * FROM tmp_processed_spectra")
        connection.unregister("tmp_processed_spectra")
        connection.register("tmp_processed_points", points_df)
        connection.execute("INSERT INTO biosample_processed_points SELECT * FROM tmp_processed_points")
        connection.unregister("tmp_processed_points")

    if not class_summary_df.empty:
        connection.register("tmp_class_summary", class_summary_df)
        connection.execute("INSERT INTO biosample_class_summary SELECT * FROM tmp_class_summary")
        connection.unregister("tmp_class_summary")


def build_audit_table(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rows = []
    for dataset_id in FROZEN_DATASETS:
        counts = connection.execute(
            """
            select
                count(*) as spectra_count,
                count(distinct sample_id) as sample_count,
                min(x_min) as min_x,
                max(x_max) as max_x,
                count(distinct n_points) as n_point_variants,
                count(distinct x_min || '|' || x_max) as axis_variants,
                any_value(m.preprocessing_summary) as preprocessing_summary
            from biosample_metadata m
            join biosample_spectra s
              on m.biosample_id = s.biosample_id
             and m.dataset_id = s.dataset_id
             and m.source_row_id = s.source_row_id
            where m.dataset_id = ?
            """,
            [dataset_id],
        ).fetchdf().iloc[0].to_dict()
        axis_type = "native_index_only" if dataset_id == "stroke_urine_sers" else "explicit_raman_shift"
        rows.append(
            {
                "dataset_id": dataset_id,
                "raw_ingest_available": "yes",
                "spectra_count": int(counts["spectra_count"]),
                "sample_count": int(counts["sample_count"]),
                "axis_type": axis_type,
                "baseline_correction_status": "unknown",
                "baseline_method": "",
                "cropping_status": "not_applied",
                "crop_range": "",
                "normalization_status": "not_applied",
                "normalization_method": "",
                "qc_exclusion_status": "parser_level_only",
                "training_ready_now": "no",
                "main_blocker": (
                    "native index only; no shared Raman-shift harmonization"
                    if dataset_id == "stroke_urine_sers"
                    else "no frozen Global v2 processed layer yet"
                ),
                "harmonization_needed": "yes",
            }
        )
    return pd.DataFrame(rows)


def build_audit_report(audit_df: pd.DataFrame) -> str:
    lines = [
        "# Global v2 Preprocessing Audit",
        "",
        "This audit checks the frozen Global v2 corpus after raw ingest and before encoder training. It reflects what is explicitly present in GAIRA, not what might have happened upstream in the original studies.",
        "",
        "## Dataset Audit Table",
        "",
        markdown_table(audit_df),
        "",
        "## Main Findings",
        "",
        "- None of the frozen Global v2 datasets had a GAIRA `biosample_processed_spectra` layer before this harmonization pass.",
        "- Parser-level exclusions were present, but baseline correction, shared cropping, interpolation, and normalization had not been applied consistently inside GAIRA.",
        "- Five datasets expose explicit Raman-shift axes and can be harmonized onto one common Global v2 grid.",
        "- `stroke_urine_sers` remains the outlier because the released cohort matrix only exposes a native 0-4095 index axis.",
    ]
    return "\n".join(lines) + "\n"


def build_spec_report() -> str:
    return "\n".join(
        [
            "# Global v2 Preprocessing Spec",
            "",
            "## Shared Rule",
            "",
            "Global v2 uses one explicit-axis preprocessing lane for all frozen datasets that expose a real Raman-shift axis, and a separate native-index preservation lane for datasets that do not.",
            "",
            "## 1. Axis Alignment Rule",
            "",
            "- Explicit-axis datasets are cropped to the strict per-spectrum overlap window shared across the frozen corpus: `565-1681 cm^-1`.",
            "- Those spectra are interpolated onto a common `1 cm^-1` grid, yielding `1117` points per spectrum.",
            "- Datasets with no explicit Raman-shift axis are not remapped to cm^-1.",
            "",
            "## 2. Crop Window",
            "",
            "- Explicit-axis lane: `565-1681 cm^-1`.",
            "- Native-index lane: no Raman-shift crop; preserve the released native index axis.",
            "",
            "## 3. Baseline Correction Rule",
            "",
            "- Apply AsLS baseline correction with `lambda=1e6`, `p=0.01`, `niter=15`.",
            "- Apply it after interpolation for explicit-axis spectra and directly on the released native index for native-index spectra.",
            "",
            "## 4. Normalization Rule",
            "",
            "- Apply per-spectrum `vector_l2` normalization after baseline correction.",
            "- This keeps the shared encoder focused on spectral shape instead of absolute magnitude differences across labs and substrates.",
            "",
            "## 5. Smoothing Rule",
            "",
            "- None.",
            "- No smoothing is applied in the frozen Global v2 lane because we do not want to erase narrow Raman/SERS structure without a dataset-specific justification.",
            "",
            "## 6. QC Exclusion Rule",
            "",
            "- Drop spectra with non-finite values.",
            "- Drop spectra with insufficient axis coverage for the explicit-axis crop window.",
            "- Drop spectra that collapse to zero norm after baseline correction and normalization.",
            "- Preserve dropped-spectrum counts explicitly in per-dataset QC summaries.",
            "",
            "## 7. Native Index / Incompatible Axis Handling",
            "",
            "- `stroke_urine_sers` is processed into a provenance-preserving native-index layer with AsLS baseline correction and vector normalization.",
            "- It is not promoted into the initial shared-axis Global v2 encoder training pool because that would require fabricating a Raman-shift map.",
        ]
    ) + "\n"


def build_corpus_summary_report(summary_df: pd.DataFrame) -> str:
    lines = [
        "# Global v2 Preprocessed Corpus Summary",
        "",
        markdown_table(summary_df),
        "",
        "## Corpus State",
        "",
        "- Five datasets are fully harmonized onto the shared explicit-axis Global v2 lane.",
        "- `stroke_urine_sers` has a valid processed layer but remains a native-index special case.",
        "- The initial encoder build should use the explicit-axis harmonized pool and keep stroke separate until a defensible axis map is recovered.",
    ]
    return "\n".join(lines) + "\n"


def build_freeze_checkpoint(summary_df: pd.DataFrame) -> str:
    harmonized = summary_df[summary_df["training_ready"] == "yes"]["dataset_id"].tolist()
    caveat = summary_df[summary_df["training_ready"] != "yes"]["dataset_id"].tolist()
    return "\n".join(
        [
            "# Global v2 Preprocessing Freeze Checkpoint",
            "",
            "1. Are all frozen Global v2 datasets now on a consistent enough preprocessing lane to begin encoder training?",
            "No, not as one single lane. Five datasets are on a consistent explicit-axis lane; one dataset (`stroke_urine_sers`) remains native-index only.",
            "",
            "2. Which datasets are fully harmonized?",
            f"{', '.join(harmonized)}",
            "",
            "3. Which datasets still carry caveats?",
            f"{', '.join(caveat)}",
            "",
            "4. Is any dataset too problematic and should be held out from initial Global v2 training?",
            "Yes. `stroke_urine_sers` should be held out from the initial shared-axis Global v2 encoder training run until a defensible Raman-shift mapping is recovered or a separate native-index training strategy is chosen.",
            "",
            "Decision: proceed to initial Global v2 encoder training with the five explicit-axis harmonized datasets now. Keep stroke in reserve as a processed but held-out native-index cohort.",
        ]
    ) + "\n"


def main() -> None:
    storage = ensure_storage_dirs()
    db_path = resolve_storage_path(storage.get("database"))
    if db_path is None:
        raise RuntimeError("Storage config did not resolve the DuckDB path.")

    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    (AUDIT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    (AUDIT_ROOT / "report").mkdir(parents=True, exist_ok=True)
    PREPROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(db_path), read_only=True) as read_con:
        audit_df = build_audit_table(read_con)
        audit_df.to_csv(AUDIT_ROOT / "tables" / "global_v2_preprocessing_audit.csv", index=False)
        audit_md = AUDIT_ROOT / "report" / "global_v2_preprocessing_audit.md"
        write_markdown(audit_md, build_audit_report(audit_df))
        build_pdf_report(audit_md, [], audit_md.with_suffix(".pdf"))
        write_markdown(AUDIT_ROOT / "report" / "global_v2_preprocessing_spec.md", build_spec_report())

        raw_frames = {dataset_id: load_raw_dataset_frame(read_con, dataset_id) for dataset_id in FROZEN_DATASETS}

    summary_rows = []
    with duckdb.connect(str(db_path)) as write_con:
        for dataset_id in FROZEN_DATASETS:
            dataset_df = raw_frames[dataset_id]
            if dataset_id == "stroke_urine_sers":
                processed_df, processed_meta_df, dropped_df, summary = preprocess_native_dataset(dataset_df, dataset_id)
            else:
                processed_df, processed_meta_df, dropped_df, summary = preprocess_explicit_dataset(dataset_df, dataset_id)
            class_summary_df = class_summary_from_processed(processed_meta_df, processed_df)
            insert_processed_layer(
                connection=write_con,
                dataset_id=dataset_id,
                processed_df=processed_df,
                processed_meta_df=processed_meta_df,
                class_summary_df=class_summary_df,
                processing_version=summary["processing_version"],
            )
            write_dataset_outputs(
                dataset_id=dataset_id,
                processed_df=processed_df,
                processed_meta_df=processed_meta_df,
                dropped_df=dropped_df,
                summary=summary,
            )
            summary_rows.append(
                {
                    "dataset_id": dataset_id,
                    "role": "core_training",
                    "final_usable_spectra": summary["output_spectra"],
                    "final_usable_samples": int(processed_meta_df["sample_id"].nunique()) if not processed_meta_df.empty else 0,
                    "final_axis_status": summary["axis_type"],
                    "processing_version": summary["processing_version"],
                    "baseline_method": summary["baseline_method"],
                    "normalization_method": summary["normalization_method"],
                    "crop_range": (
                        f"{summary['final_axis_min']:.0f}-{summary['final_axis_max']:.0f}"
                        if summary["axis_type"] == "explicit_raman_shift"
                        else "native_index_preserved"
                    ),
                    "training_ready": summary["training_ready"],
                    "remaining_caveats": (
                        ""
                        if summary["training_ready"] == "yes"
                        else "native index only; hold out from initial shared-axis training"
                    ),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(AUDIT_ROOT / "tables" / "global_v2_preprocessed_corpus_summary.csv", index=False)
    summary_md = AUDIT_ROOT / "report" / "global_v2_preprocessed_corpus_summary.md"
    write_markdown(summary_md, build_corpus_summary_report(summary_df))
    build_pdf_report(summary_md, [], summary_md.with_suffix(".pdf"))
    write_markdown(AUDIT_ROOT / "report" / "global_v2_preprocessing_freeze_checkpoint.md", build_freeze_checkpoint(summary_df))

    print(f"Wrote preprocessing audit and processed layer under {AUDIT_ROOT} and {PREPROCESSED_ROOT}")


if __name__ == "__main__":
    main()
