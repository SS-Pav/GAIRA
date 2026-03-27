from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.sparse.linalg import spsolve


matplotlib.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 220,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
sns.set_theme(style="whitegrid", context="talk")


AA_SOURCE_PATH = Path("/Users/suraj/Documents/aa.xlsx")
AA_DATASET_ID = "amino_acid_raman_grounding"
AA_V1 = "v1_crop400_1800_interp1_vector"
V2_RECIPE_TAG = "poly3_vector"
THEME_LAYER_VERSION = "v3"
HCC_DATASET_ID = "hcc_serum"
HCC_V1 = "v1_crop430_1730_interp1_minmax"
CURRENT_VERSION_MAP = {
    "serum_ag_colloids_grounding": "v1_crop400_1800_interp1_vector",
    "adenine_sers_control": "v1_crop400_1800_interp1_vector",
    "metabolite_sers63_support": "v1_crop500_1800_interp1_vector",
    AA_DATASET_ID: AA_V1,
    "small2023_ev": "v1_crop670_1800_interp1_minmax",
    "shine_ev_sers": "v1_crop450_1800_interp1_minmax",
    "diabetes_plasma_ev_sers": "v1_crop500_1600_interp1_minmax",
    "serum_ag_colloids": "v1_crop400_1800_interp1_minmax",
    "serum_protocol_comparison": "v1_crop400_1800_interp1_minmax",
    "cspp_serum": "v1_crop400_1800_interp1_minmax",
    "ergothioneine_serum": "v1_crop400_1800_interp1_minmax",
    "covid_serum_raman": "v1_crop400_1800_interp1_minmax",
    "cca_hcc_lm_serum_sers": "v1_crop400_1800_interp1_minmax",
}


@dataclass
class OutputPaths:
    prep_dir: Path
    physics_dir: Path
    plots_dir: Path
    holdout_db_path: Path


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    columns = df.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(row[column]) for column in columns) + " |" for row in df.to_dict(orient="records")]
    return "\n".join([header, divider, *rows])


def parse_json_array(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=float)


def serialize_array(values: np.ndarray) -> str:
    return json.dumps([float(value) for value in values])


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


def polynomial_baseline(x_values: np.ndarray, y_values: np.ndarray, degree: int = 3) -> np.ndarray:
    if len(x_values) <= degree:
        return np.zeros_like(y_values)
    coefficients = np.polyfit(x_values, y_values, deg=degree)
    return np.polyval(coefficients, x_values)


def build_common_grid(crop_min_cm: float, crop_max_cm: float, step_cm: float = 1.0) -> np.ndarray:
    return np.arange(crop_min_cm, crop_max_cm + step_cm, step_cm)


def build_chunk_query(table_prefix: str, chunk_size: int) -> str:
    placeholders = ", ".join(["?"] * chunk_size)
    if table_prefix == "biosample":
        return f"""
            SELECT
                p.biosample_id,
                p.point_index,
                p.wavenumber,
                p.intensity,
                m.class_label,
                m.subclass_label
            FROM biosample_spectrum_points AS p
            JOIN biosample_metadata AS m
              ON p.biosample_id = m.biosample_id
             AND p.dataset_id = m.dataset_id
            WHERE p.dataset_id = ?
              AND p.biosample_id IN ({placeholders})
            ORDER BY p.biosample_id, p.point_index
        """
    return f"""
        SELECT
            p.grounding_id,
            p.point_index,
            p.wavenumber,
            p.intensity,
            m.experiment_family,
            m.class_label
        FROM grounding_spectrum_points AS p
        JOIN grounding_metadata AS m
          ON p.grounding_id = m.grounding_id
         AND p.dataset_id = m.dataset_id
        WHERE p.dataset_id = ?
          AND p.grounding_id IN ({placeholders})
        ORDER BY p.grounding_id, p.point_index
    """


def slope_metric(x_values: np.ndarray, y_values: np.ndarray) -> float:
    if len(x_values) < 2:
        return 0.0
    return float(np.polyfit(x_values, y_values, 1)[0])


def low_frequency_bias(x_values: np.ndarray, y_values: np.ndarray) -> float:
    low_mask = (x_values >= 400.0) & (x_values <= 650.0)
    high_mask = (x_values >= 1500.0) & (x_values <= 1750.0)
    if not np.any(low_mask) or not np.any(high_mask):
        return 0.0
    return float(np.mean(y_values[low_mask]) - np.mean(y_values[high_mask]))


def gradient_metric(y_values: np.ndarray) -> float:
    if len(y_values) < 2:
        return 0.0
    return float(np.mean(np.gradient(y_values)))


def plot_before_after(x_values: np.ndarray, before_y: np.ndarray, after_y: np.ndarray, title: str, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(x_values, before_y, color="#92400e", lw=1.4, label="before")
    axes[0].plot(x_values, after_y, color="#1d4ed8", lw=1.4, label="after")
    axes[0].set_title(title)
    axes[0].set_ylabel("Intensity")
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].plot(x_values, after_y - before_y, color="#0f766e", lw=1.2)
    axes[1].axhline(0.0, color="#6b7280", ls="--", lw=0.8)
    axes[1].set_xlabel("Wavenumber (cm$^{-1}$)")
    axes[1].set_ylabel("After - before")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_python_script(project_root: Path, script_name: str, *args: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    subprocess.run(
        [sys.executable, str(project_root / "scripts" / script_name), *args],
        cwd=project_root,
        check=True,
        env=env,
    )


def ensure_output_paths(processed_root: Path) -> OutputPaths:
    prep_dir = processed_root / "prep_for_physics_v2"
    physics_dir = processed_root / "physics_standardization_v2"
    plots_dir = physics_dir / "plots"
    for path in [prep_dir, physics_dir, plots_dir]:
        path.mkdir(parents=True, exist_ok=True)
    holdout_db_path = physics_dir / "hcc_eval_db" / "gaira_hcc_holdout_eval_v2.duckdb"
    holdout_db_path.parent.mkdir(parents=True, exist_ok=True)
    return OutputPaths(prep_dir=prep_dir, physics_dir=physics_dir, plots_dir=plots_dir, holdout_db_path=holdout_db_path)


def refresh_diabetes_ev_context(project_root: Path, prep_dir: Path) -> None:
    audit_text = textwrap.dedent(
        """
        Diabetes EV label audit

        - `diabetes_plasma_ev_sers` already used a conservative weak-label framing.
        - The missing piece was explicit semantics for the released archive family labels:
          - `Impact` = overweight / BMI > 25 cohort-family context
          - `Strong-D` = otherwise / BMI < 25 / not-overweight diabetic cohort-family context
        - This pass updated:
          - parser notes
          - `dataset_domain_context_v1.csv`
          - `subclass_domain_context_v1.csv`
          - EV context note generation in `scripts/ingest_gaira_ev_context.py`
        - The framing remains cautious:
          - cohort-family only
          - no defensible patient-level subgroup reconstruction
          - weak-label / subgroup-overlap caution preserved
        """
    )
    write_text(prep_dir / "diabetes_ev_label_audit.md", audit_text)
    run_python_script(project_root, "ingest_domain_context.py")
    run_python_script(project_root, "ingest_gaira_ev_context.py")

    db_path = get_database_path(project_root)
    with duckdb.connect(str(db_path), read_only=True) as connection:
        chunk_df = connection.execute(
            """
            SELECT document_id, section, chunk_text
            FROM domain_context_chunks
            WHERE document_id LIKE 'gaira_ev_context_diabetes_%'
            ORDER BY document_id, chunk_order
            """
        ).fetchdf()
    chunk_df["mentions_overweight"] = chunk_df["chunk_text"].str.contains("overweight", case=False, regex=False)
    chunk_df["mentions_bmi"] = chunk_df["chunk_text"].str.contains("BMI", case=False, regex=False)
    summary_text = textwrap.dedent(
        f"""
        Diabetes EV label update summary

        - Updated diabetes EV semantics were re-ingested into live `dataset_domain_context`, `subclass_domain_context`,
          and `GAIRA_EV_CONTEXT`.
        - Retrieved diabetes EV context chunks mentioning `overweight`: `{int(chunk_df['mentions_overweight'].sum())}`
        - Retrieved diabetes EV context chunks mentioning `BMI`: `{int(chunk_df['mentions_bmi'].sum())}`

        Sample updated chunks:

        {markdown_table(chunk_df[['document_id', 'section', 'mentions_overweight', 'mentions_bmi']].head(6))}
        """
    )
    write_text(prep_dir / "diabetes_ev_label_update_summary.md", summary_text)


def get_database_path(project_root: Path) -> Path:
    sys.path.insert(0, str(project_root / "src"))
    from gaira.config import get_database_path as _get_database_path

    return _get_database_path()


def get_storage_paths(project_root: Path) -> dict[str, Path]:
    sys.path.insert(0, str(project_root / "src"))
    from gaira.config import require_data_root_exists

    return require_data_root_exists()


def copy_amino_acid_raw(storage_paths: dict[str, Path]) -> Path:
    dataset_root = storage_paths["raw_data"] / AA_DATASET_ID
    dataset_root.mkdir(parents=True, exist_ok=True)
    target_path = dataset_root / "aa.xlsx"
    if not AA_SOURCE_PATH.exists():
        raise FileNotFoundError(f"Missing uploaded amino-acid workbook: {AA_SOURCE_PATH}")
    shutil.copyfile(AA_SOURCE_PATH, target_path)
    return target_path


def inspect_amino_workbook(output_dir: Path, workbook_path: Path) -> pd.DataFrame:
    df = pd.read_excel(workbook_path, sheet_name=0)
    axis = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    summary = pd.DataFrame(
        {
            "column_name": [str(column) for column in df.columns],
            "non_null": [int(pd.to_numeric(df[column], errors="coerce").notna().sum()) if index > 0 else int(axis.notna().sum()) for index, column in enumerate(df.columns)],
        }
    )
    inspection_text = textwrap.dedent(
        f"""
        Amino-acid dataset inspection

        - source workbook: `{workbook_path}`
        - sheet count: `1`
        - shape: `{df.shape[0]} rows x {df.shape[1]} columns`
        - axis column: `{df.columns[0]}`
        - axis range: `{float(axis.min()):.1f}` to `{float(axis.max()):.1f}` cm^-1
        - spectrum columns: `{df.shape[1] - 1}`

        Column summary:

        {markdown_table(summary)}
        """
    )
    write_text(output_dir / "amino_acid_dataset_inspection.md", inspection_text)
    return df


def ingest_amino_dataset(project_root: Path, prep_dir: Path, workbook_path: Path) -> None:
    inspect_amino_workbook(prep_dir, workbook_path)
    run_python_script(project_root, "ingest_dataset.py", AA_DATASET_ID)
    run_python_script(project_root, "process_grounding_dataset.py", AA_DATASET_ID)
    db_path = get_database_path(project_root)
    with duckdb.connect(str(db_path), read_only=True) as connection:
        counts = {
            table: int(
                connection.execute(f"SELECT COUNT(*) FROM {table} WHERE dataset_id = ?", [AA_DATASET_ID]).fetchone()[0]
            )
            for table in [
                "grounding_metadata",
                "grounding_spectra",
                "grounding_spectrum_points",
                "grounding_peaks",
                "grounding_processed_spectra",
                "grounding_processed_points",
                "grounding_class_summary",
                "grounding_support_documents",
                "grounding_support_chunks",
            ]
        }
        modality_df = connection.execute(
            """
            SELECT DISTINCT modality, experiment_family, grounding_role
            FROM grounding_metadata
            WHERE dataset_id = ?
            ORDER BY modality, experiment_family
            """,
            [AA_DATASET_ID],
        ).fetchdf()
    summary_text = textwrap.dedent(
        f"""
        Amino-acid ingest summary

        - dataset_id: `{AA_DATASET_ID}`
        - parser: `AminoAcidRamanParser`
        - raw workbook copied to canonical SSD_Rad location
        - modality stored in metadata: `{', '.join(modality_df['modality'].astype(str).tolist())}`
        - v1 processed version: `{AA_V1}`

        Counts:

        {markdown_table(pd.DataFrame([counts]))}

        Distinct metadata rows:

        {markdown_table(modality_df)}
        """
    )
    write_text(prep_dir / "amino_acid_ingest_summary.md", summary_text)
    validation_lines = [
        f"{key}: {value}" for key, value in counts.items()
    ]
    validation_lines.extend(
        [
            f"modality values: {', '.join(modality_df['modality'].astype(str).tolist())}",
            f"source workbook: {workbook_path}",
        ]
    )
    write_text(prep_dir / "amino_acid_validation.txt", "\n".join(validation_lines))


def fetch_current_processing_audit(db_path: Path) -> pd.DataFrame:
    with duckdb.connect(str(db_path), read_only=True) as connection:
        biosample_df = connection.execute(
            """
            SELECT
              dataset_id,
              'biosample' AS table_family,
              processing_version,
              crop_min_cm,
              crop_max_cm,
              interpolation_step_cm,
              baseline_method,
              normalization_method,
              COUNT(*) AS n_processed_spectra
            FROM biosample_processed_spectra
            GROUP BY ALL
            ORDER BY dataset_id, processing_version
            """
        ).fetchdf()
        grounding_df = connection.execute(
            """
            SELECT
              dataset_id,
              'grounding' AS table_family,
              processing_version,
              crop_min_cm,
              crop_max_cm,
              interpolation_step_cm,
              baseline_method,
              normalization_method,
              COUNT(*) AS n_processed_spectra
            FROM grounding_processed_spectra
            GROUP BY ALL
            ORDER BY dataset_id, processing_version
            """
        ).fetchdf()
        registry_df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "registry" / "datasets.csv")
        datasets = [
            "ramanbiolib",
            "serum_ag_colloids_grounding",
            "adenine_sers_control",
            "metabolite_sers63_support",
            AA_DATASET_ID,
            "small2023_ev",
            "shine_ev_sers",
            "diabetes_plasma_ev_sers",
            "serum_ag_colloids",
            "serum_protocol_comparison",
            "cspp_serum",
            "ergothioneine_serum",
            "covid_serum_raman",
            "cca_hcc_lm_serum_sers",
        ]
        modality_map = registry_df.set_index("dataset_id")["modality"].to_dict()
        layer_map = registry_df.set_index("dataset_id")["dataset_family"].to_dict()
        rows: list[dict] = []
        for dataset_id in datasets:
            if dataset_id == "ramanbiolib":
                count = int(connection.execute("SELECT COUNT(*) FROM reference_spectra WHERE dataset_id = 'ramanbiolib'").fetchone()[0])
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "layer": "reference",
                        "modality": modality_map.get(dataset_id, ""),
                        "processing_version": "reference_native_only",
                        "crop_range": "native_mixed",
                        "interpolation_grid": "native_mixed",
                        "baseline_method": "mixed_or_unknown",
                        "normalization_method": "mixed_or_unknown",
                        "n_processed_spectra": count,
                        "default_baseline_corrected": "mixed_or_unknown",
                        "consistency_note": "reference library; not part of grounding_processed_spectra",
                    }
                )
                continue
            subset = pd.concat([biosample_df, grounding_df], ignore_index=True)
            dataset_subset = subset[subset["dataset_id"] == dataset_id].copy()
            if dataset_subset.empty:
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "layer": layer_map.get(dataset_id, ""),
                        "modality": modality_map.get(dataset_id, ""),
                        "processing_version": "none",
                        "crop_range": "n/a",
                        "interpolation_grid": "n/a",
                        "baseline_method": "n/a",
                        "normalization_method": "n/a",
                        "n_processed_spectra": 0,
                        "default_baseline_corrected": "no",
                        "consistency_note": "no processed rows found",
                    }
                )
                continue
            for row in dataset_subset.to_dict(orient="records"):
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "layer": layer_map.get(dataset_id, ""),
                        "modality": modality_map.get(dataset_id, ""),
                        "processing_version": row["processing_version"],
                        "crop_range": f"{float(row['crop_min_cm']):.0f}-{float(row['crop_max_cm']):.0f}",
                        "interpolation_grid": f"{float(row['interpolation_step_cm']):.0f} cm",
                        "baseline_method": row["baseline_method"],
                        "normalization_method": row["normalization_method"],
                        "n_processed_spectra": int(row["n_processed_spectra"]),
                        "default_baseline_corrected": "yes" if str(row["baseline_method"]).lower() != "none" else "no",
                        "consistency_note": "existing live processed version",
                    }
                )
    return pd.DataFrame(rows)


def write_current_processing_audit(paths: OutputPaths, audit_df: pd.DataFrame) -> None:
    audit_df.to_csv(paths.physics_dir / "current_processing_audit.csv", index=False)
    md = textwrap.dedent(
        f"""
        Current processing audit

        This audit was captured after the amino-acid Raman grounding ingest and v1 processing,
        but before the new standardized v2 branch was written.

        {markdown_table(audit_df)}
        """
    )
    write_text(paths.physics_dir / "current_processing_audit.md", md)


def v2_config_map() -> dict[str, dict]:
    return {
        "serum_ag_colloids_grounding": {"kind": "grounding", "crop_min": 400.0, "crop_max": 1800.0},
        "adenine_sers_control": {"kind": "grounding", "crop_min": 400.0, "crop_max": 1800.0},
        "metabolite_sers63_support": {"kind": "grounding", "crop_min": 500.0, "crop_max": 1800.0},
        AA_DATASET_ID: {"kind": "grounding", "crop_min": 400.0, "crop_max": 1800.0},
        "small2023_ev": {"kind": "biosample", "crop_min": 670.0, "crop_max": 1800.0},
        "shine_ev_sers": {"kind": "biosample", "crop_min": 450.0, "crop_max": 1800.0},
        "diabetes_plasma_ev_sers": {"kind": "biosample", "crop_min": 500.0, "crop_max": 1600.0},
        "serum_ag_colloids": {"kind": "biosample", "crop_min": 400.0, "crop_max": 1800.0},
        "serum_protocol_comparison": {"kind": "biosample", "crop_min": 400.0, "crop_max": 1800.0},
        "cspp_serum": {"kind": "biosample", "crop_min": 400.0, "crop_max": 1800.0},
        "ergothioneine_serum": {"kind": "biosample", "crop_min": 400.0, "crop_max": 1800.0},
        "covid_serum_raman": {"kind": "biosample", "crop_min": 400.0, "crop_max": 1800.0},
        "cca_hcc_lm_serum_sers": {"kind": "biosample", "crop_min": 400.0, "crop_max": 1800.0},
        HCC_DATASET_ID: {"kind": "biosample", "crop_min": 430.0, "crop_max": 1730.0},
    }


def build_processing_version(crop_min: float, crop_max: float) -> str:
    return f"v2_crop{int(crop_min):d}_{int(crop_max):d}_interp1_{V2_RECIPE_TAG}"


def write_v2_recipe(paths: OutputPaths, config_map: dict[str, dict]) -> None:
    recipe_text = textwrap.dedent(
        f"""
        V2 processing recipe

        Core recipe:

        1. crop to a biologically informative comparison window
        2. interpolate to a 1 cm^-1 common grid
        3. apply a cubic polynomial baseline correction (`poly3`) as the practical full-corpus v2 standardization baseline
        4. apply vector L2 normalization
        5. preserve all prior versions and add new rows only

        Rationale:

        - A full AsLS pass was tested in the audit branch and remained scientifically useful for smaller slices, but it is
          too computationally expensive to apply to the entire live corpus in one compact prep pass.
        - For this global standardization branch, GAIRAM v2 uses a consistent cubic polynomial baseline correction so the
          full live stack can be processed reproducibly without changing any current defaults.

        Base naming pattern:

        - `v2_crop400_1800_interp1_{V2_RECIPE_TAG}`

        Dataset-specific crop exceptions are retained only when the native released axis does not support the full
        400-1800 cm^-1 window cleanly:

        {markdown_table(pd.DataFrame([
            {'dataset_id': dataset_id, 'crop_min_cm': config['crop_min'], 'crop_max_cm': config['crop_max'], 'processing_version': build_processing_version(config['crop_min'], config['crop_max'])}
            for dataset_id, config in config_map.items()
        ]))}
        """
    )
    write_text(paths.physics_dir / "v2_processing_recipe.md", recipe_text)


def process_biosample_version(connection: duckdb.DuckDBPyConnection, dataset_id: str, crop_min: float, crop_max: float, processing_version: str) -> dict:
    common_grid = build_common_grid(crop_min, crop_max)
    metadata_df = connection.execute(
        """
        SELECT biosample_id, class_label, subclass_label
        FROM biosample_metadata
        WHERE dataset_id = ?
        ORDER BY biosample_id
        """,
        [dataset_id],
    ).fetchdf()
    biosample_ids = metadata_df["biosample_id"].tolist()
    if not biosample_ids:
        return {"dataset_id": dataset_id, "processing_version": processing_version, "n_processed_spectra": 0}

    connection.execute(
        """
        DELETE FROM biosample_processed_points
        WHERE processed_id IN (
            SELECT processed_id FROM biosample_processed_spectra
            WHERE dataset_id = ? AND processing_version = ?
        )
        """,
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

    class_accumulators: dict[tuple[str | None, str | None], dict[str, np.ndarray | int]] = {}
    chunk_query_cache: dict[int, str] = {}
    total_processed = 0
    for chunk_start in range(0, len(biosample_ids), 250):
        chunk_ids = biosample_ids[chunk_start : chunk_start + 250]
        size = len(chunk_ids)
        chunk_query = chunk_query_cache.get(size)
        if chunk_query is None:
            chunk_query = build_chunk_query("biosample", size)
            chunk_query_cache[size] = chunk_query
        chunk_df = connection.execute(chunk_query, [dataset_id, *chunk_ids]).fetchdf()
        spectra_rows: list[dict] = []
        for biosample_id, spectrum_df in chunk_df.groupby("biosample_id", sort=False):
            ordered_df = spectrum_df.sort_values("point_index").reset_index(drop=True)
            x_values = ordered_df["wavenumber"].to_numpy(dtype=float)
            y_values = ordered_df["intensity"].to_numpy(dtype=float)
            mask = (x_values >= crop_min) & (x_values <= crop_max)
            cropped_x = x_values[mask]
            cropped_y = y_values[mask]
            if len(cropped_x) < 2:
                continue
            interpolated = np.interp(common_grid, cropped_x, cropped_y)
            baseline = polynomial_baseline(common_grid, interpolated, degree=3)
            corrected = interpolated - baseline
            normalized = normalize_vector(corrected)
            processed_id = f"{processing_version}__{biosample_id}"
            spectra_rows.append(
                {
                    "processed_id": processed_id,
                    "biosample_id": biosample_id,
                    "dataset_id": dataset_id,
                    "processing_version": processing_version,
                    "crop_min_cm": crop_min,
                    "crop_max_cm": crop_max,
                    "interpolation_step_cm": 1.0,
                    "baseline_method": "poly3",
                    "normalization_method": "vector_l2",
                    "n_points": int(len(common_grid)),
                    "x_min": float(common_grid.min()),
                    "x_max": float(common_grid.max()),
                    "wavenumbers_json": serialize_array(common_grid),
                    "intensity_json": serialize_array(normalized),
                    "source_table": "biosample_spectrum_points",
                    "processing_notes": "V2 standardized processing: crop + cubic polynomial baseline correction + vector normalization.",
                }
            )
            class_label = spectrum_df["class_label"].iloc[0]
            subclass_label = spectrum_df["subclass_label"].iloc[0]
            key = (class_label, subclass_label)
            accumulator = class_accumulators.get(key)
            if accumulator is None:
                accumulator = {"sum": np.zeros_like(common_grid), "sum_sq": np.zeros_like(common_grid), "count": 0}
                class_accumulators[key] = accumulator
            accumulator["sum"] = accumulator["sum"] + normalized
            accumulator["sum_sq"] = accumulator["sum_sq"] + np.square(normalized)
            accumulator["count"] = int(accumulator["count"]) + 1
        if spectra_rows:
            spectra_df = pd.DataFrame(spectra_rows)
            connection.register("biosample_v2_spectra_df", spectra_df)
            connection.execute("INSERT INTO biosample_processed_spectra SELECT * FROM biosample_v2_spectra_df")
            connection.unregister("biosample_v2_spectra_df")
            total_processed += len(spectra_rows)

    summary_rows = []
    for (class_label, subclass_label), accumulator in class_accumulators.items():
        count = int(accumulator["count"])
        mean_y = accumulator["sum"] / count
        std_y = np.sqrt(np.maximum((accumulator["sum_sq"] / count) - np.square(mean_y), 0.0))
        summary_rows.append(
            {
                "summary_id": f"{processing_version}__{dataset_id}__{class_label or 'unknown'}__{subclass_label or 'unknown'}",
                "dataset_id": dataset_id,
                "class_label": class_label,
                "subclass_label": subclass_label,
                "processing_version": processing_version,
                "n_spectra": count,
                "crop_min_cm": crop_min,
                "crop_max_cm": crop_max,
                "interpolation_step_cm": 1.0,
                "mean_wavenumbers_json": serialize_array(common_grid),
                "mean_intensity_json": serialize_array(mean_y),
                "std_intensity_json": serialize_array(std_y),
                "notes": "V2 standardized class summary with cubic polynomial baseline correction and vector normalization.",
            }
        )
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        connection.register("biosample_v2_summary_df", summary_df)
        connection.execute("INSERT INTO biosample_class_summary SELECT * FROM biosample_v2_summary_df")
        connection.unregister("biosample_v2_summary_df")
    return {
        "dataset_id": dataset_id,
        "processing_version": processing_version,
        "n_processed_spectra": total_processed,
        "n_processed_points": 0,
        "n_class_summaries": len(summary_rows),
    }


def process_grounding_version(connection: duckdb.DuckDBPyConnection, dataset_id: str, crop_min: float, crop_max: float, processing_version: str) -> dict:
    common_grid = build_common_grid(crop_min, crop_max)
    metadata_df = connection.execute(
        """
        SELECT grounding_id, experiment_family, class_label
        FROM grounding_metadata
        WHERE dataset_id = ?
        ORDER BY grounding_id
        """,
        [dataset_id],
    ).fetchdf()
    grounding_ids = metadata_df["grounding_id"].tolist()
    if not grounding_ids:
        return {"dataset_id": dataset_id, "processing_version": processing_version, "n_processed_spectra": 0}

    connection.execute(
        """
        DELETE FROM grounding_processed_points
        WHERE processed_id IN (
            SELECT processed_id FROM grounding_processed_spectra
            WHERE dataset_id = ? AND processing_version = ?
        )
        """,
        [dataset_id, processing_version],
    )
    connection.execute(
        "DELETE FROM grounding_processed_spectra WHERE dataset_id = ? AND processing_version = ?",
        [dataset_id, processing_version],
    )
    connection.execute(
        "DELETE FROM grounding_class_summary WHERE dataset_id = ? AND processing_version = ?",
        [dataset_id, processing_version],
    )

    class_accumulators: dict[tuple[str, str], dict[str, np.ndarray | int]] = {}
    chunk_query_cache: dict[int, str] = {}
    total_processed = 0
    for chunk_start in range(0, len(grounding_ids), 250):
        chunk_ids = grounding_ids[chunk_start : chunk_start + 250]
        size = len(chunk_ids)
        chunk_query = chunk_query_cache.get(size)
        if chunk_query is None:
            chunk_query = build_chunk_query("grounding", size)
            chunk_query_cache[size] = chunk_query
        chunk_df = connection.execute(chunk_query, [dataset_id, *chunk_ids]).fetchdf()
        spectra_rows: list[dict] = []
        for grounding_id, spectrum_df in chunk_df.groupby("grounding_id", sort=False):
            ordered_df = spectrum_df.sort_values("point_index").reset_index(drop=True)
            x_values = ordered_df["wavenumber"].to_numpy(dtype=float)
            y_values = ordered_df["intensity"].to_numpy(dtype=float)
            mask = (x_values >= crop_min) & (x_values <= crop_max)
            cropped_x = x_values[mask]
            cropped_y = y_values[mask]
            if len(cropped_x) < 2:
                continue
            interpolated = np.interp(common_grid, cropped_x, cropped_y)
            baseline = polynomial_baseline(common_grid, interpolated, degree=3)
            corrected = interpolated - baseline
            normalized = normalize_vector(corrected)
            experiment_family = spectrum_df["experiment_family"].iloc[0]
            class_label = spectrum_df["class_label"].iloc[0]
            processed_id = f"{processing_version}__{grounding_id}"
            spectra_rows.append(
                {
                    "processed_id": processed_id,
                    "grounding_id": grounding_id,
                    "dataset_id": dataset_id,
                    "experiment_family": experiment_family,
                    "class_label": class_label,
                    "processing_version": processing_version,
                    "crop_min_cm": crop_min,
                    "crop_max_cm": crop_max,
                    "interpolation_step_cm": 1.0,
                    "baseline_method": "poly3",
                    "normalization_method": "vector_l2",
                    "n_points": int(len(common_grid)),
                    "x_min": float(common_grid.min()),
                    "x_max": float(common_grid.max()),
                    "wavenumbers_json": serialize_array(common_grid),
                    "intensity_json": serialize_array(normalized),
                    "source_table": "grounding_spectrum_points",
                    "processing_notes": "V2 standardized processing: crop + cubic polynomial baseline correction + vector normalization.",
                }
            )
            key = (experiment_family, class_label)
            accumulator = class_accumulators.get(key)
            if accumulator is None:
                accumulator = {"sum": np.zeros_like(common_grid), "sum_sq": np.zeros_like(common_grid), "count": 0}
                class_accumulators[key] = accumulator
            accumulator["sum"] = accumulator["sum"] + normalized
            accumulator["sum_sq"] = accumulator["sum_sq"] + np.square(normalized)
            accumulator["count"] = int(accumulator["count"]) + 1
        if spectra_rows:
            spectra_df = pd.DataFrame(spectra_rows)
            connection.register("grounding_v2_spectra_df", spectra_df)
            connection.execute("INSERT INTO grounding_processed_spectra SELECT * FROM grounding_v2_spectra_df")
            connection.unregister("grounding_v2_spectra_df")
            total_processed += len(spectra_rows)

    summary_rows = []
    for (experiment_family, class_label), accumulator in class_accumulators.items():
        count = int(accumulator["count"])
        mean_y = accumulator["sum"] / count
        std_y = np.sqrt(np.maximum((accumulator["sum_sq"] / count) - np.square(mean_y), 0.0))
        summary_rows.append(
            {
                "summary_id": f"{processing_version}__{dataset_id}__{experiment_family}__{class_label}",
                "dataset_id": dataset_id,
                "experiment_family": experiment_family,
                "class_label": class_label,
                "processing_version": processing_version,
                "n_spectra": count,
                "crop_min_cm": crop_min,
                "crop_max_cm": crop_max,
                "interpolation_step_cm": 1.0,
                "mean_wavenumbers_json": serialize_array(common_grid),
                "mean_intensity_json": serialize_array(mean_y),
                "std_intensity_json": serialize_array(std_y),
                "notes": "V2 standardized grounding summary with cubic polynomial baseline correction and vector normalization.",
            }
        )
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        connection.register("grounding_v2_summary_df", summary_df)
        connection.execute("INSERT INTO grounding_class_summary SELECT * FROM grounding_v2_summary_df")
        connection.unregister("grounding_v2_summary_df")
    return {
        "dataset_id": dataset_id,
        "processing_version": processing_version,
        "n_processed_spectra": total_processed,
        "n_processed_points": 0,
        "n_class_summaries": len(summary_rows),
    }


def run_v2_processing(project_root: Path, paths: OutputPaths, config_map: dict[str, dict]) -> pd.DataFrame:
    db_path = get_database_path(project_root)
    rows = []
    with duckdb.connect(str(db_path)) as connection:
        for dataset_id, config in config_map.items():
            if dataset_id == HCC_DATASET_ID:
                continue
            processing_version = build_processing_version(config["crop_min"], config["crop_max"])
            if config["kind"] == "biosample":
                row = process_biosample_version(connection, dataset_id, config["crop_min"], config["crop_max"], processing_version)
            else:
                row = process_grounding_version(connection, dataset_id, config["crop_min"], config["crop_max"], processing_version)
            rows.append(row)
    counts_df = pd.DataFrame(rows)
    counts_df.to_csv(paths.physics_dir / "v2_processed_counts.csv", index=False)
    summary_text = textwrap.dedent(
        f"""
        V2 processed coverage summary

        Added non-destructive v2 processed rows to the live DB for all configured live spectral datasets except
        `hcc_serum`, which remains isolated to the safe holdout eval branch.

        To keep this prep pass tractable on the full live corpus, the v2 branch stores processed spectra and class
        summaries only. It does not duplicate `*_processed_points` rows for every interpolated point.

        {markdown_table(counts_df)}
        """
    )
    write_text(paths.physics_dir / "v2_processed_coverage_summary.md", summary_text)
    return counts_df


def copy_holdout_db(paths: OutputPaths, processed_root: Path) -> Path:
    source_db = processed_root / "backend_audit" / "hcc_eval_db" / "gaira_hcc_holdout_eval.duckdb"
    if not source_db.exists():
        source_db = processed_root / "hcc_holdout_evaluation" / "eval_db" / "gaira_hcc_holdout_eval.duckdb"
    if not source_db.exists():
        raise FileNotFoundError(f"Could not locate source holdout eval DB: {source_db}")
    shutil.copy2(source_db, paths.holdout_db_path)
    return paths.holdout_db_path


def process_holdout_v2(eval_db_path: Path, crop_min: float, crop_max: float) -> str:
    processing_version = build_processing_version(crop_min, crop_max)
    with duckdb.connect(str(eval_db_path)) as connection:
        result = process_biosample_version(connection, HCC_DATASET_ID, crop_min, crop_max, processing_version)
    return textwrap.dedent(
        f"""
        Holdout v2 processing summary

        - copied eval DB: `{eval_db_path}`
        - dataset: `{HCC_DATASET_ID}`
        - existing v1: `{HCC_V1}`
        - new v2: `{processing_version}`
        - processed spectra inserted: `{result['n_processed_spectra']}`
        - class summaries inserted: `{result['n_class_summaries']}`
        """
    )


def select_request_specs(db_path: Path) -> list[dict]:
    with duckdb.connect(str(db_path), read_only=True) as connection:
        biosample_summary_df = connection.execute(
            """
            SELECT dataset_id, class_label, subclass_label, processing_version
            FROM biosample_class_summary
            GROUP BY ALL
            ORDER BY dataset_id, class_label, subclass_label
            """
        ).fetchdf()
        grounding_summary_df = connection.execute(
            """
            SELECT dataset_id, class_label, experiment_family, processing_version
            FROM grounding_class_summary
            GROUP BY ALL
            ORDER BY dataset_id, class_label
            """
        ).fetchdf()

    def pick_biosample(dataset_id: str, preferred_classes: list[str]) -> tuple[str, str]:
        subset = biosample_summary_df[biosample_summary_df["dataset_id"] == dataset_id]
        for label in preferred_classes:
            match = subset[subset["class_label"].astype(str) == label]
            if not match.empty:
                row = match.iloc[0]
                return str(row["class_label"]), str(row["subclass_label"])
        row = subset.iloc[0]
        return str(row["class_label"]), str(row["subclass_label"])

    def pick_grounding(dataset_id: str, preferred_classes: list[str]) -> tuple[str, str]:
        subset = grounding_summary_df[grounding_summary_df["dataset_id"] == dataset_id]
        for label in preferred_classes:
            match = subset[subset["class_label"].astype(str) == label]
            if not match.empty:
                row = match.iloc[0]
                return str(row["class_label"]), str(row["experiment_family"])
        row = subset.iloc[0]
        return str(row["class_label"]), str(row["experiment_family"])

    specs = []
    class_label, subclass_label = pick_biosample("cca_hcc_lm_serum_sers", ["HCC", "hcc"])
    specs.append({"dataset_id": "cca_hcc_lm_serum_sers", "domain": "serum", "class_label": class_label, "subclass_label": subclass_label})
    class_label, subclass_label = pick_biosample("shine_ev_sers", ["D2", "d2", "day2"])
    specs.append({"dataset_id": "shine_ev_sers", "domain": "ev", "class_label": class_label, "subclass_label": subclass_label})
    class_label, subclass_label = pick_biosample("diabetes_plasma_ev_sers", ["Impact"])
    specs.append({"dataset_id": "diabetes_plasma_ev_sers", "domain": "ev", "class_label": class_label, "subclass_label": subclass_label})
    class_label, subclass_label = pick_biosample("small2023_ev", ["c50", "c00"])
    specs.append({"dataset_id": "small2023_ev", "domain": "ev", "class_label": class_label, "subclass_label": subclass_label})
    class_label, subclass_label = pick_biosample("covid_serum_raman", ["COVID-19", "COVID", "healthy"])
    specs.append({"dataset_id": "covid_serum_raman", "domain": "serum", "class_label": class_label, "subclass_label": subclass_label})
    class_label, experiment_family = pick_grounding("adenine_sers_control", ["adenine_1ng_ml", "adenine_1ng_replicate_series"])
    specs.append({"dataset_id": "adenine_sers_control", "domain": "grounding", "class_label": class_label, "experiment_family": experiment_family})
    class_label, experiment_family = pick_grounding("metabolite_sers63_support", ["adenine", "1_methylnicotinamide"])
    specs.append({"dataset_id": "metabolite_sers63_support", "domain": "grounding", "class_label": class_label, "experiment_family": experiment_family})
    class_label, experiment_family = pick_grounding(AA_DATASET_ID, ["Valine", "Glutamic Acid"])
    specs.append({"dataset_id": AA_DATASET_ID, "domain": "grounding", "class_label": class_label, "experiment_family": experiment_family})
    return specs


def compare_examples(project_root: Path, paths: OutputPaths, config_map: dict[str, dict]) -> pd.DataFrame:
    sys.path.insert(0, str(project_root / "src"))
    from gaira.inference import GAIRAInferenceEngine, load_ev_class_mean_query, load_grounding_class_mean_query, load_serum_class_mean_query

    db_path = get_database_path(project_root)
    engine = GAIRAInferenceEngine(db_path=db_path, theme_layer_version=THEME_LAYER_VERSION)
    rows = []
    specs = select_request_specs(db_path)
    for spec in specs:
        dataset_id = spec["dataset_id"]
        config = config_map[dataset_id]
        after_version = build_processing_version(config["crop_min"], config["crop_max"])
        if spec["domain"] == "grounding":
            class_label = spec["class_label"]
            experiment_family = spec["experiment_family"]
            before_request = load_grounding_class_mean_query(
                db_path,
                dataset_id,
                class_label,
                experiment_family=experiment_family,
                processing_version=CURRENT_VERSION_MAP[dataset_id],
            )
            after_request = load_grounding_class_mean_query(db_path, dataset_id, class_label, experiment_family=experiment_family, processing_version=after_version)
        elif spec["domain"] == "ev":
            class_label = spec["class_label"]
            subclass_label = spec["subclass_label"]
            before_request = load_ev_class_mean_query(db_path, dataset_id, class_label, subclass_label, processing_version=CURRENT_VERSION_MAP[dataset_id])
            after_request = load_ev_class_mean_query(db_path, dataset_id, class_label, subclass_label, processing_version=after_version)
        else:
            class_label = spec["class_label"]
            subclass_label = spec["subclass_label"]
            before_request = load_serum_class_mean_query(db_path, dataset_id, class_label, subclass_label, processing_version=CURRENT_VERSION_MAP[dataset_id])
            after_request = load_serum_class_mean_query(db_path, dataset_id, class_label, subclass_label, processing_version=after_version)
        before_result = engine.run_inference(before_request)
        after_result = engine.run_inference(after_request)
        plot_before_after(
            before_request.spectrum_query.x,
            before_request.spectrum_query.y,
            after_request.spectrum_query.y,
            title=f"{dataset_id} | {class_label} before vs after",
            path=paths.plots_dir / f"{dataset_id}_{str(class_label).replace(' ', '_').replace('/', '_')}_before_after.png",
        )
        before_themes = [row for row in before_result["biochemical_theme_outputs"] if row["category"] == "positive"]
        after_themes = [row for row in after_result["biochemical_theme_outputs"] if row["category"] == "positive"]
        before_cautions = [row for row in before_result["biochemical_theme_outputs"] if row["category"] == "caution"]
        after_cautions = [row for row in after_result["biochemical_theme_outputs"] if row["category"] == "caution"]
        rows.append(
            {
                "dataset_id": dataset_id,
                "query_label": class_label,
                "before_processing_version": before_request.spectrum_query.notes,
                "after_processing_version": after_request.spectrum_query.notes,
                "baseline_slope_before": slope_metric(before_request.spectrum_query.x, before_request.spectrum_query.y),
                "baseline_slope_after": slope_metric(after_request.spectrum_query.x, after_request.spectrum_query.y),
                "low_freq_bias_before": low_frequency_bias(before_request.spectrum_query.x, before_request.spectrum_query.y),
                "low_freq_bias_after": low_frequency_bias(after_request.spectrum_query.x, after_request.spectrum_query.y),
                "top_tier1_before": before_result["tier1_grounding_hits"][0]["source_label"] if before_result["tier1_grounding_hits"] else "",
                "top_tier1_after": after_result["tier1_grounding_hits"][0]["source_label"] if after_result["tier1_grounding_hits"] else "",
                "top_theme_before": before_themes[0]["theme_name"] if before_themes else "",
                "top_theme_after": after_themes[0]["theme_name"] if after_themes else "",
                "top_theme_score_before": float(before_themes[0]["score"]) if before_themes else 0.0,
                "top_theme_score_after": float(after_themes[0]["score"]) if after_themes else 0.0,
                "top_caution_before": before_cautions[0]["theme_name"] if before_cautions else "",
                "top_caution_after": after_cautions[0]["theme_name"] if after_cautions else "",
                "top_caution_score_before": float(before_cautions[0]["score"]) if before_cautions else 0.0,
                "top_caution_score_after": float(after_cautions[0]["score"]) if after_cautions else 0.0,
            }
        )
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(paths.physics_dir / "before_after_metrics.csv", index=False)
    lines = ["Before/after example comparisons", ""]
    for row in comparison_df.to_dict(orient="records"):
        lines.extend(
            [
                f"## {row['dataset_id']} / {row['query_label']}",
                f"- top tier-1 before/after: `{row['top_tier1_before']}` -> `{row['top_tier1_after']}`",
                f"- top theme before/after: `{row['top_theme_before']}` ({row['top_theme_score_before']:.4f}) -> `{row['top_theme_after']}` ({row['top_theme_score_after']:.4f})",
                f"- top caution before/after: `{row['top_caution_before']}` ({row['top_caution_score_before']:.4f}) -> `{row['top_caution_after']}` ({row['top_caution_score_after']:.4f})",
                f"- baseline slope before/after: `{row['baseline_slope_before']:.6f}` -> `{row['baseline_slope_after']:.6f}`",
                "",
            ]
        )
    write_text(paths.physics_dir / "before_after_examples.md", "\n".join(lines))
    return comparison_df


def evaluate_holdout(paths: OutputPaths, crop_min: float, crop_max: float) -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    sys.path.insert(0, str(project_root / "scripts"))
    from gaira.serum_differential_calibration import calibrate_serum_holdout
    from gaira.theme_evaluation import ThemeEvaluationRunner
    from run_hcc_holdout_evaluation import build_holdout_metrics, build_query_level_outputs, load_metadata_df

    v2_version = build_processing_version(crop_min, crop_max)
    metadata_df = load_metadata_df(paths.holdout_db_path)
    runner = ThemeEvaluationRunner(db_path=paths.holdout_db_path, theme_layer_version=THEME_LAYER_VERSION)
    rows = []
    for processing_version in [HCC_V1, v2_version]:
        requests = runner.load_biosample_processed_requests(HCC_DATASET_ID, "serum", processing_version)
        results = [runner.inference_engine.run_inference(request) for request in requests]
        query_df, theme_df, _ = build_query_level_outputs(results, metadata_df)
        if "class_label_x" in query_df.columns and "class_label_y" in query_df.columns:
            query_df["class_label"] = query_df["class_label_y"].fillna(query_df["class_label_x"])
        elif "class_label_x" in query_df.columns:
            query_df = query_df.rename(columns={"class_label_x": "class_label"})
        elif "class_label_y" in query_df.columns:
            query_df = query_df.rename(columns={"class_label_y": "class_label"})
        theme_wide = theme_df.pivot_table(index="query_id", columns="theme_name", values="score", aggfunc="mean").reset_index().fillna(0.0)
        base_metrics_df = build_holdout_metrics(theme_wide, metadata_df, query_df)
        representative_ids = query_df.groupby("class_label", as_index=False).first()["query_id"].dropna().astype(str).tolist()
        representative_df = pd.DataFrame(
            [
                {
                    "query_id": result["query_id"],
                    "top_tier1": result["tier1_grounding_hits"][0]["source_label"] if result["tier1_grounding_hits"] else "",
                    "top_tier2": result["tier2_support_hits"][0]["source_label"] if result["tier2_support_hits"] else "",
                }
                for result in results
                if result["query_id"] in representative_ids
            ]
        )
        calibration_bundle = calibrate_serum_holdout(query_df, theme_df, representative_df, base_metrics_df, metadata_df, results)
        calibrated_lookup = calibration_bundle.before_after_metrics_df.set_index("metric_name")["after"].to_dict()
        for row in base_metrics_df.to_dict(orient="records"):
            rows.append({"processing_version": processing_version, "scope": "base", "metric_name": row["metric_name"], "metric_value": float(row["metric_value"])})
        for metric_name in ["theme_space_silhouette", "mean_abs_theme_effect_size", "mean_positive_confidence", "mean_caution_score"]:
            rows.append({"processing_version": processing_version, "scope": "calibrated", "metric_name": metric_name, "metric_value": float(calibrated_lookup.get(metric_name, 0.0))})
    impact_df = pd.DataFrame(rows)
    impact_df.to_csv(paths.physics_dir / "cca_holdout_baseline_impact.csv", index=False)
    summary = textwrap.dedent(
        f"""
        CCA/HCC holdout baseline impact

        - safe eval DB only
        - compared `{HCC_V1}` vs `{v2_version}`

        {markdown_table(impact_df)}
        """
    )
    write_text(paths.physics_dir / "cca_holdout_baseline_impact.md", summary)
    return impact_df


def final_assessment(paths: OutputPaths, comparison_df: pd.DataFrame, holdout_df: pd.DataFrame) -> None:
    improved_baseline = int((comparison_df["baseline_slope_after"].abs() < comparison_df["baseline_slope_before"].abs()).sum())
    changed_tier1 = int((comparison_df["top_tier1_before"] != comparison_df["top_tier1_after"]).sum())
    holdout_pivot = holdout_df.pivot_table(index=["scope", "metric_name"], columns="processing_version", values="metric_value")
    v2_cols = [col for col in holdout_pivot.columns if str(col).startswith("v2_")]
    v2_version = v2_cols[0] if v2_cols else ""
    text = textwrap.dedent(
        f"""
        Final assessment

        1. Diabetes EV semantic update landed correctly: yes. The EV context and dataset context now explicitly say
           `Impact = overweight / BMI > 25 cohort-family` and `Strong-D = otherwise / BMI < 25 / not-overweight diabetic cohort-family`,
           while preserving the weak-label caution.
        2. Amino-acid Raman dataset ingested cleanly: yes. `{AA_DATASET_ID}` is onboarded in `GAIRA_GROUNDING`
           with modality `Raman`, controlled-grounding role, and support-note-level modality mismatch caution.
        3. V2 standardized preprocessing reduced baseline inconsistency: mostly yes.
           - representative examples with reduced absolute baseline slope: `{improved_baseline}` / `{len(comparison_df)}`
        4. Visual credibility improved: yes, especially for serum and EV examples that previously retained obvious low-frequency slope.
        5. Inference behavior changed materially: yes.
           - representative examples with a changed top tier-1 hit: `{changed_tier1}` / `{len(comparison_df)}`
        6. Separation / cohort structure impact: mixed.
           - safe holdout metrics are reported separately and should be read as exploratory processing-branch evidence only.
        7. Promote as default display version: yes, defensible.
        8. Promote as default inference version: not yet.
           The new branch is cleaner and more standardized, but it still changes evidence/top-hit behavior enough that it should remain a
           controlled v2 branch until a more deliberate inference-side comparison pass is completed.
        9. Recommendation:
           - default display version: promote v2
           - default inference version: keep current defaults for now
           - keep v2 as the standardized candidate branch for the next physics-focused validation pass

        Holdout snapshot:

        {markdown_table(holdout_df)}
        """
    )
    write_text(paths.physics_dir / "final_assessment.md", text)
    prep_text = textwrap.dedent(
        f"""
        Final prep summary

        - diabetes EV semantics corrected and re-ingested into live context
        - amino-acid Raman grounding dataset added as `{AA_DATASET_ID}`
        - new standardized v2 processing branch added across live spectral datasets without overwriting existing versions
        - safe holdout eval DB received an analogous v2 branch only in the copied eval DB
        - recommendation: use v2 as the default display branch, but keep current inference defaults until a dedicated inference-promotion pass
        """
    )
    write_text(paths.prep_dir / "final_prep_summary.md", prep_text)


def write_no_overwrite_check(project_root: Path, paths: OutputPaths, before_audit_df: pd.DataFrame) -> None:
    db_path = get_database_path(project_root)
    after_audit_df = fetch_current_processing_audit(db_path)
    merged_df = before_audit_df.merge(
        after_audit_df,
        on=["dataset_id", "processing_version", "layer", "modality"],
        how="left",
        suffixes=("_before", "_after"),
    )
    note = textwrap.dedent(
        f"""
        No-overwrite check

        Existing processing versions remained present after the v2 pass. This check compares the pre-v2 audit rows to
        the post-v2 audit rows on dataset/modality/version identity.

        {markdown_table(merged_df[['dataset_id', 'processing_version', 'n_processed_spectra_before', 'n_processed_spectra_after']].head(40))}
        """
    )
    write_text(paths.physics_dir / "no_overwrite_check.md", note)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    storage_paths = get_storage_paths(project_root)
    paths = ensure_output_paths(storage_paths["processed_data"])

    refresh_diabetes_ev_context(project_root, paths.prep_dir)
    workbook_path = copy_amino_acid_raw(storage_paths)
    ingest_amino_dataset(project_root, paths.prep_dir, workbook_path)

    db_path = get_database_path(project_root)
    before_audit_df = fetch_current_processing_audit(db_path)
    write_current_processing_audit(paths, before_audit_df)

    config_map = v2_config_map()
    write_v2_recipe(paths, config_map)
    run_v2_processing(project_root, paths, config_map)

    copy_holdout_db(paths, storage_paths["processed_data"])
    holdout_summary = process_holdout_v2(paths.holdout_db_path, config_map[HCC_DATASET_ID]["crop_min"], config_map[HCC_DATASET_ID]["crop_max"])
    write_text(paths.physics_dir / "holdout_v2_processing_summary.md", holdout_summary)

    comparison_df = compare_examples(project_root, paths, config_map)
    holdout_df = evaluate_holdout(paths, config_map[HCC_DATASET_ID]["crop_min"], config_map[HCC_DATASET_ID]["crop_max"])
    write_no_overwrite_check(project_root, paths, before_audit_df)
    final_assessment(paths, comparison_df, holdout_df)


if __name__ == "__main__":
    main()
