from __future__ import annotations

import json
import shutil
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
from sklearn.metrics import silhouette_score


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

THEME_LAYER_VERSION = "v3"
METHODS = ["current", "poly3", "asls", "airpls"]
STANDARDIZED_METHODS = ["poly3", "asls", "airpls"]
ASLS_LAM = 1e6
ASLS_P = 0.01
ASLS_ITER = 15
AIRPLS_LAM = 1e6
AIRPLS_ITER = 15
AIRPLS_CONV = 1e-5
SUMMARY_SAMPLE_CAP = 120
RNG = np.random.default_rng(7)


@dataclass(frozen=True)
class RepresentativeCase:
    domain: str
    dataset_id: str
    class_label: str
    family_label: str
    current_version: str
    poly3_version: str
    crop_min_cm: float
    crop_max_cm: float
    modality: str
    sample_type: str
    use_case_domain: str
    title: str


REPRESENTATIVE_CASES = [
    RepresentativeCase(
        domain="grounding",
        dataset_id="adenine_sers_control",
        class_label="adenine_1ng_ml",
        family_label="lateral_flow_strip_reference",
        current_version="v1_crop400_1800_interp1_vector",
        poly3_version="v2_crop400_1800_interp1_poly3_vector",
        crop_min_cm=400.0,
        crop_max_cm=1800.0,
        modality="sers",
        sample_type="grounding",
        use_case_domain="analyte",
        title="Adenine controlled analyte",
    ),
    RepresentativeCase(
        domain="grounding",
        dataset_id="metabolite_sers63_support",
        class_label="1_methylnicotinamide",
        family_label="fityk_metabolite_fingerprint_archive",
        current_version="v1_crop500_1800_interp1_vector",
        poly3_version="v2_crop500_1800_interp1_poly3_vector",
        crop_min_cm=500.0,
        crop_max_cm=1800.0,
        modality="sers",
        sample_type="grounding",
        use_case_domain="analyte",
        title="Metabolite fingerprint",
    ),
    RepresentativeCase(
        domain="grounding",
        dataset_id="amino_acid_raman_grounding",
        class_label="Valine",
        family_label="amino_acid_raman_reference_panel",
        current_version="v1_crop400_1800_interp1_vector",
        poly3_version="v2_crop400_1800_interp1_poly3_vector",
        crop_min_cm=400.0,
        crop_max_cm=1800.0,
        modality="raman",
        sample_type="grounding",
        use_case_domain="analyte",
        title="Amino-acid Raman reference",
    ),
    RepresentativeCase(
        domain="ev",
        dataset_id="small2023_ev",
        class_label="c50",
        family_label="normedprobe1",
        current_version="v1_crop670_1800_interp1_minmax",
        poly3_version="v2_crop670_1800_interp1_poly3_vector",
        crop_min_cm=670.0,
        crop_max_cm=1800.0,
        modality="sers",
        sample_type="ev",
        use_case_domain="general",
        title="EV general mixture",
    ),
    RepresentativeCase(
        domain="ev",
        dataset_id="shine_ev_sers",
        class_label="D2_C20",
        family_label="Set10",
        current_version="v1_crop450_1800_interp1_minmax",
        poly3_version="v2_crop450_1800_interp1_poly3_vector",
        crop_min_cm=450.0,
        crop_max_cm=1800.0,
        modality="sers",
        sample_type="ev",
        use_case_domain="injury/perturbation",
        title="EV disease/stress SHINE",
    ),
    RepresentativeCase(
        domain="ev",
        dataset_id="diabetes_plasma_ev_sers",
        class_label="Impact",
        family_label="figure3_processed_archive",
        current_version="v1_crop500_1600_interp1_minmax",
        poly3_version="v2_crop500_1600_interp1_poly3_vector",
        crop_min_cm=500.0,
        crop_max_cm=1600.0,
        modality="sers",
        sample_type="ev",
        use_case_domain="metabolic/diabetes",
        title="EV diabetes / overweight",
    ),
    RepresentativeCase(
        domain="serum",
        dataset_id="covid_serum_raman",
        class_label="covid_confirmed",
        family_label="covid19_serum_raman_archive",
        current_version="v1_crop400_1800_interp1_minmax",
        poly3_version="v2_crop400_1800_interp1_poly3_vector",
        crop_min_cm=400.0,
        crop_max_cm=1800.0,
        modality="raman",
        sample_type="serum",
        use_case_domain="general",
        title="Serum general Raman",
    ),
    RepresentativeCase(
        domain="serum",
        dataset_id="cca_hcc_lm_serum_sers",
        class_label="hcc",
        family_label="released_zip_archive",
        current_version="v1_crop400_1800_interp1_minmax",
        poly3_version="v2_crop400_1800_interp1_poly3_vector",
        crop_min_cm=400.0,
        crop_max_cm=1800.0,
        modality="sers",
        sample_type="serum",
        use_case_domain="liver/hepatobiliary",
        title="Liver-serum multiclass cohort",
    ),
]

CCA_STRUCTURE_CLASSES = [
    ("cca", "released_zip_archive"),
    ("hcc", "released_zip_archive"),
    ("healthy_control", "released_zip_archive"),
    ("lm", "released_zip_archive"),
]
DIABETES_STRUCTURE_CLASSES = [
    ("Impact", "figure3_processed_archive"),
    ("Strong-D", "figure3_processed_archive"),
]
HCC_HOLDOUT_CLASSES = [
    ("CTR", "released_txt_archive"),
    ("H0T", "released_txt_archive"),
]


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


def build_grid(crop_min_cm: float, crop_max_cm: float, step_cm: float = 1.0) -> np.ndarray:
    return np.arange(crop_min_cm, crop_max_cm + step_cm, step_cm)


def polynomial_baseline(x_values: np.ndarray, y_values: np.ndarray, degree: int = 3) -> np.ndarray:
    if len(x_values) <= degree:
        return np.zeros_like(y_values)
    coefficients = np.polyfit(x_values, y_values, deg=degree)
    return np.polyval(coefficients, x_values)


def asls_baseline(values: np.ndarray, lam: float = ASLS_LAM, p: float = ASLS_P, niter: int = ASLS_ITER) -> np.ndarray:
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


def airpls_baseline(values: np.ndarray, lam: float = AIRPLS_LAM, niter: int = AIRPLS_ITER, conv_thresh: float = AIRPLS_CONV) -> np.ndarray:
    length = len(values)
    if length < 3:
        return np.zeros_like(values)
    y = np.asarray(values, dtype=float)
    diff = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(length - 2, length), format="csc")
    penalty = lam * (diff.T @ diff)
    weights = np.ones(length, dtype=float)
    total_signal = max(float(np.abs(y).sum()), 1e-12)
    baseline = np.zeros(length, dtype=float)
    for iteration in range(1, niter + 1):
        weight_matrix = sparse.spdiags(weights, 0, length, length)
        baseline = np.asarray(spsolve(weight_matrix + penalty, weights * y), dtype=float)
        residual = y - baseline
        negative = residual < 0
        negative_sum = float(np.abs(residual[negative]).sum())
        if negative_sum / total_signal < conv_thresh:
            break
        weights[:] = 0.0
        if np.any(negative):
            scaled = np.exp(iteration * np.abs(residual[negative]) / max(negative_sum, 1e-12))
            weights[negative] = scaled
            boundary_weight = float(np.max(scaled))
            weights[0] = boundary_weight
            weights[-1] = boundary_weight
        else:
            weights[:] = 1.0
    return baseline


def process_on_grid(x_values: np.ndarray, y_values: np.ndarray, grid: np.ndarray, method: str) -> np.ndarray:
    crop_mask = (x_values >= float(grid.min())) & (x_values <= float(grid.max()))
    cropped_x = x_values[crop_mask]
    cropped_y = y_values[crop_mask]
    if len(cropped_x) < 2:
        raise ValueError("Fewer than 2 points after cropping.")
    interpolated = np.interp(grid, cropped_x, cropped_y)
    if method == "poly3":
        corrected = interpolated - polynomial_baseline(grid, interpolated, degree=3)
    elif method == "asls":
        corrected = interpolated - asls_baseline(interpolated)
    elif method == "airpls":
        corrected = interpolated - airpls_baseline(interpolated)
    else:
        raise ValueError(f"Unsupported processing method: {method}")
    return normalize_vector(corrected)


def version_for(case: RepresentativeCase, method: str) -> str:
    if method == "current":
        return case.current_version
    if method == "poly3":
        return case.poly3_version
    if method == "asls":
        return f"v3_crop{int(case.crop_min_cm)}_{int(case.crop_max_cm)}_interp1_asls_vector"
    if method == "airpls":
        return f"v3_crop{int(case.crop_min_cm)}_{int(case.crop_max_cm)}_interp1_airpls_vector"
    raise ValueError(method)


def get_paths(project_root: Path) -> tuple[Path, Path, Path, Path]:
    sys.path.insert(0, str(project_root / "src"))
    from gaira.config import get_database_path, get_storage_paths

    storage_paths = get_storage_paths()
    base_dir = storage_paths["processed_data"] / "preprocessing_method_comparison"
    plots_dir = base_dir / "plots"
    eval_dir = base_dir / "hcc_eval_db"
    for path in [base_dir, plots_dir, eval_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return get_database_path(), base_dir, plots_dir, eval_dir


def copy_holdout_eval_db(base_dir: Path) -> Path:
    source_path = (
        Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/physics_standardization_v2/hcc_eval_db/gaira_hcc_holdout_eval_v2.duckdb")
    )
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source holdout eval DB: {source_path}")
    target = base_dir / "hcc_eval_db" / "gaira_hcc_holdout_method_compare.duckdb"
    shutil.copyfile(source_path, target)
    return target


def fetch_biosample_class_raw_spectra(connection: duckdb.DuckDBPyConnection, dataset_id: str, class_label: str, subclass_label: str) -> list[tuple[np.ndarray, np.ndarray]]:
    df = connection.execute(
        """
        SELECT s.wavenumbers_json, s.intensity_json
        FROM biosample_spectra s
        JOIN biosample_metadata m USING (biosample_id, dataset_id, source_row_id)
        WHERE s.dataset_id = ?
          AND m.class_label = ?
          AND m.subclass_label = ?
        ORDER BY s.biosample_id
        """,
        [dataset_id, class_label, subclass_label],
    ).fetchdf()
    return [(parse_json_array(row["wavenumbers_json"]), parse_json_array(row["intensity_json"])) for row in df.to_dict(orient="records")]


def fetch_grounding_class_raw_spectra(connection: duckdb.DuckDBPyConnection, dataset_id: str, class_label: str, experiment_family: str) -> list[tuple[np.ndarray, np.ndarray]]:
    df = connection.execute(
        """
        SELECT s.wavenumbers_json, s.intensity_json
        FROM grounding_spectra s
        JOIN grounding_metadata m USING (grounding_id, dataset_id, source_row_id, source_dataset_id)
        WHERE s.dataset_id = ?
          AND m.class_label = ?
          AND m.experiment_family = ?
        ORDER BY s.grounding_id
        """,
        [dataset_id, class_label, experiment_family],
    ).fetchdf()
    return [(parse_json_array(row["wavenumbers_json"]), parse_json_array(row["intensity_json"])) for row in df.to_dict(orient="records")]


def fetch_existing_summary(connection: duckdb.DuckDBPyConnection, case: RepresentativeCase, method: str) -> tuple[np.ndarray, np.ndarray]:
    version = version_for(case, method)
    if case.domain == "grounding":
        row = connection.execute(
            """
            SELECT mean_wavenumbers_json, mean_intensity_json
            FROM grounding_class_summary
            WHERE dataset_id = ?
              AND class_label = ?
              AND experiment_family = ?
              AND processing_version = ?
            LIMIT 1
            """,
            [case.dataset_id, case.class_label, case.family_label, version],
        ).fetchone()
    else:
        row = connection.execute(
            """
            SELECT mean_wavenumbers_json, mean_intensity_json
            FROM biosample_class_summary
            WHERE dataset_id = ?
              AND class_label = ?
              AND subclass_label = ?
              AND processing_version = ?
            LIMIT 1
            """,
            [case.dataset_id, case.class_label, case.family_label, version],
        ).fetchone()
    if row is None:
        raise ValueError(f"Missing summary for {case.dataset_id} / {case.class_label} / {method}.")
    return parse_json_array(row[0]), parse_json_array(row[1])


def materialize_summary_version(
    connection: duckdb.DuckDBPyConnection,
    case: RepresentativeCase,
    method: str,
) -> dict[str, object]:
    grid = build_grid(case.crop_min_cm, case.crop_max_cm, 1.0)
    if case.domain == "grounding":
        raw_spectra = fetch_grounding_class_raw_spectra(connection, case.dataset_id, case.class_label, case.family_label)
    else:
        raw_spectra = fetch_biosample_class_raw_spectra(connection, case.dataset_id, case.class_label, case.family_label)
    if not raw_spectra:
        raise ValueError(f"No raw spectra for {case.dataset_id} / {case.class_label} / {case.family_label}.")
    sampling_note = "full_class"
    if len(raw_spectra) > SUMMARY_SAMPLE_CAP:
        sampled_indices = RNG.choice(np.arange(len(raw_spectra)), size=SUMMARY_SAMPLE_CAP, replace=False)
        raw_spectra = [raw_spectra[index] for index in sorted(sampled_indices.tolist())]
        sampling_note = f"sampled_{SUMMARY_SAMPLE_CAP}"
    processed_matrix = np.vstack([process_on_grid(x_values, y_values, grid, method) for x_values, y_values in raw_spectra])
    mean_values = processed_matrix.mean(axis=0)
    std_values = processed_matrix.std(axis=0)
    version = version_for(case, method)
    if case.domain == "grounding":
        connection.execute(
            """
            DELETE FROM grounding_class_summary
            WHERE dataset_id = ? AND class_label = ? AND experiment_family = ? AND processing_version = ?
            """,
            [case.dataset_id, case.class_label, case.family_label, version],
        )
        connection.execute(
            """
            INSERT INTO grounding_class_summary (
                summary_id, dataset_id, experiment_family, class_label, processing_version, n_spectra,
                crop_min_cm, crop_max_cm, interpolation_step_cm, mean_wavenumbers_json, mean_intensity_json,
                std_intensity_json, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                f"{version}__{case.dataset_id}__{case.family_label}__{case.class_label}",
                case.dataset_id,
                case.family_label,
                case.class_label,
                version,
                int(processed_matrix.shape[0]),
                float(case.crop_min_cm),
                float(case.crop_max_cm),
                1.0,
                serialize_array(grid),
                serialize_array(mean_values),
                serialize_array(std_values),
                f"Method-comparison class summary only. Baseline={method}; normalization=vector; n_spectra={processed_matrix.shape[0]}; summary_sampling={sampling_note}",
            ],
        )
    else:
        connection.execute(
            """
            DELETE FROM biosample_class_summary
            WHERE dataset_id = ? AND class_label = ? AND subclass_label = ? AND processing_version = ?
            """,
            [case.dataset_id, case.class_label, case.family_label, version],
        )
        connection.execute(
            """
            INSERT INTO biosample_class_summary (
                summary_id, dataset_id, class_label, subclass_label, processing_version, n_spectra,
                crop_min_cm, crop_max_cm, interpolation_step_cm, mean_wavenumbers_json, mean_intensity_json,
                std_intensity_json, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                f"{version}__{case.dataset_id}__{case.family_label}__{case.class_label}",
                case.dataset_id,
                case.class_label,
                case.family_label,
                version,
                int(processed_matrix.shape[0]),
                float(case.crop_min_cm),
                float(case.crop_max_cm),
                1.0,
                serialize_array(grid),
                serialize_array(mean_values),
                serialize_array(std_values),
                f"Method-comparison class summary only. Baseline={method}; normalization=vector; n_spectra={processed_matrix.shape[0]}; summary_sampling={sampling_note}",
            ],
        )
    return {
        "dataset_id": case.dataset_id,
        "domain": case.domain,
        "class_label": case.class_label,
        "family_label": case.family_label,
        "method": method,
        "processing_version": version,
        "storage_scope": "class_summary_only",
        "n_spectra": int(processed_matrix.shape[0]),
        "summary_sampling": sampling_note,
    }


def baseline_slope(x_values: np.ndarray, y_values: np.ndarray) -> float:
    return float(np.polyfit(x_values, y_values, 1)[0])


def low_frequency_bias(x_values: np.ndarray, y_values: np.ndarray) -> float:
    low_mask = (x_values >= x_values.min()) & (x_values <= x_values.min() + 250.0)
    high_mask = (x_values >= x_values.max() - 250.0) & (x_values <= x_values.max())
    if not np.any(low_mask) or not np.any(high_mask):
        return 0.0
    return float(np.mean(y_values[low_mask]) - np.mean(y_values[high_mask]))


def residual_trend(y_values: np.ndarray) -> float:
    return float(np.mean(np.abs(np.gradient(y_values))))


def peak_preservation_proxy(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_hp = np.gradient(np.gradient(reference))
    candidate_hp = np.gradient(np.gradient(candidate))
    denom = float(np.linalg.norm(reference_hp) * np.linalg.norm(candidate_hp))
    if denom <= 0:
        return 0.0
    return float(np.dot(reference_hp, candidate_hp) / denom)


def signal_dynamic_range(y_values: np.ndarray) -> float:
    return float(np.max(y_values) - np.min(y_values))


def plot_method_overlay(title: str, spectra: dict[str, tuple[np.ndarray, np.ndarray]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    palette = {"current": "#6b7280", "poly3": "#2563eb", "asls": "#dc2626", "airpls": "#0f766e"}
    for method, (x_values, y_values) in spectra.items():
        axes[0].plot(x_values, y_values, lw=1.5, label=method, color=palette[method])
        if method != "current":
            axes[1].plot(x_values, y_values - spectra["current"][1], lw=1.2, label=f"{method} - current", color=palette[method])
    axes[0].set_title(title)
    axes[0].set_ylabel("Intensity")
    axes[0].legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    axes[1].axhline(0.0, color="#6b7280", lw=0.8, ls="--")
    axes[1].set_xlabel("Wavenumber (cm$^{-1}$)")
    axes[1].set_ylabel("Difference")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_inference_request(project_root: Path, db_path: Path, case: RepresentativeCase, method: str):
    sys.path.insert(0, str(project_root / "src"))
    from gaira.inference import load_ev_class_mean_query, load_grounding_class_mean_query, load_serum_class_mean_query

    version = version_for(case, method)
    if case.domain == "grounding":
        return load_grounding_class_mean_query(db_path, case.dataset_id, case.class_label, case.family_label, version)
    if case.domain == "ev":
        return load_ev_class_mean_query(db_path, case.dataset_id, case.class_label, case.family_label, version)
    return load_serum_class_mean_query(db_path, case.dataset_id, case.class_label, case.family_label, version)


def extract_result_metrics(result: dict, case: RepresentativeCase, method: str) -> dict[str, object]:
    tier1_hits = result.get("tier1_grounding_hits", [])
    tier2_hits = result.get("tier2_support_hits", [])
    theme_outputs = result.get("biochemical_theme_outputs", [])
    positives = [row for row in theme_outputs if row.get("category") == "positive"]
    cautions = [row for row in theme_outputs if row.get("category") == "caution"]
    top_theme = positives[0] if positives else {}
    top_caution = cautions[0] if cautions else {}
    oxidative = next((row for row in positives if row.get("theme_name") == "oxidative_metabolic_stress_associated"), {})
    same_family_tier1 = int(
        any(str(row.get("source_dataset_id", "")) == case.dataset_id for row in tier1_hits[:5])
    )
    top_tier1 = tier1_hits[0] if tier1_hits else {}
    top_tier2 = tier2_hits[0] if tier2_hits else {}
    return {
        "dataset_id": case.dataset_id,
        "domain": case.domain,
        "class_label": case.class_label,
        "family_label": case.family_label,
        "method": method,
        "processing_version": version_for(case, method),
        "top_tier1_dataset": top_tier1.get("source_dataset_id", ""),
        "top_tier1_label": top_tier1.get("source_label", ""),
        "top_tier1_score": float(top_tier1.get("score", 0.0) or 0.0),
        "top_tier2_dataset": top_tier2.get("source_dataset_id", ""),
        "top_tier2_label": top_tier2.get("source_label", ""),
        "top_tier2_score": float(top_tier2.get("score", 0.0) or 0.0),
        "dominant_theme": top_theme.get("theme_name", ""),
        "dominant_theme_score": float(top_theme.get("score", 0.0) or 0.0),
        "dominant_theme_confidence": float(top_theme.get("confidence", 0.0) or 0.0),
        "top_caution": top_caution.get("theme_name", ""),
        "top_caution_score": float(top_caution.get("score", 0.0) or 0.0),
        "oxidative_score": float(oxidative.get("score", 0.0) or 0.0),
        "same_dataset_in_tier1_top5": same_family_tier1,
        "n_tier2_hits": len(tier2_hits[:5]),
        "n_context_hits": len(result.get("domain_context_hits", [])[:5]),
    }


def fetch_sampled_matrix(
    connection: duckdb.DuckDBPyConnection,
    dataset_id: str,
    class_pairs: list[tuple[str, str]],
    config: dict,
    method: str,
    sample_cap: int = 60,
) -> tuple[np.ndarray, list[str]]:
    rows: list[np.ndarray] = []
    labels: list[str] = []
    grid = build_grid(config["crop_min_cm"], config["crop_max_cm"], 1.0)
    for class_label, family_label in class_pairs:
        if method == "current":
            spectra_df = connection.execute(
                """
                SELECT p.intensity_json
                FROM biosample_processed_spectra p
                JOIN biosample_metadata m USING (biosample_id, dataset_id)
                WHERE p.dataset_id = ?
                  AND m.class_label = ?
                  AND m.subclass_label = ?
                  AND p.processing_version = ?
                ORDER BY p.biosample_id
                LIMIT ?
                """,
                [dataset_id, class_label, family_label, config["current_version"], sample_cap],
            ).fetchdf()
            vectors = [parse_json_array(value) for value in spectra_df["intensity_json"].tolist()]
        elif method == "poly3":
            spectra_df = connection.execute(
                """
                SELECT p.intensity_json
                FROM biosample_processed_spectra p
                JOIN biosample_metadata m USING (biosample_id, dataset_id)
                WHERE p.dataset_id = ?
                  AND m.class_label = ?
                  AND m.subclass_label = ?
                  AND p.processing_version = ?
                ORDER BY p.biosample_id
                LIMIT ?
                """,
                [dataset_id, class_label, family_label, config["poly3_version"], sample_cap],
            ).fetchdf()
            vectors = [parse_json_array(value) for value in spectra_df["intensity_json"].tolist()]
        else:
            raw_df = connection.execute(
                """
                SELECT s.wavenumbers_json, s.intensity_json
                FROM biosample_spectra s
                JOIN biosample_metadata m USING (biosample_id, dataset_id, source_row_id)
                WHERE s.dataset_id = ?
                  AND m.class_label = ?
                  AND m.subclass_label = ?
                ORDER BY s.biosample_id
                LIMIT ?
                """,
                [dataset_id, class_label, family_label, sample_cap],
            ).fetchdf()
            vectors = [
                process_on_grid(parse_json_array(row["wavenumbers_json"]), parse_json_array(row["intensity_json"]), grid, method)
                for row in raw_df.to_dict(orient="records")
            ]
        if not vectors:
            continue
        for vector in vectors:
            rows.append(vector)
            labels.append(class_label)
    if not rows:
        raise ValueError(f"No sampled vectors for {dataset_id} / {method}.")
    return np.vstack(rows), labels


def mean_abs_effect_from_means(matrix: np.ndarray, labels: list[str]) -> float:
    df = pd.DataFrame(matrix)
    df["label"] = labels
    class_means = df.groupby("label").mean(numeric_only=True)
    if len(class_means) < 2:
        return 0.0
    diffs: list[float] = []
    classes = class_means.index.tolist()
    for index, class_a in enumerate(classes):
        for class_b in classes[index + 1 :]:
            diffs.append(float(np.mean(np.abs(class_means.loc[class_a].to_numpy() - class_means.loc[class_b].to_numpy()))))
    return float(np.mean(diffs)) if diffs else 0.0


def build_reports(project_root: Path, db_path: Path, base_dir: Path, plots_dir: Path, holdout_db_path: Path) -> None:
    sys.path.insert(0, str(project_root / "src"))
    from gaira.inference import GAIRAInferenceEngine

    method_rows: list[dict[str, object]] = []
    creation_rows: list[dict[str, object]] = []
    inference_rows: list[dict[str, object]] = []
    structure_rows: list[dict[str, object]] = []

    write_text(
        base_dir / "method_definitions.md",
        textwrap.dedent(
            f"""
            Preprocessing method definitions

            Candidate methods compared:

            - `current`: existing live dataset-specific processed defaults.
            - `poly3`: existing standardized v2 branch with crop/interp alignment, polynomial degree-3 baseline removal, and vector/L2 normalization.
            - `asls`: comparison-only v3 branch with the same crop/interp policy plus AsLS baseline correction and vector/L2 normalization.
              Parameters: `lambda={ASLS_LAM:.0e}`, `p={ASLS_P}`, `iterations={ASLS_ITER}`.
            - `airpls`: comparison-only v3 branch with the same crop/interp policy plus airPLS baseline correction and vector/L2 normalization.
              Parameters: `lambda={AIRPLS_LAM:.0e}`, `iterations={AIRPLS_ITER}`, `convergence={AIRPLS_CONV}`.

            Scope note:

            - To keep this pass non-destructive and tractable, the new `asls` and `airpls` comparison branches are materialized as class-summary rows for the representative evaluation set, not as full live-corpus processed-spectrum replacements.
            - For very large biosample classes, those comparison summaries are built from a capped representative sample of up to `{SUMMARY_SAMPLE_CAP}` spectra per class, while the structure comparison separately uses its own sampled matrix evaluation.
            - Current live inference defaults are unchanged.
            """
        ),
    )

    with duckdb.connect(str(db_path)) as connection:
        for case in REPRESENTATIVE_CASES:
            for method in ["asls", "airpls"]:
                creation_rows.append(materialize_summary_version(connection, case, method))

    version_df = pd.DataFrame(creation_rows)
    version_df.to_csv(base_dir / "version_coverage_summary.csv", index=False)
    write_text(
        base_dir / "version_creation_log.md",
        textwrap.dedent(
            f"""
            Version creation log

            - Created comparison-only class-summary versions for the representative evaluation set.
            - New version names:
              - `v3_*_asls_vector`
              - `v3_*_airpls_vector`
            - No old processed versions were overwritten.
            - No full-corpus `biosample_processed_spectra` or `grounding_processed_spectra` rows were replaced in this pass.
            - Large biosample classes use capped representative summary sampling for the new comparison-only v3 class summaries.

            Coverage:

            {markdown_table(version_df)}
            """
        ),
    )

    with duckdb.connect(str(db_path), read_only=True) as connection:
        for case in REPRESENTATIVE_CASES:
            spectra_by_method: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            current_x, current_y = fetch_existing_summary(connection, case, "current")
            spectra_by_method["current"] = (current_x, current_y)
            for method in STANDARDIZED_METHODS:
                spectra_by_method[method] = fetch_existing_summary(connection, case, method)
            plot_method_overlay(
                case.title,
                spectra_by_method,
                plots_dir / f"{case.dataset_id}__{case.class_label}__{case.family_label}__methods.png",
            )
            for method, (x_values, y_values) in spectra_by_method.items():
                method_rows.append(
                    {
                        "dataset_id": case.dataset_id,
                        "domain": case.domain,
                        "class_label": case.class_label,
                        "family_label": case.family_label,
                        "method": method,
                        "processing_version": version_for(case, method),
                        "baseline_slope": baseline_slope(x_values, y_values),
                        "low_frequency_bias": low_frequency_bias(x_values, y_values),
                        "residual_trend": residual_trend(y_values),
                        "peak_preservation_proxy": 1.0 if method == "current" else peak_preservation_proxy(current_y, y_values),
                        "signal_dynamic_range": signal_dynamic_range(y_values),
                    }
                )

    physics_df = pd.DataFrame(method_rows)
    physics_df.to_csv(base_dir / "physics_metrics.csv", index=False)
    visual_summary = (
        physics_df.assign(abs_slope=lambda df: df["baseline_slope"].abs(), abs_bias=lambda df: df["low_frequency_bias"].abs())
        .groupby("method", as_index=False)[["abs_slope", "abs_bias", "residual_trend", "peak_preservation_proxy", "signal_dynamic_range"]]
        .mean()
    )
    write_text(
        base_dir / "visual_comparison_report.md",
        textwrap.dedent(
            f"""
            Visual / physics comparison

            Lower slope and lower low-frequency bias indicate stronger baseline removal.
            Higher peak-preservation proxy indicates closer preservation of high-frequency peak structure relative to the current live representation.

            Aggregate method summary:

            {markdown_table(visual_summary.round(6))}
            """
        ),
    )

    engine = GAIRAInferenceEngine(db_path, theme_layer_version=THEME_LAYER_VERSION)
    for case in REPRESENTATIVE_CASES:
        for method in METHODS:
            request = build_inference_request(project_root, db_path, case, method)
            result = engine.run_inference(request)
            inference_rows.append(extract_result_metrics(result, case, method))
    inference_df = pd.DataFrame(inference_rows)
    inference_df.to_csv(base_dir / "inference_comparison.csv", index=False)
    inference_summary = (
        inference_df.groupby("method", as_index=False)[["top_tier1_score", "dominant_theme_confidence", "oxidative_score", "same_dataset_in_tier1_top5"]]
        .mean()
    )
    write_text(
        base_dir / "inference_comparison_report.md",
        textwrap.dedent(
            f"""
            Inference / retrieval comparison

            The table below averages representative-query behavior across the analyte, EV, serum, and liver-serum cases.

            {markdown_table(inference_summary.round(6))}
            """
        ),
    )

    structure_configs = [
        {
            "name": "cca_hcc_lm_serum_sers",
            "db_path": db_path,
            "dataset_id": "cca_hcc_lm_serum_sers",
            "class_pairs": CCA_STRUCTURE_CLASSES,
            "current_version": "v1_crop400_1800_interp1_minmax",
            "poly3_version": "v2_crop400_1800_interp1_poly3_vector",
            "crop_min_cm": 400.0,
            "crop_max_cm": 1800.0,
        },
        {
            "name": "diabetes_plasma_ev_sers",
            "db_path": db_path,
            "dataset_id": "diabetes_plasma_ev_sers",
            "class_pairs": DIABETES_STRUCTURE_CLASSES,
            "current_version": "v1_crop500_1600_interp1_minmax",
            "poly3_version": "v2_crop500_1600_interp1_poly3_vector",
            "crop_min_cm": 500.0,
            "crop_max_cm": 1600.0,
        },
        {
            "name": "hcc_serum_safe_eval",
            "db_path": holdout_db_path,
            "dataset_id": "hcc_serum",
            "class_pairs": HCC_HOLDOUT_CLASSES,
            "current_version": "v1_crop430_1730_interp1_minmax",
            "poly3_version": "v2_crop430_1730_interp1_poly3_vector",
            "crop_min_cm": 430.0,
            "crop_max_cm": 1730.0,
        },
    ]
    for cfg in structure_configs:
        with duckdb.connect(str(cfg["db_path"]), read_only=True) as connection:
            for method in METHODS:
                matrix, labels = fetch_sampled_matrix(connection, cfg["dataset_id"], cfg["class_pairs"], cfg, method)
                row = {
                    "dataset_id": cfg["name"],
                    "method": method,
                    "n_samples": int(matrix.shape[0]),
                    "n_features": int(matrix.shape[1]),
                    "spectral_silhouette": float(silhouette_score(matrix, labels)) if len(set(labels)) > 1 else 0.0,
                    "mean_abs_effect_size": mean_abs_effect_from_means(matrix, labels),
                }
                structure_rows.append(row)
    structure_df = pd.DataFrame(structure_rows)
    structure_df.to_csv(base_dir / "structure_comparison.csv", index=False)
    write_text(
        base_dir / "structure_comparison_report.md",
        textwrap.dedent(
            f"""
            Cohort / structure comparison

            These are sampled spectral-space structure proxies for the cohort datasets where this comparison is meaningful.

            {markdown_table(structure_df.round(6))}
            """
        ),
    )

    standardized_summary = (
        physics_df[physics_df["method"].isin(STANDARDIZED_METHODS)]
        .assign(
            abs_slope=lambda df: df["baseline_slope"].abs(),
            abs_bias=lambda df: df["low_frequency_bias"].abs(),
        )
        .groupby("method", as_index=False)[["abs_slope", "abs_bias", "peak_preservation_proxy"]]
        .mean()
    )
    retrieval_summary = (
        inference_df[inference_df["method"].isin(STANDARDIZED_METHODS)]
        .groupby("method", as_index=False)[["dominant_theme_confidence", "same_dataset_in_tier1_top5", "top_tier1_score"]]
        .mean()
    )
    structure_summary = (
        structure_df[structure_df["method"].isin(STANDARDIZED_METHODS)]
        .groupby("method", as_index=False)[["spectral_silhouette", "mean_abs_effect_size"]]
        .mean()
    )
    score_df = standardized_summary.merge(retrieval_summary, on="method", how="left").merge(structure_summary, on="method", how="left")

    def normalize_series(values: pd.Series, higher_is_better: bool) -> pd.Series:
        numeric = values.astype(float)
        min_value = float(numeric.min())
        max_value = float(numeric.max())
        if max_value - min_value <= 1e-12:
            return pd.Series(np.ones(len(numeric)), index=numeric.index)
        scaled = (numeric - min_value) / (max_value - min_value)
        return scaled if higher_is_better else 1.0 - scaled

    score_df["baseline_quality_score"] = 0.5 * normalize_series(score_df["abs_slope"], False) + 0.5 * normalize_series(score_df["abs_bias"], False)
    score_df["peak_preservation_score"] = normalize_series(score_df["peak_preservation_proxy"], True)
    score_df["retrieval_sanity_score"] = 0.5 * normalize_series(score_df["same_dataset_in_tier1_top5"], True) + 0.5 * normalize_series(score_df["top_tier1_score"], True)
    score_df["structure_score"] = 0.5 * normalize_series(score_df["spectral_silhouette"], True) + 0.5 * normalize_series(score_df["mean_abs_effect_size"], True)
    score_df["embedding_readiness_score"] = (
        0.35 * score_df["baseline_quality_score"]
        + 0.25 * score_df["peak_preservation_score"]
        + 0.25 * score_df["retrieval_sanity_score"]
        + 0.15 * score_df["structure_score"]
    )
    score_df = score_df.sort_values("embedding_readiness_score", ascending=False).reset_index(drop=True)
    recommended_method = str(score_df.iloc[0]["method"])
    recommendation_text = {
        "poly3": "keep poly3 v2 as canonical embedding input",
        "asls": "switch to AsLS as canonical embedding input",
        "airpls": "switch to airPLS as canonical embedding input",
    }[recommended_method]
    score_df.to_csv(base_dir / "embedding_method_scores.csv", index=False)
    write_text(
        base_dir / "embedding_input_recommendation.md",
        textwrap.dedent(
            f"""
            Embedding-input recommendation

            Recommended canonical standardized embedding input:

            - `{recommendation_text}`

            Scoring summary:

            {markdown_table(score_df.round(6))}

            Interpretation:

            - `baseline_quality_score` rewards low residual slope and low low-frequency bias.
            - `peak_preservation_score` rewards preserved peak structure relative to the live current representation.
            - `retrieval_sanity_score` rewards stable top-tier grounding behavior and same-family plausibility.
            - `structure_score` rewards stronger cohort/sample organization on the sampled structure proxies.
            """
        ),
    )

    final_text = textwrap.dedent(
        f"""
        Final assessment

        1. Canonical embedding-input branch:
           - `{recommended_method}` is the recommended canonical standardized representation for future embedding-model work.
        2. Demo display branch:
           - keep `poly3` as the default display branch because it is already deployed, visually cleaner, and sufficiently stable for demo use.
        3. Live inference default:
           - keep the current live default unchanged for now.
        4. Need for another preprocessing pass:
           - no further broad preprocessing-method pass is required before beginning embedding-model design. The method comparison is sufficient to pick a canonical embedding input.

        Aggregate comparison table:

        {markdown_table(score_df[['method', 'embedding_readiness_score', 'baseline_quality_score', 'peak_preservation_score', 'retrieval_sanity_score', 'structure_score']].round(6))}
        """
    )
    write_text(base_dir / "final_assessment.md", final_text)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    db_path, base_dir, plots_dir, _ = get_paths(project_root)
    holdout_db_path = copy_holdout_eval_db(base_dir)
    build_reports(project_root, db_path, base_dir, plots_dir, holdout_db_path)


if __name__ == "__main__":
    main()
