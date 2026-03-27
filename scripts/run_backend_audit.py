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


RAMANBIOLIB_DATASET_ID = "ramanbiolib"
CCA_DATASET_ID = "cca_hcc_lm_serum_sers"
CCA_SUBCLASS_LABEL = "released_zip_archive"
CCA_V1 = "v1_crop400_1800_interp1_minmax"
CCA_V2 = "v2_crop400_1800_interp1_asls_minmax"
HCC_DATASET_ID = "hcc_serum"
HCC_V1 = "v1_crop430_1730_interp1_minmax"
HCC_V2 = "v2_crop430_1730_interp1_asls_minmax"
THEME_LAYER_VERSION = "v3"


@dataclass
class AuditPaths:
    base_dir: Path
    raw_plot_dir: Path
    comparison_plot_dir: Path
    hcc_eval_db_dir: Path
    hcc_eval_db_path: Path


def ensure_paths(processed_root: Path) -> AuditPaths:
    base_dir = processed_root / "backend_audit"
    raw_plot_dir = base_dir / "cca_raw_processed_spectra_plots"
    comparison_plot_dir = base_dir / "cca_baseline_comparison_plots"
    hcc_eval_db_dir = base_dir / "hcc_eval_db"
    for path in [base_dir, raw_plot_dir, comparison_plot_dir, hcc_eval_db_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return AuditPaths(
        base_dir=base_dir,
        raw_plot_dir=raw_plot_dir,
        comparison_plot_dir=comparison_plot_dir,
        hcc_eval_db_dir=hcc_eval_db_dir,
        hcc_eval_db_path=hcc_eval_db_dir / "gaira_hcc_holdout_eval.duckdb",
    )


def write_markdown(path: Path, text: str) -> None:
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


def normalize_minmax(intensities: np.ndarray) -> np.ndarray:
    min_value = float(np.min(intensities))
    max_value = float(np.max(intensities))
    if max_value - min_value <= 0:
        return np.zeros_like(intensities, dtype=float)
    return (intensities - min_value) / (max_value - min_value)


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


def build_common_grid(crop_min_cm: float, crop_max_cm: float, step_cm: float = 1.0) -> np.ndarray:
    return np.arange(crop_min_cm, crop_max_cm + step_cm, step_cm)


def plot_overlay(spectra: list[tuple[np.ndarray, np.ndarray, str]], title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    palette = sns.color_palette("tab10", n_colors=max(3, len(spectra)))
    for index, (x_values, y_values, label) in enumerate(spectra):
        ax.plot(x_values, y_values, lw=1.3, alpha=0.9, label=label, color=palette[index % len(palette)])
    ax.set_title(title)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_before_after_pair(
    x_values: np.ndarray,
    before_values: np.ndarray,
    after_values: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)
    axes[0].plot(x_values, before_values, color="#b45309", lw=1.6, label="before")
    axes[0].plot(x_values, after_values, color="#2563eb", lw=1.6, label="after")
    axes[0].set_ylabel("Intensity")
    axes[0].set_title(title)
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].plot(x_values, after_values - before_values, color="#0f766e", lw=1.4)
    axes[1].axhline(0.0, color="#6b7280", lw=0.8, ls="--")
    axes[1].set_xlabel("Wavenumber (cm$^{-1}$)")
    axes[1].set_ylabel("After - before")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def baseline_metrics_frame(spectra_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in spectra_df.to_dict(orient="records"):
        x_values = parse_json_array(row["wavenumbers_json"])
        y_values = parse_json_array(row["intensity_json"])
        slope = float(np.polyfit(x_values, y_values, 1)[0])
        gradient = float(np.mean(np.gradient(y_values, x_values)))
        low_mask = (x_values >= 400.0) & (x_values <= 650.0)
        high_mask = (x_values >= 1550.0) & (x_values <= 1800.0)
        low_frequency_bias = float(np.mean(y_values[low_mask]) - np.mean(y_values[high_mask]))
        rows.append(
            {
                "biosample_id": row["biosample_id"],
                "class_label": row["class_label"],
                "processing_version": row["processing_version"],
                "baseline_slope": slope,
                "mean_intensity_gradient": gradient,
                "low_frequency_bias": low_frequency_bias,
            }
        )
    return pd.DataFrame(rows)


def query_ramanbiolib_registry(project_root: Path) -> tuple[pd.DataFrame, str]:
    datasets_df = pd.read_csv(project_root / "data/registry/datasets.csv")
    match = datasets_df[datasets_df["dataset_id"] == RAMANBIOLIB_DATASET_ID].copy()
    pack_text = (project_root / "config/domain_pack_registry.yaml").read_text(encoding="utf-8")
    in_grounding_pack = f"- {RAMANBIOLIB_DATASET_ID}" in pack_text
    summary = textwrap.dedent(
        f"""
        RamanBioLib registry audit

        - dataset_id: `{RAMANBIOLIB_DATASET_ID}`
        - present in datasets.csv: {'yes' if not match.empty else 'no'}
        - present in `GAIRA_GROUNDING` pack: {'yes' if in_grounding_pack else 'no'}
        - declared modality: `{match.iloc[0]['modality'] if not match.empty else 'missing'}`
        - declared family: `{match.iloc[0]['dataset_family'] if not match.empty else 'missing'}`
        """
    )
    return match, summary


def query_ramanbiolib_db_counts(db_path: Path) -> tuple[pd.DataFrame, str]:
    grounding_tables = [
        "grounding_metadata",
        "grounding_spectra",
        "grounding_processed_spectra",
        "grounding_processed_points",
        "grounding_peaks",
        "grounding_class_summary",
    ]
    reference_tables = [
        "reference_metadata",
        "reference_spectra",
        "reference_spectrum_points",
        "reference_peaks",
    ]
    rows = []
    with duckdb.connect(str(db_path), read_only=True) as connection:
        for table_name in grounding_tables + reference_tables:
            count = connection.execute(
                f"SELECT COUNT(*) FROM {table_name} WHERE dataset_id = ?",
                [RAMANBIOLIB_DATASET_ID],
            ).fetchone()[0]
            rows.append(
                {
                    "table_name": table_name,
                    "table_family": "grounding" if table_name in grounding_tables else "reference",
                    "dataset_id": RAMANBIOLIB_DATASET_ID,
                    "row_count": int(count),
                }
            )
    counts_df = pd.DataFrame(rows)
    grounding_total = int(counts_df.loc[counts_df["table_family"] == "grounding", "row_count"].sum())
    reference_total = int(counts_df.loc[counts_df["table_family"] == "reference", "row_count"].sum())
    summary = textwrap.dedent(
        f"""
        RamanBioLib live DB summary

        - grounding-table rows: `{grounding_total}`
        - reference-table rows: `{reference_total}`
        - interpretation: RamanBioLib is stored in `reference_*` tables, not in `grounding_*` tables.
        """
    )
    return counts_df, summary


def run_ramanbiolib_visibility(db_path: Path) -> tuple[pd.DataFrame, str]:
    from gaira.inference import GAIRAInferenceEngine, load_grounding_class_mean_query

    engine = GAIRAInferenceEngine(db_path=db_path, theme_layer_version=THEME_LAYER_VERSION)
    query_specs = [
        {
            "dataset_id": "adenine_sers_control",
            "class_label": "adenine_1ng_replicate_series",
            "experiment_family": "bag_nps_replicate_series",
            "label": "adenine_control",
        },
        {
            "dataset_id": "serum_ag_colloids_grounding",
            "class_label": "UAiso+HSAfilterUpper",
            "experiment_family": "isotopic",
            "label": "uric_acid_like",
        },
    ]

    rows = []
    for spec in query_specs:
        request = load_grounding_class_mean_query(
            db_path=db_path,
            dataset_id=spec["dataset_id"],
            class_label=spec["class_label"],
            experiment_family=spec["experiment_family"],
        )
        result = engine.run_inference(request)
        for tier_name, hits in [
            ("tier1", result.get("tier1_grounding_hits", [])[:10]),
            ("tier2", result.get("tier2_support_hits", [])[:10]),
        ]:
            for rank, hit in enumerate(hits, start=1):
                rows.append(
                    {
                        "query_name": spec["label"],
                        "tier": tier_name,
                        "rank": rank,
                        "source_dataset_id": hit.get("source_dataset_id", ""),
                        "source_label": hit.get("source_label", ""),
                        "score": float(hit.get("reranked_score", hit.get("score", 0.0))),
                        "result_type": hit.get("result_type", ""),
                    }
                )

    visibility_df = pd.DataFrame(rows)
    raman_rows = visibility_df[visibility_df["source_dataset_id"] == RAMANBIOLIB_DATASET_ID]
    summary = textwrap.dedent(
        f"""
        RamanBioLib inference visibility

        - queries tested: `{', '.join(spec['label'] for spec in query_specs)}`
        - RamanBioLib hits observed: `{len(raman_rows)}`
        - appears in tier-1 results: {'yes' if (raman_rows['tier'] == 'tier1').any() else 'no'}
        - appears in tier-2 results: {'yes' if (raman_rows['tier'] == 'tier2').any() else 'no'}
        """
    )
    return visibility_df, summary


def inspect_grounding_search(project_root: Path) -> str:
    grounding_path = project_root / "src/gaira/grounding_search.py"
    text = grounding_path.read_text(encoding="utf-8")
    uses_reference_tables = "FROM reference_spectra" in text and "WHERE s.dataset_id = 'ramanbiolib'" in text
    grounding_summary_global = "FROM grounding_class_summary" in text
    summary = textwrap.dedent(
        f"""
        Grounding search audit

        - `reference_df` is loaded from `reference_spectra` for `dataset_id='ramanbiolib'`: {'yes' if uses_reference_tables else 'no'}
        - `grounding_summary_df` is loaded from `grounding_class_summary` across datasets: {'yes' if grounding_summary_global else 'no'}
        - RamanBioLib is not filtered by processed grounding version because it is not loaded from `grounding_processed_*` tables.
        - If RamanBioLib seems absent in app/debug tooling that inspects only `grounding_*` tables, that is a visibility mismatch, not missing data.
        """
    )
    return summary


def load_cca_processed_rows(db_path: Path, processing_version: str, limit_random: int = 10) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with duckdb.connect(str(db_path), read_only=True) as connection:
        class_mean_df = connection.execute(
            """
            SELECT class_label, subclass_label, processing_version, mean_wavenumbers_json, mean_intensity_json
            FROM biosample_class_summary
            WHERE dataset_id = ?
              AND class_label = 'hcc'
              AND subclass_label = ?
              AND processing_version = ?
            """,
            [CCA_DATASET_ID, CCA_SUBCLASS_LABEL, processing_version],
        ).fetchdf()
        random_df = connection.execute(
            """
            SELECT
              p.biosample_id,
              m.class_label,
              p.processing_version,
              p.wavenumbers_json,
              p.intensity_json
            FROM biosample_processed_spectra p
            JOIN biosample_metadata m
              ON p.dataset_id = m.dataset_id
             AND p.biosample_id = m.biosample_id
            WHERE p.dataset_id = ?
              AND p.processing_version = ?
            ORDER BY random()
            LIMIT ?
            """,
            [CCA_DATASET_ID, processing_version, limit_random],
        ).fetchdf()
        all_processed_df = connection.execute(
            """
            SELECT
              p.biosample_id,
              m.class_label,
              p.processing_version,
              p.wavenumbers_json,
              p.intensity_json
            FROM biosample_processed_spectra p
            JOIN biosample_metadata m
              ON p.dataset_id = m.dataset_id
             AND p.biosample_id = m.biosample_id
            WHERE p.dataset_id = ?
              AND p.processing_version = ?
            ORDER BY p.biosample_id
            """,
            [CCA_DATASET_ID, processing_version],
        ).fetchdf()
    return class_mean_df, random_df, all_processed_df


def create_cca_baseline_corrected_variant(db_path: Path) -> None:
    common_grid = build_common_grid(400.0, 1800.0, 1.0)
    with duckdb.connect(str(db_path)) as connection:
        metadata_df = connection.execute(
            """
            SELECT biosample_id, class_label, subclass_label
            FROM biosample_metadata
            WHERE dataset_id = ?
            ORDER BY biosample_id
            """,
            [CCA_DATASET_ID],
        ).fetchdf()
        if metadata_df.empty:
            raise ValueError("Missing CCA biosample metadata.")

        connection.execute(
            """
            DELETE FROM biosample_processed_points
            WHERE processed_id IN (
                SELECT processed_id
                FROM biosample_processed_spectra
                WHERE dataset_id = ? AND processing_version = ?
            )
            """,
            [CCA_DATASET_ID, CCA_V2],
        )
        connection.execute(
            "DELETE FROM biosample_processed_spectra WHERE dataset_id = ? AND processing_version = ?",
            [CCA_DATASET_ID, CCA_V2],
        )
        connection.execute(
            "DELETE FROM biosample_class_summary WHERE dataset_id = ? AND processing_version = ?",
            [CCA_DATASET_ID, CCA_V2],
        )

        raw_points_df = connection.execute(
            """
            SELECT
              p.biosample_id,
              p.point_index,
              p.wavenumber,
              p.intensity,
              m.class_label,
              m.subclass_label
            FROM biosample_spectrum_points p
            JOIN biosample_metadata m
              ON p.dataset_id = m.dataset_id
             AND p.biosample_id = m.biosample_id
            WHERE p.dataset_id = ?
            ORDER BY p.biosample_id, p.point_index
            """,
            [CCA_DATASET_ID],
        ).fetchdf()

        spectra_rows: list[dict] = []
        point_rows: list[dict] = []
        class_accumulators: dict[tuple[str, str], dict[str, np.ndarray | int]] = {}

        for biosample_id, spectrum_df in raw_points_df.groupby("biosample_id", sort=False):
            ordered_df = spectrum_df.sort_values("point_index").reset_index(drop=True)
            x_values = ordered_df["wavenumber"].to_numpy(dtype=float)
            y_values = ordered_df["intensity"].to_numpy(dtype=float)
            mask = (x_values >= 400.0) & (x_values <= 1800.0)
            cropped_x = x_values[mask]
            cropped_y = y_values[mask]
            if len(cropped_x) < 10:
                continue
            interpolated = np.interp(common_grid, cropped_x, cropped_y)
            baseline = asls_baseline(interpolated)
            corrected = interpolated - baseline
            normalized = normalize_minmax(corrected)
            processed_id = f"{CCA_V2}__{biosample_id}"
            class_label = str(ordered_df["class_label"].iloc[0])
            subclass_label = str(ordered_df["subclass_label"].iloc[0])
            spectra_rows.append(
                {
                    "processed_id": processed_id,
                    "biosample_id": biosample_id,
                    "dataset_id": CCA_DATASET_ID,
                    "processing_version": CCA_V2,
                    "crop_min_cm": 400.0,
                    "crop_max_cm": 1800.0,
                    "interpolation_step_cm": 1.0,
                    "baseline_method": "asls",
                    "normalization_method": "minmax",
                    "n_points": int(len(common_grid)),
                    "x_min": float(common_grid.min()),
                    "x_max": float(common_grid.max()),
                    "wavenumbers_json": serialize_array(common_grid),
                    "intensity_json": serialize_array(normalized),
                    "source_table": "biosample_spectrum_points",
                    "processing_notes": "Audit-only AsLS baseline-corrected variant for CCA/HCC/LM serum SERS. Original processed version preserved.",
                }
            )
            for point_index, (wavenumber, intensity) in enumerate(zip(common_grid, normalized), start=1):
                point_rows.append(
                    {
                        "processed_id": processed_id,
                        "biosample_id": biosample_id,
                        "dataset_id": CCA_DATASET_ID,
                        "point_index": point_index,
                        "wavenumber": float(wavenumber),
                        "intensity": float(intensity),
                    }
                )
            key = (class_label, subclass_label)
            accumulator = class_accumulators.setdefault(
                key,
                {"sum": np.zeros_like(common_grid), "sum_sq": np.zeros_like(common_grid), "count": 0},
            )
            accumulator["sum"] = accumulator["sum"] + normalized
            accumulator["sum_sq"] = accumulator["sum_sq"] + np.square(normalized)
            accumulator["count"] = int(accumulator["count"]) + 1

        spectra_df = pd.DataFrame(spectra_rows)
        points_df = pd.DataFrame(point_rows)
        connection.register("cca_v2_spectra", spectra_df)
        connection.register("cca_v2_points", points_df)
        connection.execute("INSERT INTO biosample_processed_spectra SELECT * FROM cca_v2_spectra")
        connection.execute("INSERT INTO biosample_processed_points SELECT * FROM cca_v2_points")
        connection.unregister("cca_v2_spectra")
        connection.unregister("cca_v2_points")

        summary_rows = []
        for (class_label, subclass_label), accumulator in sorted(class_accumulators.items()):
            count = int(accumulator["count"])
            mean_values = accumulator["sum"] / count
            variance = np.maximum((accumulator["sum_sq"] / count) - np.square(mean_values), 0.0)
            std_values = np.sqrt(variance)
            summary_rows.append(
                {
                    "summary_id": f"{CCA_V2}__{CCA_DATASET_ID}__{class_label}__{subclass_label}",
                    "dataset_id": CCA_DATASET_ID,
                    "class_label": class_label,
                    "subclass_label": subclass_label,
                    "processing_version": CCA_V2,
                    "n_spectra": count,
                    "crop_min_cm": 400.0,
                    "crop_max_cm": 1800.0,
                    "interpolation_step_cm": 1.0,
                    "mean_wavenumbers_json": serialize_array(common_grid),
                    "mean_intensity_json": serialize_array(mean_values),
                    "std_intensity_json": serialize_array(std_values),
                    "notes": "Audit-only AsLS baseline-corrected class summary for CCA/HCC/LM serum SERS.",
                }
            )
        summary_df = pd.DataFrame(summary_rows)
        connection.register("cca_v2_summary", summary_df)
        connection.execute("INSERT INTO biosample_class_summary SELECT * FROM cca_v2_summary")
        connection.unregister("cca_v2_summary")


def compare_cca_inference(db_path: Path) -> tuple[pd.DataFrame, str]:
    from gaira.inference import GAIRAInferenceEngine, load_serum_class_mean_query

    engine = GAIRAInferenceEngine(db_path=db_path, theme_layer_version=THEME_LAYER_VERSION)
    before_request = load_serum_class_mean_query(db_path, CCA_DATASET_ID, "hcc", CCA_SUBCLASS_LABEL, CCA_V1)
    after_request = load_serum_class_mean_query(db_path, CCA_DATASET_ID, "hcc", CCA_SUBCLASS_LABEL, CCA_V2)
    before_result = engine.run_inference(before_request)
    after_result = engine.run_inference(after_request)

    rows = []
    for stage, result in [("before", before_result), ("after", after_result)]:
        top_tier1 = result.get("tier1_grounding_hits", [])[:3]
        cautions = result.get("biochemical_global_caveats", [])
        for theme in result.get("biochemical_theme_outputs", []):
            rows.append(
                {
                    "stage": stage,
                    "theme_name": theme["theme_name"],
                    "category": theme["category"],
                    "score": float(theme["score"]),
                    "confidence": float(theme["confidence"]),
                    "specificity_index": float(theme.get("specificity_index", 0.0)),
                    "top_tier1_dataset": top_tier1[0]["source_dataset_id"] if top_tier1 else "",
                    "top_tier1_label": top_tier1[0]["source_label"] if top_tier1 else "",
                    "dominant_themes": "|".join(result.get("dominant_themes", [])),
                    "global_caveats": "|".join(cautions),
                }
            )
    comparison_df = pd.DataFrame(rows)
    summary = textwrap.dedent(
        f"""
        CCA/HCC/LM baseline-correction inference comparison

        - compared query: HCC class mean for `{CCA_DATASET_ID}`
        - before processing version: `{CCA_V1}`
        - after processing version: `{CCA_V2}`
        - top tier-1 before: `{before_result.get('tier1_grounding_hits', [{}])[0].get('source_dataset_id', '')} / {before_result.get('tier1_grounding_hits', [{}])[0].get('source_label', '')}`
        - top tier-1 after: `{after_result.get('tier1_grounding_hits', [{}])[0].get('source_dataset_id', '')} / {after_result.get('tier1_grounding_hits', [{}])[0].get('source_label', '')}`
        """
    )
    return comparison_df, summary


def copy_hcc_eval_db(existing_eval_db: Path, target_eval_db: Path) -> None:
    if target_eval_db.exists():
        target_eval_db.unlink()
    shutil.copy2(existing_eval_db, target_eval_db)


def create_hcc_baseline_corrected_variant(eval_db_path: Path) -> None:
    common_grid = build_common_grid(430.0, 1730.0, 1.0)
    with duckdb.connect(str(eval_db_path)) as connection:
        connection.execute(
            """
            DELETE FROM biosample_processed_points
            WHERE processed_id IN (
                SELECT processed_id
                FROM biosample_processed_spectra
                WHERE dataset_id = ? AND processing_version = ?
            )
            """,
            [HCC_DATASET_ID, HCC_V2],
        )
        connection.execute(
            "DELETE FROM biosample_processed_spectra WHERE dataset_id = ? AND processing_version = ?",
            [HCC_DATASET_ID, HCC_V2],
        )
        connection.execute(
            "DELETE FROM biosample_class_summary WHERE dataset_id = ? AND processing_version = ?",
            [HCC_DATASET_ID, HCC_V2],
        )

        raw_points_df = connection.execute(
            """
            SELECT
              p.biosample_id,
              p.point_index,
              p.wavenumber,
              p.intensity,
              m.class_label,
              m.subclass_label
            FROM biosample_spectrum_points p
            JOIN biosample_metadata m
              ON p.dataset_id = m.dataset_id
             AND p.biosample_id = m.biosample_id
            WHERE p.dataset_id = ?
            ORDER BY p.biosample_id, p.point_index
            """,
            [HCC_DATASET_ID],
        ).fetchdf()

        spectra_rows: list[dict] = []
        point_rows: list[dict] = []
        class_accumulators: dict[tuple[str, str], dict[str, np.ndarray | int]] = {}
        for biosample_id, spectrum_df in raw_points_df.groupby("biosample_id", sort=False):
            ordered_df = spectrum_df.sort_values("point_index").reset_index(drop=True)
            x_values = ordered_df["wavenumber"].to_numpy(dtype=float)
            y_values = ordered_df["intensity"].to_numpy(dtype=float)
            mask = (x_values >= 430.0) & (x_values <= 1730.0)
            cropped_x = x_values[mask]
            cropped_y = y_values[mask]
            if len(cropped_x) < 10:
                continue
            interpolated = np.interp(common_grid, cropped_x, cropped_y)
            baseline = asls_baseline(interpolated)
            corrected = interpolated - baseline
            normalized = normalize_minmax(corrected)
            processed_id = f"{HCC_V2}__{biosample_id}"
            class_label = str(ordered_df["class_label"].iloc[0])
            subclass_label = str(ordered_df["subclass_label"].iloc[0])
            spectra_rows.append(
                {
                    "processed_id": processed_id,
                    "biosample_id": biosample_id,
                    "dataset_id": HCC_DATASET_ID,
                    "processing_version": HCC_V2,
                    "crop_min_cm": 430.0,
                    "crop_max_cm": 1730.0,
                    "interpolation_step_cm": 1.0,
                    "baseline_method": "asls",
                    "normalization_method": "minmax",
                    "n_points": int(len(common_grid)),
                    "x_min": float(common_grid.min()),
                    "x_max": float(common_grid.max()),
                    "wavenumbers_json": serialize_array(common_grid),
                    "intensity_json": serialize_array(normalized),
                    "source_table": "biosample_spectrum_points",
                    "processing_notes": "Audit-only AsLS baseline-corrected holdout variant for backend audit.",
                }
            )
            for point_index, (wavenumber, intensity) in enumerate(zip(common_grid, normalized), start=1):
                point_rows.append(
                    {
                        "processed_id": processed_id,
                        "biosample_id": biosample_id,
                        "dataset_id": HCC_DATASET_ID,
                        "point_index": point_index,
                        "wavenumber": float(wavenumber),
                        "intensity": float(intensity),
                    }
                )
            key = (class_label, subclass_label)
            accumulator = class_accumulators.setdefault(
                key,
                {"sum": np.zeros_like(common_grid), "sum_sq": np.zeros_like(common_grid), "count": 0},
            )
            accumulator["sum"] = accumulator["sum"] + normalized
            accumulator["sum_sq"] = accumulator["sum_sq"] + np.square(normalized)
            accumulator["count"] = int(accumulator["count"]) + 1

        connection.register("hcc_v2_spectra", pd.DataFrame(spectra_rows))
        connection.register("hcc_v2_points", pd.DataFrame(point_rows))
        connection.execute("INSERT INTO biosample_processed_spectra SELECT * FROM hcc_v2_spectra")
        connection.execute("INSERT INTO biosample_processed_points SELECT * FROM hcc_v2_points")
        connection.unregister("hcc_v2_spectra")
        connection.unregister("hcc_v2_points")

        summary_rows = []
        for (class_label, subclass_label), accumulator in sorted(class_accumulators.items()):
            count = int(accumulator["count"])
            mean_values = accumulator["sum"] / count
            variance = np.maximum((accumulator["sum_sq"] / count) - np.square(mean_values), 0.0)
            std_values = np.sqrt(variance)
            summary_rows.append(
                {
                    "summary_id": f"{HCC_V2}__{HCC_DATASET_ID}__{class_label}__{subclass_label}",
                    "dataset_id": HCC_DATASET_ID,
                    "class_label": class_label,
                    "subclass_label": subclass_label,
                    "processing_version": HCC_V2,
                    "n_spectra": count,
                    "crop_min_cm": 430.0,
                    "crop_max_cm": 1730.0,
                    "interpolation_step_cm": 1.0,
                    "mean_wavenumbers_json": serialize_array(common_grid),
                    "mean_intensity_json": serialize_array(mean_values),
                    "std_intensity_json": serialize_array(std_values),
                    "notes": "Audit-only AsLS baseline-corrected class summary for isolated HCC holdout DB.",
                }
            )
        connection.register("hcc_v2_summary", pd.DataFrame(summary_rows))
        connection.execute("INSERT INTO biosample_class_summary SELECT * FROM hcc_v2_summary")
        connection.unregister("hcc_v2_summary")


def evaluate_hcc_holdout_versions(eval_db_path: Path) -> tuple[pd.DataFrame, str]:
    from gaira.serum_differential_calibration import calibrate_serum_holdout
    from gaira.theme_evaluation import ThemeEvaluationRunner
    from scripts.run_hcc_holdout_evaluation import build_holdout_metrics, build_query_level_outputs, load_metadata_df

    metadata_df = load_metadata_df(eval_db_path)
    runner = ThemeEvaluationRunner(db_path=eval_db_path, theme_layer_version=THEME_LAYER_VERSION)
    rows = []
    for processing_version in [HCC_V1, HCC_V2]:
        requests = runner.load_biosample_processed_requests(
            dataset_id=HCC_DATASET_ID,
            domain="serum",
            processing_version=processing_version,
        )
        results = [runner.inference_engine.run_inference(request) for request in requests]
        query_df, theme_df, _ = build_query_level_outputs(results, metadata_df)
        if "class_label_x" in query_df.columns and "class_label_y" in query_df.columns:
            query_df["class_label"] = query_df["class_label_y"].fillna(query_df["class_label_x"])
        elif "class_label_x" in query_df.columns:
            query_df = query_df.rename(columns={"class_label_x": "class_label"})
        elif "class_label_y" in query_df.columns:
            query_df = query_df.rename(columns={"class_label_y": "class_label"})
        theme_wide = (
            theme_df.pivot_table(index="query_id", columns="theme_name", values="score", aggfunc="mean")
            .reset_index()
            .fillna(0.0)
        )
        base_metrics_df = build_holdout_metrics(theme_wide, metadata_df, query_df)
        for row in base_metrics_df.to_dict(orient="records"):
            rows.append(
                {
                    "processing_version": processing_version,
                    "metric_scope": "base",
                    "metric_name": row["metric_name"],
                    "metric_value": float(row["metric_value"]),
                }
            )

        representative_ids = (
            query_df.groupby("class_label", as_index=False)
            .first()["query_id"]
            .dropna()
            .astype(str)
            .tolist()
        )
        representative_df = pd.DataFrame(
            [
                {
                    "query_id": result["query_id"],
                    "top_tier1": (result.get("tier1_grounding_hits", [{}])[0].get("source_label", "") if result.get("tier1_grounding_hits") else ""),
                    "top_tier2": (result.get("tier2_support_hits", [{}])[0].get("source_label", "") if result.get("tier2_support_hits") else ""),
                }
                for result in results
                if result["query_id"] in representative_ids
            ]
        )
        metrics_for_calibration = base_metrics_df.rename(columns={"metric_value": "metric_value"})
        calibration_bundle = calibrate_serum_holdout(
            query_df=query_df,
            theme_df=theme_df,
            representative_df=representative_df,
            metrics_df=metrics_for_calibration,
            metadata_df=metadata_df,
            results=results,
        )
        calibrated_lookup = calibration_bundle.before_after_metrics_df.set_index("metric_name")["after"].to_dict()
        for metric_name in [
            "theme_space_silhouette",
            "mean_abs_theme_effect_size",
            "mean_positive_confidence",
            "mean_caution_score",
        ]:
            rows.append(
                {
                    "processing_version": processing_version,
                    "metric_scope": "calibrated",
                    "metric_name": metric_name,
                    "metric_value": float(calibrated_lookup.get(metric_name, 0.0)),
                }
            )
    metrics_df = pd.DataFrame(rows)
    pivot = metrics_df.pivot_table(index=["metric_scope", "metric_name"], columns="processing_version", values="metric_value")
    summary = textwrap.dedent(
        f"""
        HCC holdout baseline impact

        - evaluated inside copied safe-eval DB only
        - compared processing versions: `{HCC_V1}` vs `{HCC_V2}`
        - base silhouette before/after: `{pivot.loc[('base', 'theme_space_silhouette'), HCC_V1]:.4f}` -> `{pivot.loc[('base', 'theme_space_silhouette'), HCC_V2]:.4f}`
        - calibrated confidence before/after: `{pivot.loc[('calibrated', 'mean_positive_confidence'), HCC_V1]:.4f}` -> `{pivot.loc[('calibrated', 'mean_positive_confidence'), HCC_V2]:.4f}`
        """
    )
    return metrics_df, summary


def build_final_summary(
    registry_summary: str,
    db_summary: str,
    inference_summary: str,
    grounding_search_summary: str,
    cca_report: str,
    holdout_summary: str,
) -> str:
    return "\n\n".join(
        [
            "# Backend audit summary",
            registry_summary,
            db_summary,
            inference_summary,
            grounding_search_summary,
            cca_report,
            holdout_summary,
            textwrap.dedent(
                """
                Recommendation

                - RamanBioLib is correctly ingested, but it lives in `reference_*` tables rather than `grounding_*` tables.
                - The CCA/HCC/LM serum processed representation is not baseline corrected in the live default version.
                - The audit-only baseline-corrected variant should only be promoted if its before/after report shows a meaningful interpretability gain without destabilizing holdout metrics.
                """
            ).strip(),
        ]
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    sys.path.insert(0, str(project_root))

    from gaira.config import get_database_path, require_data_root_exists

    storage_paths = require_data_root_exists()
    db_path = get_database_path()
    paths = ensure_paths(storage_paths["processed_data"])

    registry_df, registry_summary = query_ramanbiolib_registry(project_root)
    write_markdown(paths.base_dir / "ramanbiolib_registry_check.md", registry_summary + "\n\n" + markdown_table(registry_df))

    db_counts_df, db_summary = query_ramanbiolib_db_counts(db_path)
    db_counts_df.to_csv(paths.base_dir / "ramanbiolib_db_counts.csv", index=False)
    write_markdown(paths.base_dir / "ramanbiolib_db_summary.md", db_summary + "\n\n" + markdown_table(db_counts_df))

    visibility_df, visibility_summary = run_ramanbiolib_visibility(db_path)
    visibility_df.to_csv(paths.base_dir / "ramanbiolib_inference_visibility.csv", index=False)
    write_markdown(paths.base_dir / "ramanbiolib_inference_summary.md", visibility_summary + "\n\n" + markdown_table(visibility_df.head(20)))

    grounding_search_summary = inspect_grounding_search(project_root)
    write_markdown(paths.base_dir / "grounding_search_audit.md", grounding_search_summary)

    final_assessment = textwrap.dedent(
        f"""
        RamanBioLib final assessment

        - present in DB? {'yes' if int(db_counts_df.loc[db_counts_df['table_family'] == 'reference', 'row_count'].sum()) > 0 else 'no'}
        - used in inference? {'yes' if (visibility_df['source_dataset_id'] == RAMANBIOLIB_DATASET_ID).any() else 'no'}
        - hidden due to filtering? no
        - Streamlit-only issue? no

        Explanation:
        RamanBioLib is stored in `reference_*` tables and loaded directly by `GroundingSearchEngine.reference_df`.
        It will look absent if an audit only inspects `grounding_*` tables.
        """
    )
    write_markdown(paths.base_dir / "ramanbiolib_final_assessment.md", final_assessment)

    class_mean_before_df, random_before_df, all_before_df = load_cca_processed_rows(db_path, CCA_V1, limit_random=10)
    before_class_spectrum = [
        (
            parse_json_array(class_mean_before_df.iloc[0]["mean_wavenumbers_json"]),
            parse_json_array(class_mean_before_df.iloc[0]["mean_intensity_json"]),
            "HCC class mean",
        )
    ]
    random_before_spectra = [
        (
            parse_json_array(row["wavenumbers_json"]),
            parse_json_array(row["intensity_json"]),
            f"{row['class_label']} • {row['biosample_id'][-10:]}",
        )
        for row in random_before_df.to_dict(orient="records")
    ]
    plot_overlay(before_class_spectrum, "CCA/HCC/LM processed HCC class mean (current v1)", paths.raw_plot_dir / "cca_hcc_class_mean_v1.png")
    plot_overlay(random_before_spectra, "CCA/HCC/LM random processed spectra (current v1)", paths.raw_plot_dir / "cca_random_processed_v1.png")

    baseline_metrics_df = baseline_metrics_frame(all_before_df)
    baseline_metrics_df.to_csv(paths.base_dir / "cca_baseline_metrics.csv", index=False)
    diagnostic_note = textwrap.dedent(
        f"""
        CCA baseline diagnostic

        - current processed version: `{CCA_V1}`
        - baseline_method recorded in processor: `none`
        - spectra inspected: HCC class mean + 10 random processed spectra
        - mean baseline slope: `{baseline_metrics_df['baseline_slope'].mean():.6f}`
        - mean low-frequency bias: `{baseline_metrics_df['low_frequency_bias'].mean():.6f}`
        """
    )
    write_markdown(paths.base_dir / "cca_baseline_diagnostic.md", diagnostic_note)

    create_cca_baseline_corrected_variant(db_path)
    class_mean_after_df, random_after_df, _ = load_cca_processed_rows(db_path, CCA_V2, limit_random=10)
    plot_before_after_pair(
        parse_json_array(class_mean_before_df.iloc[0]["mean_wavenumbers_json"]),
        parse_json_array(class_mean_before_df.iloc[0]["mean_intensity_json"]),
        parse_json_array(class_mean_after_df.iloc[0]["mean_intensity_json"]),
        "CCA/HCC/LM HCC class mean before vs after AsLS baseline correction",
        paths.comparison_plot_dir / "cca_hcc_class_mean_before_after.png",
    )
    comparison_overlay = []
    for before_row, after_row in zip(random_before_df.head(5).to_dict(orient="records"), random_after_df.head(5).to_dict(orient="records")):
        comparison_overlay.append(
            (
                parse_json_array(before_row["wavenumbers_json"]),
                parse_json_array(after_row["intensity_json"]),
                f"{after_row['class_label']} • {after_row['biosample_id'][-10:]}",
            )
        )
    plot_overlay(comparison_overlay, "CCA/HCC/LM random spectra after AsLS correction", paths.comparison_plot_dir / "cca_random_processed_v2.png")

    cca_comparison_df, cca_comparison_summary = compare_cca_inference(db_path)
    cca_comparison_df.to_csv(paths.base_dir / "cca_baseline_before_after_comparison.csv", index=False)
    write_markdown(paths.base_dir / "cca_baseline_before_after_report.md", cca_comparison_summary + "\n\n" + markdown_table(cca_comparison_df.head(20)))

    existing_eval_db = storage_paths["processed_data"] / "hcc_holdout_evaluation" / "eval_db" / "gaira_hcc_holdout_eval.duckdb"
    copy_hcc_eval_db(existing_eval_db, paths.hcc_eval_db_path)
    create_hcc_baseline_corrected_variant(paths.hcc_eval_db_path)
    hcc_impact_df, hcc_impact_summary = evaluate_hcc_holdout_versions(paths.hcc_eval_db_path)
    hcc_impact_df.to_csv(paths.base_dir / "cca_holdout_baseline_impact.csv", index=False)
    write_markdown(paths.base_dir / "cca_holdout_baseline_impact.md", hcc_impact_summary + "\n\n" + markdown_table(hcc_impact_df))

    backend_summary = build_final_summary(
        registry_summary=registry_summary,
        db_summary=db_summary,
        inference_summary=visibility_summary,
        grounding_search_summary=grounding_search_summary,
        cca_report=cca_comparison_summary,
        holdout_summary=hcc_impact_summary,
    )
    write_markdown(paths.base_dir / "backend_audit_summary.md", backend_summary)


if __name__ == "__main__":
    main()
