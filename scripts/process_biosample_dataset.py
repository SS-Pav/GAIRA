import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


PROCESSING_CONFIGS = {
    "shine_ev_sers": {
        "processing_version": "v1_crop450_1800_interp1_minmax",
        "crop_min_cm": 450.0,
        "crop_max_cm": 1800.0,
        "interpolation_step_cm": 1.0,
        "baseline_method": "none",
        "normalization_method": "minmax",
    },
    "small2023_ev": {
        # The released small2023_ev Calx axis is 670..1800 cm^-1.
        "processing_version": "v1_crop670_1800_interp1_minmax",
        "crop_min_cm": 670.0,
        "crop_max_cm": 1800.0,
        "interpolation_step_cm": 1.0,
        "baseline_method": "none",
        "normalization_method": "minmax",
    },
    "hcc_serum": {
        # The released R_code.R crops to 430..1730 cm^-1 after baseline handling.
        # GAIRA reuses that comparison window and a 1 cm common grid, while keeping
        # baseline handling explicit as none in this first processed-layer pass.
        "processing_version": "v1_crop430_1730_interp1_minmax",
        "crop_min_cm": 430.0,
        "crop_max_cm": 1730.0,
        "interpolation_step_cm": 1.0,
        "baseline_method": "none",
        "normalization_method": "minmax",
    },
    "serum_ag_colloids": {
        "processing_version": "v1_crop400_1800_interp1_minmax",
        "crop_min_cm": 400.0,
        "crop_max_cm": 1800.0,
        "interpolation_step_cm": 1.0,
        "baseline_method": "none",
        "normalization_method": "minmax",
    },
    "serum_protocol_comparison": {
        "processing_version": "v1_crop400_1800_interp1_minmax",
        "crop_min_cm": 400.0,
        "crop_max_cm": 1800.0,
        "interpolation_step_cm": 1.0,
        "baseline_method": "none",
        "normalization_method": "minmax",
    },
    "cspp_serum": {
        "processing_version": "v1_crop400_1800_interp1_minmax",
        "crop_min_cm": 400.0,
        "crop_max_cm": 1800.0,
        "interpolation_step_cm": 1.0,
        "baseline_method": "none",
        "normalization_method": "minmax",
    },
    "ergothioneine_serum": {
        "processing_version": "v1_crop400_1800_interp1_minmax",
        "crop_min_cm": 400.0,
        "crop_max_cm": 1800.0,
        "interpolation_step_cm": 1.0,
        "baseline_method": "none",
        "normalization_method": "minmax",
    },
    "diabetes_plasma_ev_sers": {
        "processing_version": "v1_crop500_1600_interp1_minmax",
        "crop_min_cm": 500.0,
        "crop_max_cm": 1600.0,
        "interpolation_step_cm": 1.0,
        "baseline_method": "none",
        "normalization_method": "minmax",
    },
    "covid_serum_raman": {
        "processing_version": "v1_crop400_1800_interp1_minmax",
        "crop_min_cm": 400.0,
        "crop_max_cm": 1800.0,
        "interpolation_step_cm": 1.0,
        "baseline_method": "none",
        "normalization_method": "minmax",
    },
}
SOURCE_TABLE = "biosample_spectrum_points"


def build_common_grid(config: dict) -> np.ndarray:
    """Create the shared comparison grid used for processed biosample spectra."""
    return np.arange(
        config["crop_min_cm"],
        config["crop_max_cm"] + config["interpolation_step_cm"],
        config["interpolation_step_cm"],
    )


def normalize_minmax(intensities: np.ndarray) -> np.ndarray:
    """Scale one spectrum to the 0-1 range."""
    min_value = float(np.min(intensities))
    max_value = float(np.max(intensities))
    value_range = max_value - min_value

    if value_range <= 0:
        return np.zeros_like(intensities, dtype=float)

    return (intensities - min_value) / value_range


def serialize_array(values: np.ndarray) -> str:
    """Store numeric arrays in DuckDB as compact JSON text."""
    return json.dumps([float(value) for value in values])


def build_chunk_query(chunk_size: int) -> str:
    """Create a parameterized query for one biosample chunk."""
    placeholders = ", ".join(["?"] * chunk_size)
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


def process_one_spectrum(
    dataset_id: str,
    biosample_id: str,
    spectrum_df: pd.DataFrame,
    class_label: str | None,
    subclass_label: str | None,
    common_grid: np.ndarray,
    config: dict,
) -> tuple[dict, list[dict], tuple[str | None, str | None], np.ndarray] | None:
    """Crop, interpolate, and normalize one biosample spectrum."""
    ordered_df = spectrum_df.sort_values("point_index").reset_index(drop=True)
    x_values = ordered_df["wavenumber"].to_numpy(dtype=float)
    y_values = ordered_df["intensity"].to_numpy(dtype=float)

    crop_mask = (x_values >= config["crop_min_cm"]) & (x_values <= config["crop_max_cm"])
    cropped_x = x_values[crop_mask]
    cropped_y = y_values[crop_mask]

    if len(cropped_x) < 2:
        print(f"Skipping {biosample_id}: fewer than 2 points remained after cropping.")
        return None

    interpolated_y = np.interp(common_grid, cropped_x, cropped_y)
    normalized_y = normalize_minmax(interpolated_y)
    processed_id = f"{config['processing_version']}__{biosample_id}"

    spectrum_row = {
        "processed_id": processed_id,
        "biosample_id": biosample_id,
        "dataset_id": dataset_id,
        "processing_version": config["processing_version"],
        "crop_min_cm": config["crop_min_cm"],
        "crop_max_cm": config["crop_max_cm"],
        "interpolation_step_cm": config["interpolation_step_cm"],
        "baseline_method": config["baseline_method"],
        "normalization_method": config["normalization_method"],
        "n_points": int(len(common_grid)),
        "x_min": float(common_grid.min()),
        "x_max": float(common_grid.max()),
        "wavenumbers_json": serialize_array(common_grid),
        "intensity_json": serialize_array(normalized_y),
        "source_table": SOURCE_TABLE,
        "processing_notes": (
            f"Raw biosample_spectrum_points were cropped to {config['crop_min_cm']:.0f}-{config['crop_max_cm']:.0f} cm^-1, "
            f"interpolated to a {config['interpolation_step_cm']:.0f} cm^-1 common grid, and min-max normalized."
        ),
    }

    point_rows = [
        {
            "processed_id": processed_id,
            "biosample_id": biosample_id,
            "dataset_id": dataset_id,
            "point_index": point_index,
            "wavenumber": float(wavenumber),
            "intensity": float(intensity),
        }
        for point_index, (wavenumber, intensity) in enumerate(
            zip(common_grid, normalized_y),
            start=1,
        )
    ]

    return spectrum_row, point_rows, (class_label, subclass_label), normalized_y


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from gaira.config import get_database_path, require_data_root_exists

    parser = argparse.ArgumentParser(description="Build a processed biosample layer in DuckDB.")
    parser.add_argument("dataset_id", help="Dataset identifier to process")
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=250,
        help="Number of spectra to process per chunk",
    )
    args = parser.parse_args()

    if args.dataset_id not in PROCESSING_CONFIGS:
        print(
            "Processed biosample support is only implemented for configured datasets in PROCESSING_CONFIGS."
        )
        return

    config = PROCESSING_CONFIGS[args.dataset_id]
    require_data_root_exists()
    db_path = get_database_path()
    common_grid = build_common_grid(config)
    class_accumulators: dict[tuple[str | None, str | None], dict[str, np.ndarray | int]] = defaultdict(dict)

    print(f"Processing biosample dataset: {args.dataset_id}")
    print(f"Database: {db_path}")
    print(f"Processing version: {config['processing_version']}")
    print(f"Common comparison grid: {common_grid[0]:.1f} to {common_grid[-1]:.1f} cm^-1")
    print(f"Processed points per spectrum: {len(common_grid)}")

    with duckdb.connect(str(db_path)) as connection:
        metadata_df = connection.execute(
            """
            SELECT biosample_id, class_label, subclass_label
            FROM biosample_metadata
            WHERE dataset_id = ?
            ORDER BY biosample_id
            """,
            [args.dataset_id],
        ).fetchdf()

        if metadata_df.empty:
            print("No biosample metadata rows were found. Ingest the raw dataset first.")
            return

        biosample_ids = metadata_df["biosample_id"].tolist()
        print(f"Raw biosample spectra available: {len(biosample_ids)}")

        existing_processed_ids = connection.execute(
            """
            SELECT processed_id
            FROM biosample_processed_spectra
            WHERE dataset_id = ? AND processing_version = ?
            """,
            [args.dataset_id, config["processing_version"]],
        ).fetchdf()

        if not existing_processed_ids.empty:
            print("Deleting previous processed rows for this dataset and processing version.")
            connection.execute(
                """
                DELETE FROM biosample_processed_points
                WHERE processed_id IN (
                    SELECT processed_id
                    FROM biosample_processed_spectra
                    WHERE dataset_id = ? AND processing_version = ?
                )
                """,
                [args.dataset_id, config["processing_version"]],
            )
            connection.execute(
                """
                DELETE FROM biosample_processed_spectra
                WHERE dataset_id = ? AND processing_version = ?
                """,
                [args.dataset_id, config["processing_version"]],
            )
            connection.execute(
                """
                DELETE FROM biosample_class_summary
                WHERE dataset_id = ? AND processing_version = ?
                """,
                [args.dataset_id, config["processing_version"]],
            )

        total_processed_spectra = 0
        total_processed_points = 0
        skipped_spectra = 0
        chunk_query_cache: dict[int, str] = {}

        for chunk_start in range(0, len(biosample_ids), args.chunk_size):
            chunk_ids = biosample_ids[chunk_start : chunk_start + args.chunk_size]
            chunk_size = len(chunk_ids)
            chunk_query = chunk_query_cache.get(chunk_size)
            if chunk_query is None:
                chunk_query = build_chunk_query(chunk_size)
                chunk_query_cache[chunk_size] = chunk_query

            chunk_df = connection.execute(
                chunk_query,
                [args.dataset_id, *chunk_ids],
            ).fetchdf()

            if chunk_df.empty:
                print(f"Chunk starting at {chunk_start} returned no rows.")
                continue

            spectra_rows: list[dict] = []
            point_rows: list[dict] = []

            for biosample_id, spectrum_df in chunk_df.groupby("biosample_id", sort=False):
                class_label = spectrum_df["class_label"].iloc[0]
                subclass_label = spectrum_df["subclass_label"].iloc[0]
                processed_result = process_one_spectrum(
                    dataset_id=args.dataset_id,
                    biosample_id=biosample_id,
                    spectrum_df=spectrum_df,
                    class_label=class_label,
                    subclass_label=subclass_label,
                    common_grid=common_grid,
                    config=config,
                )

                if processed_result is None:
                    skipped_spectra += 1
                    continue

                spectrum_row, spectrum_point_rows, group_key, normalized_y = processed_result
                spectra_rows.append(spectrum_row)
                point_rows.extend(spectrum_point_rows)

                accumulator = class_accumulators[group_key]
                if not accumulator:
                    accumulator["sum"] = np.zeros_like(common_grid, dtype=float)
                    accumulator["sum_sq"] = np.zeros_like(common_grid, dtype=float)
                    accumulator["count"] = 0

                accumulator["sum"] = accumulator["sum"] + normalized_y
                accumulator["sum_sq"] = accumulator["sum_sq"] + np.square(normalized_y)
                accumulator["count"] = int(accumulator["count"]) + 1

            if spectra_rows:
                spectra_insert_df = pd.DataFrame(spectra_rows)
                connection.register("processed_spectra_chunk", spectra_insert_df)
                connection.execute(
                    "INSERT INTO biosample_processed_spectra SELECT * FROM processed_spectra_chunk"
                )
                connection.unregister("processed_spectra_chunk")

                points_insert_df = pd.DataFrame(point_rows)
                connection.register("processed_points_chunk", points_insert_df)
                connection.execute(
                    "INSERT INTO biosample_processed_points SELECT * FROM processed_points_chunk"
                )
                connection.unregister("processed_points_chunk")

                total_processed_spectra += len(spectra_rows)
                total_processed_points += len(point_rows)

            print(
                f"Processed chunk {chunk_start + 1}-{chunk_start + chunk_size}: "
                f"{len(spectra_rows)} spectra, {len(point_rows)} processed points"
            )

        summary_rows: list[dict] = []
        for (class_label, subclass_label), accumulator in sorted(class_accumulators.items()):
            count = int(accumulator["count"])
            if count == 0:
                continue

            mean_intensity = accumulator["sum"] / count
            variance = np.maximum((accumulator["sum_sq"] / count) - np.square(mean_intensity), 0.0)
            std_intensity = np.sqrt(variance)
            label_part = class_label or "unknown_class"
            subclass_part = subclass_label or "unknown_subclass"

            summary_rows.append(
                {
                    "summary_id": f"{config['processing_version']}__{args.dataset_id}__{label_part}__{subclass_part}",
                    "dataset_id": args.dataset_id,
                    "class_label": class_label,
                    "subclass_label": subclass_label,
                    "processing_version": config["processing_version"],
                    "n_spectra": count,
                    "crop_min_cm": config["crop_min_cm"],
                    "crop_max_cm": config["crop_max_cm"],
                    "interpolation_step_cm": config["interpolation_step_cm"],
                    "mean_wavenumbers_json": serialize_array(common_grid),
                    "mean_intensity_json": serialize_array(mean_intensity),
                    "std_intensity_json": serialize_array(std_intensity),
                    "notes": (
                        "Class summary was computed from processed biosample spectra grouped "
                        "by class_label and subclass_label."
                    ),
                }
            )

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            connection.register("processed_summary_rows", summary_df)
            connection.execute(
                "INSERT INTO biosample_class_summary SELECT * FROM processed_summary_rows"
            )
            connection.unregister("processed_summary_rows")

    print("Processed biosample layer complete.")
    print(f"Inserted biosample_processed_spectra rows: {total_processed_spectra}")
    print(f"Inserted biosample_processed_points rows: {total_processed_points}")
    print(f"Inserted biosample_class_summary rows: {len(summary_rows)}")
    print(f"Skipped spectra: {skipped_spectra}")


if __name__ == "__main__":
    main()
