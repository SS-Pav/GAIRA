from pathlib import Path

import duckdb


PROCESSING_VERSION = "v1_crop400_1800_interp1_vector"
DATASET_ID = "serum_ag_colloids_grounding"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / "data" / "gaira.duckdb"

    with duckdb.connect(str(db_path), read_only=True) as connection:
        processed_spectra_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM grounding_processed_spectra
            WHERE dataset_id = ? AND processing_version = ?
            """,
            [DATASET_ID, PROCESSING_VERSION],
        ).fetchone()[0]
        processed_points_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM grounding_processed_points
            WHERE dataset_id = ?
            """,
            [DATASET_ID],
        ).fetchone()[0]
        class_summary_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM grounding_class_summary
            WHERE dataset_id = ? AND processing_version = ?
            """,
            [DATASET_ID, PROCESSING_VERSION],
        ).fetchone()[0]

        spectra_preview = connection.execute(
            """
            SELECT processed_id, grounding_id, experiment_family, class_label, n_points, x_min, x_max, normalization_method
            FROM grounding_processed_spectra
            WHERE dataset_id = ? AND processing_version = ?
            ORDER BY grounding_id
            LIMIT 5
            """,
            [DATASET_ID, PROCESSING_VERSION],
        ).fetchdf()
        class_preview = connection.execute(
            """
            SELECT experiment_family, class_label, n_spectra, crop_min_cm, crop_max_cm, interpolation_step_cm
            FROM grounding_class_summary
            WHERE dataset_id = ? AND processing_version = ?
            ORDER BY experiment_family, class_label
            LIMIT 10
            """,
            [DATASET_ID, PROCESSING_VERSION],
        ).fetchdf()
        point_count_summary = connection.execute(
            """
            SELECT
                MIN(point_count) AS min_points,
                MAX(point_count) AS max_points,
                AVG(point_count) AS avg_points
            FROM (
                SELECT processed_id, COUNT(*) AS point_count
                FROM grounding_processed_points
                WHERE dataset_id = ?
                GROUP BY processed_id
            )
            """,
            [DATASET_ID],
        ).fetchdf()
        family_counts = connection.execute(
            """
            SELECT experiment_family, COUNT(*) AS summary_rows
            FROM grounding_class_summary
            WHERE dataset_id = ? AND processing_version = ?
            GROUP BY experiment_family
            ORDER BY experiment_family
            """,
            [DATASET_ID, PROCESSING_VERSION],
        ).fetchdf()

    print(f"grounding_processed_spectra count: {processed_spectra_count}")
    print(f"grounding_processed_points count: {processed_points_count}")
    print(f"grounding_class_summary count: {class_summary_count}")
    print("\nFirst 5 processed grounding spectra rows:")
    print(spectra_preview.to_string(index=False))
    print("\nFirst 10 grounding class summaries:")
    print(class_preview.to_string(index=False))
    print("\nProcessed point count summary:")
    print(point_count_summary.to_string(index=False))
    print("\nSummary rows by experiment_family:")
    print(family_counts.to_string(index=False))


if __name__ == "__main__":
    main()
