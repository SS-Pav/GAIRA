from pathlib import Path

import duckdb


PROCESSING_VERSION = "v1_crop450_1800_interp1_minmax"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / "data" / "gaira.duckdb"
    dataset_id = "shine_ev_sers"

    with duckdb.connect(str(db_path), read_only=True) as connection:
        processed_spectra_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM biosample_processed_spectra
            WHERE dataset_id = ? AND processing_version = ?
            """,
            [dataset_id, PROCESSING_VERSION],
        ).fetchone()[0]
        processed_points_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM biosample_processed_points
            WHERE dataset_id = ?
            """,
            [dataset_id],
        ).fetchone()[0]
        class_summary_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM biosample_class_summary
            WHERE dataset_id = ? AND processing_version = ?
            """,
            [dataset_id, PROCESSING_VERSION],
        ).fetchone()[0]

        spectra_preview = connection.execute(
            """
            SELECT processed_id, biosample_id, n_points, x_min, x_max, normalization_method
            FROM biosample_processed_spectra
            WHERE dataset_id = ? AND processing_version = ?
            ORDER BY biosample_id
            LIMIT 5
            """,
            [dataset_id, PROCESSING_VERSION],
        ).fetchdf()
        class_preview = connection.execute(
            """
            SELECT class_label, subclass_label, n_spectra, crop_min_cm, crop_max_cm, interpolation_step_cm
            FROM biosample_class_summary
            WHERE dataset_id = ? AND processing_version = ?
            ORDER BY class_label, subclass_label
            LIMIT 5
            """,
            [dataset_id, PROCESSING_VERSION],
        ).fetchdf()
        point_count_summary = connection.execute(
            """
            SELECT
                MIN(point_count) AS min_points,
                MAX(point_count) AS max_points,
                AVG(point_count) AS avg_points
            FROM (
                SELECT processed_id, COUNT(*) AS point_count
                FROM biosample_processed_points
                WHERE dataset_id = ?
                GROUP BY processed_id
            )
            """,
            [dataset_id],
        ).fetchdf()
        class_counts = connection.execute(
            """
            SELECT class_label, COUNT(*) AS summary_rows
            FROM biosample_class_summary
            WHERE dataset_id = ? AND processing_version = ?
            GROUP BY class_label
            ORDER BY class_label
            """,
            [dataset_id, PROCESSING_VERSION],
        ).fetchdf()

    print(f"biosample_processed_spectra count: {processed_spectra_count}")
    print(f"biosample_processed_points count: {processed_points_count}")
    print(f"biosample_class_summary count: {class_summary_count}")
    print("\nFirst 5 processed spectra rows:")
    print(spectra_preview.to_string(index=False))
    print("\nFirst 5 class summaries:")
    print(class_preview.to_string(index=False))
    print("\nProcessed point count summary:")
    print(point_count_summary.to_string(index=False))
    print("\nUnique class_label counts in class summary:")
    print(class_counts.to_string(index=False))


if __name__ == "__main__":
    main()
