from pathlib import Path

import duckdb


PROCESSING_VERSION = "v1_crop450_1800_interp1_minmax"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / "data" / "gaira.duckdb"
    dataset_id = "shine_ev_sers"

    with duckdb.connect(str(db_path), read_only=True) as connection:
        metadata_class_counts = connection.execute(
            """
            SELECT class_label, COUNT(*) AS n_spectra
            FROM biosample_metadata
            WHERE dataset_id = ?
            GROUP BY class_label
            ORDER BY class_label
            """,
            [dataset_id],
        ).fetchdf()
        processed_by_class = connection.execute(
            """
            SELECT m.class_label, COUNT(*) AS n_processed
            FROM biosample_processed_spectra AS p
            JOIN biosample_metadata AS m
              ON p.biosample_id = m.biosample_id
             AND p.dataset_id = m.dataset_id
            WHERE p.dataset_id = ? AND p.processing_version = ?
            GROUP BY m.class_label
            ORDER BY m.class_label
            """,
            [dataset_id, PROCESSING_VERSION],
        ).fetchdf()
        processed_by_subclass = connection.execute(
            """
            SELECT m.subclass_label, COUNT(*) AS n_processed
            FROM biosample_processed_spectra AS p
            JOIN biosample_metadata AS m
              ON p.biosample_id = m.biosample_id
             AND p.dataset_id = m.dataset_id
            WHERE p.dataset_id = ? AND p.processing_version = ?
            GROUP BY m.subclass_label
            ORDER BY m.subclass_label
            """,
            [dataset_id, PROCESSING_VERSION],
        ).fetchdf()
        mean_point_count = connection.execute(
            """
            SELECT AVG(point_count) AS mean_points
            FROM (
                SELECT processed_id, COUNT(*) AS point_count
                FROM biosample_processed_points
                WHERE dataset_id = ?
                GROUP BY processed_id
            )
            """,
            [dataset_id],
        ).fetchone()[0]
        first_class_labels = connection.execute(
            """
            SELECT DISTINCT class_label
            FROM biosample_metadata
            WHERE dataset_id = ?
            ORDER BY class_label
            LIMIT 10
            """,
            [dataset_id],
        ).fetchdf()
        first_class_summaries = connection.execute(
            """
            SELECT class_label, subclass_label, n_spectra
            FROM biosample_class_summary
            WHERE dataset_id = ? AND processing_version = ?
            ORDER BY class_label, subclass_label
            LIMIT 10
            """,
            [dataset_id, PROCESSING_VERSION],
        ).fetchdf()

    print("Class counts from biosample_metadata:")
    print(metadata_class_counts.to_string(index=False))
    print("\nProcessed spectra counts by class_label:")
    print(processed_by_class.to_string(index=False))
    print("\nProcessed spectra counts by subclass_label:")
    print(processed_by_subclass.to_string(index=False))
    print(f"\nMean number of processed points per spectrum: {mean_point_count:.2f}")
    print("\nFirst few class labels:")
    print(first_class_labels.to_string(index=False))
    print("\nFirst few class summary entries:")
    print(first_class_summaries.to_string(index=False))


if __name__ == "__main__":
    main()
