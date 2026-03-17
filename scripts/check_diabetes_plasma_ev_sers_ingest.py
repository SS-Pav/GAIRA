import duckdb
import pandas as pd
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / "data" / "gaira.duckdb"
    dataset_id = "diabetes_plasma_ev_sers"

    with duckdb.connect(str(db_path), read_only=True) as connection:
        for table_name in [
            "biosample_metadata",
            "biosample_spectra",
            "biosample_spectrum_points",
            "biosample_peaks",
        ]:
            count = connection.execute(
                f"SELECT COUNT(*) FROM {table_name} WHERE dataset_id = ?",
                [dataset_id],
            ).fetchone()[0]
            print(f"{table_name} count: {count}")

        print()
        print("Metadata by class_label:")
        print(
            connection.execute(
                """
                SELECT class_label, subclass_label, COUNT(*) AS n
                FROM biosample_metadata
                WHERE dataset_id = ?
                GROUP BY class_label, subclass_label
                ORDER BY class_label, subclass_label
                """,
                [dataset_id],
            ).fetchdf().to_string(index=False)
        )

        print()
        print("Raw spectra summary:")
        print(
            connection.execute(
                """
                SELECT
                  MIN(x_min) AS min_x,
                  MAX(x_max) AS max_x,
                  MIN(n_points) AS min_points,
                  MAX(n_points) AS max_points
                FROM biosample_spectra
                WHERE dataset_id = ?
                """,
                [dataset_id],
            ).fetchdf().to_string(index=False)
        )

        print()
        print("First metadata rows:")
        print(
            connection.execute(
                """
                SELECT biosample_id, sample_id, replicate_id, class_label, subclass_label, matrix, disease_context, source_file
                FROM biosample_metadata
                WHERE dataset_id = ?
                ORDER BY biosample_id
                LIMIT 5
                """,
                [dataset_id],
            ).fetchdf().to_string(index=False)
        )

        print()
        print("Final framing:")
        print(
            connection.execute(
                """
                SELECT
                  class_label,
                  COUNT(*) AS n,
                  MIN(notes) AS notes
                FROM biosample_metadata
                WHERE dataset_id = ?
                GROUP BY class_label
                ORDER BY class_label
                """,
                [dataset_id],
            ).fetchdf().to_string(index=False)
        )

        print()
        print("First spectra rows:")
        print(
            connection.execute(
                """
                SELECT biosample_id, x_min, x_max, n_points, normalized_flag
                FROM biosample_spectra
                WHERE dataset_id = ?
                ORDER BY biosample_id
                LIMIT 5
                """,
                [dataset_id],
            ).fetchdf().to_string(index=False)
        )

        print()
        print("First peak rows:")
        print(
            connection.execute(
                """
                SELECT biosample_id, peak_rank, peak_cm, peak_intensity, prominence
                FROM biosample_peaks
                WHERE dataset_id = ?
                ORDER BY biosample_id, peak_rank
                LIMIT 10
                """,
                [dataset_id],
            ).fetchdf().to_string(index=False)
        )


if __name__ == "__main__":
    main()
