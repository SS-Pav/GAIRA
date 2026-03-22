import duckdb
from pathlib import Path
import sys


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from gaira.config import get_database_path

    db_path = get_database_path()
    dataset_id = "serum_ag_colloids_grounding"

    with duckdb.connect(str(db_path), read_only=True) as connection:
        for table_name in [
            "grounding_metadata",
            "grounding_spectra",
            "grounding_spectrum_points",
            "grounding_peaks",
        ]:
            count = connection.execute(
                f"SELECT COUNT(*) FROM {table_name} WHERE dataset_id = ?",
                [dataset_id],
            ).fetchone()[0]
            print(f"{table_name} count: {count}")

        print()
        print("Grounding rows by experiment_family and class_label:")
        print(
            connection.execute(
                """
                SELECT experiment_family, class_label, COUNT(*) AS n
                FROM grounding_metadata
                WHERE dataset_id = ?
                GROUP BY experiment_family, class_label
                ORDER BY experiment_family, class_label
                """,
                [dataset_id],
            ).fetchdf().to_string(index=False)
        )

        print()
        print("Grounding spectra summary:")
        print(
            connection.execute(
                """
                SELECT
                  MIN(x_min) AS min_x,
                  MAX(x_max) AS max_x,
                  MIN(n_points) AS min_points,
                  MAX(n_points) AS max_points
                FROM grounding_spectra
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
                SELECT grounding_id, experiment_family, grounding_role, compound_label, concentration_label, replicate_id, source_file
                FROM grounding_metadata
                WHERE dataset_id = ?
                ORDER BY grounding_id
                LIMIT 10
                """,
                [dataset_id],
            ).fetchdf().to_string(index=False)
        )


if __name__ == "__main__":
    main()
