import duckdb
from pathlib import Path
import sys


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_database_path

    db_path = get_database_path()
    dataset_id = "cspp_serum"

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
        print("Counts by subclass and class:")
        print(
            connection.execute(
                """
                SELECT subclass_label, class_label, COUNT(*) AS n
                FROM biosample_metadata
                WHERE dataset_id = ?
                GROUP BY subclass_label, class_label
                ORDER BY subclass_label, class_label
                """,
                [dataset_id],
            ).fetchdf().to_string(index=False)
        )

        print()
        print("Axis summary:")
        print(
            connection.execute(
                """
                SELECT MIN(x_min) AS min_x, MAX(x_max) AS max_x, MIN(n_points) AS min_points, MAX(n_points) AS max_points
                FROM biosample_spectra
                WHERE dataset_id = ?
                """,
                [dataset_id],
            ).fetchdf().to_string(index=False)
        )


if __name__ == "__main__":
    main()
