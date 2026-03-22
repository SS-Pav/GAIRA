import duckdb
from pathlib import Path
import sys


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_database_path

    db_path = get_database_path()
    dataset_id = "adenine_sers_control"

    with duckdb.connect(str(db_path), read_only=True) as connection:
        for table_name in [
            "grounding_metadata",
            "grounding_spectra",
            "grounding_spectrum_points",
            "grounding_peaks",
            "grounding_processed_spectra",
            "grounding_class_summary",
            "grounding_support_documents",
            "grounding_support_chunks",
            "grounding_support_spectra",
            "grounding_support_spectrum_points",
        ]:
            count = connection.execute(
                f"SELECT COUNT(*) FROM {table_name} WHERE dataset_id = ?",
                [dataset_id],
            ).fetchone()[0]
            print(f"{table_name} count: {count}")

        print()
        print("Counts by grounding family / class:")
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


if __name__ == "__main__":
    main()
