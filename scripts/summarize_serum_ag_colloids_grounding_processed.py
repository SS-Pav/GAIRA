from pathlib import Path
import sys

import duckdb


DATASET_ID = "serum_ag_colloids_grounding"
PROCESSING_VERSION = "v1_crop400_1800_interp1_vector"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from gaira.config import get_database_path

    db_path = get_database_path()

    with duckdb.connect(str(db_path), read_only=True) as connection:
        family_counts = connection.execute(
            """
            SELECT experiment_family, COUNT(*) AS n_spectra
            FROM grounding_processed_spectra
            WHERE dataset_id = ? AND processing_version = ?
            GROUP BY experiment_family
            ORDER BY experiment_family
            """,
            [DATASET_ID, PROCESSING_VERSION],
        ).fetchdf()

        class_counts = connection.execute(
            """
            SELECT experiment_family, class_label, COUNT(*) AS n_spectra
            FROM grounding_processed_spectra
            WHERE dataset_id = ? AND processing_version = ?
            GROUP BY experiment_family, class_label
            ORDER BY experiment_family, class_label
            """,
            [DATASET_ID, PROCESSING_VERSION],
        ).fetchdf()

        summary_counts = connection.execute(
            """
            SELECT experiment_family, class_label, n_spectra
            FROM grounding_class_summary
            WHERE dataset_id = ? AND processing_version = ?
            ORDER BY experiment_family, class_label
            """,
            [DATASET_ID, PROCESSING_VERSION],
        ).fetchdf()

    print("Processed grounding spectra by family:")
    print(family_counts.to_string(index=False))
    print("\nProcessed grounding spectra by family and class_label:")
    print(class_counts.to_string(index=False))
    print("\nGrounding class summary rows:")
    print(summary_counts.to_string(index=False))


if __name__ == "__main__":
    main()
