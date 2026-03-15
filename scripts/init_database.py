from pathlib import Path

import duckdb


def main() -> None:
    # Keep paths relative to the GAIRA project root.
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    db_path = data_dir / "gaira.duckdb"

    # Ensure the data directory exists before creating the database file.
    data_dir.mkdir(exist_ok=True)

    with duckdb.connect(str(db_path)) as db:
        # Registry-friendly dataset table.
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id TEXT,
                name TEXT,
                source_url TEXT,
                source_type TEXT,
                modality TEXT,
                sample_type TEXT,
                matrix_type TEXT,
                n_spectra INTEGER,
                notes TEXT
            )
            """
        )

        # Sample-level metadata for future ingestion.
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS samples (
                sample_id TEXT,
                dataset_id TEXT,
                sample_name TEXT,
                sample_type TEXT,
                matrix_type TEXT,
                pure_or_mixture TEXT,
                biomolecule_label TEXT,
                condition_label TEXT,
                cohort_label TEXT,
                replicate_group TEXT,
                wavelength_nm TEXT,
                modality TEXT,
                notes TEXT
            )
            """
        )

        # Spectrum file tracking for raw and processed data.
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS spectra (
                spectrum_id TEXT,
                sample_id TEXT,
                dataset_id TEXT,
                x_path TEXT,
                y_path TEXT,
                x_min REAL,
                x_max REAL,
                n_points INTEGER,
                raw_or_processed TEXT,
                normalization TEXT,
                baseline_method TEXT,
                source_file TEXT,
                notes TEXT
            )
            """
        )

    print("GAIRA database initialized with datasets, samples, and spectra tables.")


if __name__ == "__main__":
    main()
