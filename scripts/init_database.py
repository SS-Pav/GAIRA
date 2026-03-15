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

        # Reference-level metadata for RamanBioLib-style biomolecule libraries.
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS reference_metadata (
                ref_id TEXT,
                dataset_id TEXT,
                source_row_id INTEGER,
                component TEXT,
                biochemical_class TEXT,
                submission_date TEXT,
                contact TEXT,
                source TEXT,
                reference TEXT,
                extraction_method TEXT,
                peak_identification TEXT,
                interpolation_method TEXT,
                extra_preprocessing TEXT,
                complete_sample_name TEXT,
                sample_source TEXT,
                sample_composition TEXT,
                sample_preparation TEXT,
                sample_substrate TEXT,
                raman_technique TEXT,
                raman_system TEXT,
                delivery_optics TEXT,
                laser_wavelength_nm TEXT,
                laser_power TEXT,
                acquisition_time TEXT,
                orig_spectral_range TEXT,
                orig_spectral_resolution TEXT,
                orig_spatial_resolution TEXT,
                detector TEXT,
                calibration TEXT,
                cropping TEXT,
                spike_removal TEXT,
                denoising TEXT,
                background_removal TEXT,
                baseline_removal TEXT,
                normalization TEXT,
                additional_info TEXT
            )
            """
        )

        # Individual Raman peak annotations for each reference compound.
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS reference_peaks (
                ref_id TEXT,
                dataset_id TEXT,
                component TEXT,
                source_row_id INTEGER,
                peak_rank INTEGER,
                peak_cm REAL,
                rel_intensity REAL
            )
            """
        )

        # Full reference spectra stored as compact JSON arrays.
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS reference_spectra (
                ref_id TEXT,
                dataset_id TEXT,
                component TEXT,
                source_row_id INTEGER,
                x_min REAL,
                x_max REAL,
                n_points INTEGER,
                wavenumbers_json TEXT,
                intensity_json TEXT,
                normalized_flag TEXT,
                preprocessing_summary TEXT
            )
            """
        )

        # One row per spectrum point for simple SQL inspection and plotting later.
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS reference_spectrum_points (
                ref_id TEXT,
                dataset_id TEXT,
                component TEXT,
                source_row_id INTEGER,
                point_index INTEGER,
                wavenumber REAL,
                intensity REAL
            )
            """
        )

    print(
        "GAIRA database initialized with datasets, samples, spectra, and Raman reference tables."
    )


if __name__ == "__main__":
    main()
