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

        # Biosample-level metadata for real biological specimen datasets.
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS biosample_metadata (
                biosample_id TEXT,
                dataset_id TEXT,
                source_row_id TEXT,
                sample_id TEXT,
                patient_id TEXT,
                replicate_id TEXT,
                biosample_type TEXT,
                matrix TEXT,
                disease_context TEXT,
                class_label TEXT,
                subclass_label TEXT,
                collection_protocol TEXT,
                preparation_protocol TEXT,
                instrument TEXT,
                laser_wavelength_nm TEXT,
                spectral_range TEXT,
                preprocessing_summary TEXT,
                source_file TEXT,
                notes TEXT
            )
            """
        )

        # Full biosample spectra stored as compact JSON arrays.
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS biosample_spectra (
                biosample_id TEXT,
                dataset_id TEXT,
                source_row_id TEXT,
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

        # One row per biosample spectrum point for simple SQL inspection.
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS biosample_spectrum_points (
                biosample_id TEXT,
                dataset_id TEXT,
                source_row_id TEXT,
                point_index INTEGER,
                wavenumber REAL,
                intensity REAL
            )
            """
        )

        # Optional peak lists extracted from biosample spectra.
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS biosample_peaks (
                biosample_id TEXT,
                dataset_id TEXT,
                source_row_id TEXT,
                peak_rank INTEGER,
                peak_cm REAL,
                peak_intensity REAL,
                prominence REAL
            )
            """
        )

        # Literature and other knowledge sources for a future RAG layer.
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_sources (
                source_id TEXT,
                dataset_id TEXT,
                source_type TEXT,
                title TEXT,
                authors TEXT,
                year TEXT,
                journal TEXT,
                doi TEXT,
                url TEXT,
                citation TEXT,
                license TEXT,
                notes TEXT
            )
            """
        )

        # Chunked knowledge text with lightweight metadata for retrieval later.
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_chunks (
                chunk_id TEXT,
                source_id TEXT,
                dataset_id TEXT,
                section TEXT,
                chunk_text TEXT,
                chunk_order INTEGER,
                page_label TEXT,
                metadata_json TEXT
            )
            """
        )

        # Peak assignment statements extracted from literature.
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS peak_assignments (
                assignment_id TEXT,
                source_id TEXT,
                dataset_id TEXT,
                peak_cm REAL,
                tolerance_cm REAL,
                assigned_molecule TEXT,
                assigned_group TEXT,
                matrix_context TEXT,
                confidence_text TEXT,
                evidence_text TEXT
            )
            """
        )

        # Biomarker claims extracted from literature.
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS biomarker_claims (
                claim_id TEXT,
                source_id TEXT,
                dataset_id TEXT,
                biomarker_name TEXT,
                disease_context TEXT,
                sample_type TEXT,
                spectral_region TEXT,
                claim_text TEXT,
                evidence_strength TEXT,
                notes TEXT
            )
            """
        )

        # Notes about confounders and mitigation strategies.
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS confounder_notes (
                confounder_id TEXT,
                source_id TEXT,
                dataset_id TEXT,
                confounder_name TEXT,
                applies_to TEXT,
                note_text TEXT,
                mitigation_text TEXT
            )
            """
        )

    print(
        "GAIRA database initialized with registry, reference, biosample, and knowledge tables."
    )


if __name__ == "__main__":
    main()
