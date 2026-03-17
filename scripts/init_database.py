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

        # Processed biosample spectra aligned to a common comparison grid.
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS biosample_processed_spectra (
                processed_id TEXT,
                biosample_id TEXT,
                dataset_id TEXT,
                processing_version TEXT,
                crop_min_cm REAL,
                crop_max_cm REAL,
                interpolation_step_cm REAL,
                baseline_method TEXT,
                normalization_method TEXT,
                n_points INTEGER,
                x_min REAL,
                x_max REAL,
                wavenumbers_json TEXT,
                intensity_json TEXT,
                source_table TEXT,
                processing_notes TEXT
            )
            """
        )

        # One row per point for the processed biosample spectra.
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS biosample_processed_points (
                processed_id TEXT,
                biosample_id TEXT,
                dataset_id TEXT,
                point_index INTEGER,
                wavenumber REAL,
                intensity REAL
            )
            """
        )

        # Class-level mean and standard deviation spectra for processed biosamples.
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS biosample_class_summary (
                summary_id TEXT,
                dataset_id TEXT,
                class_label TEXT,
                subclass_label TEXT,
                processing_version TEXT,
                n_spectra INTEGER,
                crop_min_cm REAL,
                crop_max_cm REAL,
                interpolation_step_cm REAL,
                mean_wavenumbers_json TEXT,
                mean_intensity_json TEXT,
                std_intensity_json TEXT,
                notes TEXT
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

        # Explicit semantic Raman regions curated for cautious interpretation.
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_regions (
                region_id TEXT,
                dataset_id TEXT,
                region_label TEXT,
                region_min_cm REAL,
                region_max_cm REAL,
                dominant_group TEXT,
                secondary_groups TEXT,
                typical_examples TEXT,
                interpretation_note TEXT,
                caution_note TEXT
            )
            """
        )

        # Dataset-specific acquisition context for cautious interpretation.
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_context (
                context_id TEXT,
                dataset_id TEXT,
                target_dataset_id TEXT,
                modality TEXT,
                sample_type TEXT,
                measurement_state TEXT,
                substrate_type TEXT,
                enhancement_mode TEXT,
                known_biases TEXT,
                region_caution_450_700 TEXT,
                region_caution_700_900 TEXT,
                region_caution_900_1100 TEXT,
                region_caution_1100_1300 TEXT,
                region_caution_1300_1500 TEXT,
                region_caution_1500_1700 TEXT,
                interpretation_note TEXT,
                do_not_overclaim_note TEXT
            )
            """
        )

        # Lightweight measurement/domain context for routing and invariant analysis.
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_domain_context (
                dataset_id TEXT,
                dataset_family TEXT,
                context_level TEXT,
                biosample_type TEXT,
                measurement_mode TEXT,
                default_substrate_type TEXT,
                default_substrate_material TEXT,
                substrate_vendor TEXT,
                instrument_context TEXT,
                default_preprocessing_family TEXT,
                notes TEXT
            )
            """
        )

        db.execute(
            """
            CREATE TABLE IF NOT EXISTS subclass_domain_context (
                dataset_id TEXT,
                subclass_label TEXT,
                context_level TEXT,
                biosample_type TEXT,
                measurement_mode TEXT,
                substrate_type TEXT,
                substrate_material TEXT,
                substrate_vendor TEXT,
                substrate_batch_id TEXT,
                probe_family TEXT,
                spectral_axis_family TEXT,
                cross_domain_intensity_comparable TEXT,
                preprocessing_family TEXT,
                notes TEXT
            )
            """
        )

        db.execute(
            "ALTER TABLE dataset_domain_context ADD COLUMN IF NOT EXISTS substrate_vendor TEXT"
        )
        db.execute(
            "ALTER TABLE subclass_domain_context ADD COLUMN IF NOT EXISTS substrate_vendor TEXT"
        )

        # Grounding-layer metadata for controlled references and perturbation assets that are not
        # biosample benchmark datasets and do not fit the RamanBioLib reference schema cleanly.
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS grounding_metadata (
                grounding_id TEXT,
                dataset_id TEXT,
                source_dataset_id TEXT,
                source_row_id TEXT,
                experiment_family TEXT,
                grounding_role TEXT,
                modality TEXT,
                compound_label TEXT,
                class_label TEXT,
                concentration_label TEXT,
                replicate_id TEXT,
                source_file TEXT,
                biosample_context TEXT,
                substrate_type TEXT,
                substrate_material TEXT,
                instrument TEXT,
                laser_wavelength_nm TEXT,
                spectral_range TEXT,
                preprocessing_summary TEXT,
                notes TEXT
            )
            """
        )

        db.execute(
            """
            CREATE TABLE IF NOT EXISTS grounding_spectra (
                grounding_id TEXT,
                dataset_id TEXT,
                source_dataset_id TEXT,
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

        db.execute(
            """
            CREATE TABLE IF NOT EXISTS grounding_spectrum_points (
                grounding_id TEXT,
                dataset_id TEXT,
                source_dataset_id TEXT,
                source_row_id TEXT,
                point_index INTEGER,
                wavenumber REAL,
                intensity REAL
            )
            """
        )

        db.execute(
            """
            CREATE TABLE IF NOT EXISTS grounding_peaks (
                grounding_id TEXT,
                dataset_id TEXT,
                source_dataset_id TEXT,
                source_row_id TEXT,
                peak_rank INTEGER,
                peak_cm REAL,
                peak_intensity REAL,
                prominence REAL
            )
            """
        )

        db.execute(
            """
            CREATE TABLE IF NOT EXISTS grounding_processed_spectra (
                processed_id TEXT,
                grounding_id TEXT,
                dataset_id TEXT,
                experiment_family TEXT,
                class_label TEXT,
                processing_version TEXT,
                crop_min_cm REAL,
                crop_max_cm REAL,
                interpolation_step_cm REAL,
                baseline_method TEXT,
                normalization_method TEXT,
                n_points INTEGER,
                x_min REAL,
                x_max REAL,
                wavenumbers_json TEXT,
                intensity_json TEXT,
                source_table TEXT,
                processing_notes TEXT
            )
            """
        )

        db.execute(
            """
            CREATE TABLE IF NOT EXISTS grounding_processed_points (
                processed_id TEXT,
                grounding_id TEXT,
                dataset_id TEXT,
                point_index INTEGER,
                wavenumber REAL,
                intensity REAL
            )
            """
        )

        db.execute(
            """
            CREATE TABLE IF NOT EXISTS grounding_class_summary (
                summary_id TEXT,
                dataset_id TEXT,
                experiment_family TEXT,
                class_label TEXT,
                processing_version TEXT,
                n_spectra INTEGER,
                crop_min_cm REAL,
                crop_max_cm REAL,
                interpolation_step_cm REAL,
                mean_wavenumbers_json TEXT,
                mean_intensity_json TEXT,
                std_intensity_json TEXT,
                notes TEXT
            )
            """
        )

    print(
        "GAIRA database initialized with registry, reference, biosample, grounding, knowledge, and domain-context tables."
    )


if __name__ == "__main__":
    main()
