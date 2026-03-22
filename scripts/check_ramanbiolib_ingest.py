from pathlib import Path
import sys

import duckdb


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from gaira.config import get_database_path

    db_path = get_database_path()

    try:
        with duckdb.connect(str(db_path)) as connection:
            metadata_count = connection.execute(
                "SELECT COUNT(*) FROM reference_metadata WHERE dataset_id = 'ramanbiolib'"
            ).fetchone()[0]
            peaks_count = connection.execute(
                "SELECT COUNT(*) FROM reference_peaks WHERE dataset_id = 'ramanbiolib'"
            ).fetchone()[0]
            spectra_count = connection.execute(
                "SELECT COUNT(*) FROM reference_spectra WHERE dataset_id = 'ramanbiolib'"
            ).fetchone()[0]
            spectrum_points_count = connection.execute(
                "SELECT COUNT(*) FROM reference_spectrum_points WHERE dataset_id = 'ramanbiolib'"
            ).fetchone()[0]

            first_components = connection.execute(
                """
                SELECT component
                FROM reference_metadata
                WHERE dataset_id = 'ramanbiolib'
                ORDER BY component
                LIMIT 5
                """
            ).fetchdf()
            first_peaks = connection.execute(
                """
                SELECT component, peak_rank, peak_cm, rel_intensity
                FROM reference_peaks
                WHERE dataset_id = 'ramanbiolib'
                ORDER BY component, peak_rank
                LIMIT 10
                """
            ).fetchdf()
            spectra_summary = connection.execute(
                """
                SELECT component, x_min, x_max, n_points
                FROM reference_spectra
                WHERE dataset_id = 'ramanbiolib'
                ORDER BY component
                LIMIT 3
                """
            ).fetchdf()
            point_preview = connection.execute(
                """
                SELECT component, point_index, wavenumber, intensity
                FROM reference_spectrum_points
                WHERE dataset_id = 'ramanbiolib'
                ORDER BY component, point_index
                LIMIT 10
                """
            ).fetchdf()
    except duckdb.Error as exc:
        print("Could not verify the RamanBioLib ingestion yet.")
        print("Make sure `scripts/init_database.py` has run and the ingestion completed successfully.")
        print(f"DuckDB error: {exc}")
        return

    print(f"reference_metadata rows: {metadata_count}")
    print(f"reference_peaks rows: {peaks_count}")
    print(f"reference_spectra rows: {spectra_count}")
    print(f"reference_spectrum_points rows: {spectrum_points_count}")
    print("\nFirst 5 components from reference_metadata:")
    print(first_components.to_string(index=False))
    print("\nFirst 10 peaks from reference_peaks:")
    print(first_peaks.to_string(index=False))
    print("\nFirst 3 spectra summary rows from reference_spectra:")
    print(spectra_summary.to_string(index=False))
    print("\nFirst 10 rows from reference_spectrum_points:")
    print(point_preview.to_string(index=False))


if __name__ == "__main__":
    main()
