from pathlib import Path

import duckdb


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / "data" / "gaira.duckdb"
    dataset_id = "small2023_ev"

    with duckdb.connect(str(db_path), read_only=True) as connection:
        metadata_count = connection.execute(
            "SELECT COUNT(*) FROM biosample_metadata WHERE dataset_id = ?",
            [dataset_id],
        ).fetchone()[0]
        spectra_count = connection.execute(
            "SELECT COUNT(*) FROM biosample_spectra WHERE dataset_id = ?",
            [dataset_id],
        ).fetchone()[0]
        point_count = connection.execute(
            "SELECT COUNT(*) FROM biosample_spectrum_points WHERE dataset_id = ?",
            [dataset_id],
        ).fetchone()[0]
        peak_count = connection.execute(
            "SELECT COUNT(*) FROM biosample_peaks WHERE dataset_id = ?",
            [dataset_id],
        ).fetchone()[0]

        metadata_preview = connection.execute(
            """
            SELECT biosample_id, class_label, subclass_label, spectral_range, source_file
            FROM biosample_metadata
            WHERE dataset_id = ?
            ORDER BY biosample_id
            LIMIT 5
            """,
            [dataset_id],
        ).fetchdf()
        spectra_preview = connection.execute(
            """
            SELECT biosample_id, x_min, x_max, n_points, normalized_flag
            FROM biosample_spectra
            WHERE dataset_id = ?
            ORDER BY biosample_id
            LIMIT 5
            """,
            [dataset_id],
        ).fetchdf()
        peaks_preview = connection.execute(
            """
            SELECT biosample_id, peak_rank, peak_cm, peak_intensity, prominence
            FROM biosample_peaks
            WHERE dataset_id = ?
            ORDER BY biosample_id, peak_rank
            LIMIT 10
            """,
            [dataset_id],
        ).fetchdf()

    print(f"biosample_metadata count: {metadata_count}")
    print(f"biosample_spectra count: {spectra_count}")
    print(f"biosample_spectrum_points count: {point_count}")
    print(f"biosample_peaks count: {peak_count}")
    print("\nFirst biosample rows:")
    print(metadata_preview.to_string(index=False))
    print("\nFirst spectra summaries:")
    print(spectra_preview.to_string(index=False))
    print("\nFirst peak rows:")
    print(peaks_preview.to_string(index=False))


if __name__ == "__main__":
    main()
