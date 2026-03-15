from pathlib import Path

import duckdb


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / "data" / "gaira.duckdb"

    expected_tables = [
        "biosample_metadata",
        "biosample_spectra",
        "biosample_spectrum_points",
        "biosample_peaks",
        "knowledge_sources",
        "knowledge_chunks",
        "peak_assignments",
        "biomarker_claims",
        "confounder_notes",
    ]

    with duckdb.connect(str(db_path), read_only=True) as connection:
        available_tables = set(
            connection.execute("SHOW TABLES").fetchdf()["name"].tolist()
        )

    print("Infrastructure tables:")
    for table_name in expected_tables:
        status = "present" if table_name in available_tables else "missing"
        print(f"  {table_name}: {status}")


if __name__ == "__main__":
    main()
