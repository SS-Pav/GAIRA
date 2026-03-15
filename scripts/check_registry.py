from pathlib import Path

import duckdb


def main() -> None:
    # Use a project-relative database path.
    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / "data" / "gaira.duckdb"

    with duckdb.connect(str(db_path)) as connection:
        datasets_df = connection.execute(
            """
            SELECT dataset_id, name, priority, status
            FROM datasets
            """
        ).fetchdf()

    print(datasets_df.to_string(index=False))
    print(f"\nTotal rows: {len(datasets_df)}")


if __name__ == "__main__":
    main()
