from pathlib import Path

import duckdb


def main() -> None:
    # Use a project-relative database path.
    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / "data" / "gaira.duckdb"

    with duckdb.connect(str(db_path)) as connection:
        table_columns = connection.execute("DESCRIBE datasets").fetchdf()["column_name"].tolist()
        preferred_columns = [
            "dataset_id",
            "name",
            "priority",
            "status",
            "provenance_url",
            "raw_source_url",
        ]
        selected_columns = [column for column in preferred_columns if column in table_columns]

        datasets_df = connection.execute(
            f"SELECT {', '.join(selected_columns)} FROM datasets"
        ).fetchdf()

    print(datasets_df.to_string(index=False))
    print(f"\nTotal rows: {len(datasets_df)}")


if __name__ == "__main__":
    main()
