from pathlib import Path
import sys

import duckdb


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from gaira.config import get_database_path

    db_path = get_database_path()

    with duckdb.connect(str(db_path)) as connection:
        table_columns = connection.execute("DESCRIBE datasets").fetchdf()["column_name"].tolist()
        preferred_columns = [
            "dataset_id",
            "dataset_family",
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
