from pathlib import Path
import sys

import duckdb
import pandas as pd


def main() -> None:
    # Build paths from the project root so the script stays portable.
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from gaira.config import get_database_path

    csv_path = project_root / "data" / "registry" / "datasets.csv"
    db_path = get_database_path()

    if not csv_path.exists():
        print(f"Registry file not found: {csv_path}")
        return

    datasets_df = pd.read_csv(csv_path)
    extra_columns = [
        column
        for column in datasets_df.columns
        if column not in {"dataset_id", "name", "priority", "status"}
    ]

    with duckdb.connect(str(db_path)) as connection:
        # Replace the full registry table with the latest CSV contents.
        connection.register("datasets_df", datasets_df)
        connection.execute("CREATE OR REPLACE TABLE datasets AS SELECT * FROM datasets_df")

    print(f"Loaded {len(datasets_df)} dataset records into GAIRA.")
    if extra_columns:
        print(f"Detected {len(extra_columns)} additional registry columns and loaded them too.")


if __name__ == "__main__":
    main()
