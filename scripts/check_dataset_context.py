from pathlib import Path
import sys

import duckdb


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from gaira.config import get_database_path

    db_path = get_database_path()

    with duckdb.connect(str(db_path), read_only=True) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM dataset_context"
        ).fetchone()[0]
        preview_df = connection.execute(
            """
            SELECT
                context_id,
                dataset_id,
                target_dataset_id,
                modality,
                sample_type,
                substrate_type,
                enhancement_mode
            FROM dataset_context
            ORDER BY context_id
            """
        ).fetchdf()

    print(f"dataset_context row count: {count}")
    if preview_df.empty:
        print("(no dataset_context rows)")
    else:
        print(preview_df.to_string(index=False))


if __name__ == "__main__":
    main()
