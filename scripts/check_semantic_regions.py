from pathlib import Path

import duckdb


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / "data" / "gaira.duckdb"
    dataset_id = "raman_knowledge_core"

    with duckdb.connect(str(db_path), read_only=True) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM semantic_regions WHERE dataset_id = ?",
            [dataset_id],
        ).fetchone()[0]
        preview_df = connection.execute(
            """
            SELECT
                region_id,
                region_label,
                region_min_cm,
                region_max_cm,
                dominant_group,
                secondary_groups,
                typical_examples
            FROM semantic_regions
            WHERE dataset_id = ?
            ORDER BY region_min_cm
            LIMIT 12
            """,
            [dataset_id],
        ).fetchdf()

    print(f"semantic_regions count: {count}")
    if preview_df.empty:
        print("No semantic_regions rows found.")
    else:
        print(preview_df.to_string(index=False))


if __name__ == "__main__":
    main()
