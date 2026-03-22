import duckdb
from pathlib import Path
import sys


DATASET_IDS = [
    "sers_fingerprint_workingpaper_support",
    "sers24_metabolite_support",
]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_database_path

    db_path = get_database_path()

    with duckdb.connect(str(db_path), read_only=True) as connection:
        for dataset_id in DATASET_IDS:
            print(f"\n=== {dataset_id} ===")
            document_count = connection.execute(
                "SELECT COUNT(*) FROM grounding_support_documents WHERE dataset_id = ?",
                [dataset_id],
            ).fetchone()[0]
            chunk_count = connection.execute(
                "SELECT COUNT(*) FROM grounding_support_chunks WHERE dataset_id = ?",
                [dataset_id],
            ).fetchone()[0]
            print(f"grounding_support_documents count: {document_count}")
            print(f"grounding_support_chunks count: {chunk_count}")
            print(
                connection.execute(
                    """
                    SELECT citation_label, title, journal, year, source_file
                    FROM grounding_support_documents
                    WHERE dataset_id = ?
                    ORDER BY citation_label
                    """,
                    [dataset_id],
                ).fetchdf().to_string(index=False)
            )


if __name__ == "__main__":
    main()
