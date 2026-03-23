from pathlib import Path

import duckdb


SUPPORT_DATASET_ID = "liver_serum_literature_support"
SERUM_CONTEXT_LAYER = "GAIRA_SERUM_CONTEXT"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(project_root / "src"))
    from gaira.config import get_database_path, require_data_root_exists

    require_data_root_exists()
    db_path = get_database_path()

    with duckdb.connect(str(db_path), read_only=True) as connection:
        support_docs = connection.execute(
            "SELECT COUNT(*) FROM grounding_support_documents WHERE dataset_id = ?",
            [SUPPORT_DATASET_ID],
        ).fetchone()[0]
        support_chunks = connection.execute(
            "SELECT COUNT(*) FROM grounding_support_chunks WHERE dataset_id = ?",
            [SUPPORT_DATASET_ID],
        ).fetchone()[0]
        serum_docs = connection.execute(
            """
            SELECT COUNT(*) FROM domain_context_documents
            WHERE context_layer = ?
              AND intended_domain = 'serum'
              AND (
                document_id LIKE 'gaira_serum_context_liver_%'
                OR document_id LIKE 'gaira_serum_context_hcc_%'
                OR document_id LIKE 'gaira_serum_context_metabolic_%'
              )
            """,
            [SERUM_CONTEXT_LAYER],
        ).fetchone()[0]
        nature_rows = connection.execute(
            "SELECT COUNT(*) FROM grounding_metadata WHERE dataset_id = 'nature_serum_sers'"
        ).fetchone()[0]

    print(f"support docs: {support_docs}")
    print(f"support chunks: {support_chunks}")
    print(f"serum context docs: {serum_docs}")
    print(f"nature_serum_sers grounding rows: {nature_rows}")

    if support_docs < 6 or support_chunks < 20 or serum_docs < 5:
        raise SystemExit("Liver/serum literature integration counts are lower than expected.")
    if nature_rows != 0:
        raise SystemExit("nature_serum_sers unexpectedly reappeared in grounding_metadata.")


if __name__ == "__main__":
    main()
