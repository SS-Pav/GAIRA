from pathlib import Path
import sys

import duckdb


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from gaira.config import get_database_path

    db_path = get_database_path()

    with duckdb.connect(str(db_path), read_only=True) as connection:
        for table_name in ["domain_context_documents", "domain_context_chunks"]:
            count = connection.execute(
                f"""
                SELECT COUNT(*)
                FROM {table_name}
                WHERE context_layer = 'GAIRA_EV_CONTEXT'
                  AND intended_domain = 'ev'
                """
            ).fetchone()[0]
            print(f"{table_name} count: {count}")

        print()
        print("Documents by context_type and evidence_basis:")
        print(
            connection.execute(
                """
                SELECT context_type, evidence_basis, COUNT(*) AS n
                FROM domain_context_documents
                WHERE context_layer = 'GAIRA_EV_CONTEXT'
                  AND intended_domain = 'ev'
                GROUP BY context_type, evidence_basis
                ORDER BY context_type, evidence_basis
                """
            ).fetchdf().to_string(index=False)
        )

        print()
        print("First EV-context documents:")
        print(
            connection.execute(
                """
                SELECT document_id, context_type, evidence_basis, source_dataset_id, title
                FROM domain_context_documents
                WHERE context_layer = 'GAIRA_EV_CONTEXT'
                  AND intended_domain = 'ev'
                ORDER BY document_id
                """
            ).fetchdf().to_string(index=False)
        )

        print()
        print("First EV-context chunks:")
        print(
            connection.execute(
                """
                SELECT chunk_id, section, substr(chunk_text, 1, 260) AS chunk_text
                FROM domain_context_chunks
                WHERE context_layer = 'GAIRA_EV_CONTEXT'
                  AND intended_domain = 'ev'
                ORDER BY chunk_id
                LIMIT 10
                """
            ).fetchdf().to_string(index=False)
        )


if __name__ == "__main__":
    main()
