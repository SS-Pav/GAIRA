import json
from pathlib import Path
import sys

import duckdb
import pandas as pd


CONTEXT_LAYER = "GAIRA_SERUM_CONTEXT"


def build_document(
    document_id: str,
    source_dataset_id: str,
    title: str,
    notes: str,
    chunk_text: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    documents_df = pd.DataFrame(
        [
            {
                "document_id": document_id,
                "context_layer": CONTEXT_LAYER,
                "intended_domain": "serum",
                "context_type": "dataset_context",
                "evidence_basis": "derived_from_dataset_context",
                "source_dataset_id": source_dataset_id,
                "source_file": "dataset_domain_context + subclass_domain_context",
                "title": title,
                "use_for_rag": "yes",
                "notes": notes,
            }
        ]
    )
    chunks_df = pd.DataFrame(
        [
            {
                "chunk_id": f"{document_id}_chunk_01",
                "document_id": document_id,
                "context_layer": CONTEXT_LAYER,
                "intended_domain": "serum",
                "chunk_order": 1,
                "section": "dataset_context",
                "chunk_text": chunk_text,
                "metadata_json": json.dumps({"source_kind": "dataset_domain_context"}, sort_keys=True),
            }
        ]
    )
    return documents_df, chunks_df


def insert_dataframe(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    df: pd.DataFrame,
) -> int:
    if df.empty:
        return 0
    connection.register("temp_df", df)
    connection.execute(f"INSERT INTO {table_name} SELECT * FROM temp_df")
    connection.unregister("temp_df")
    return int(len(df))


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from gaira.config import get_database_path

    db_path = get_database_path()

    with duckdb.connect(str(db_path)) as connection:
        for dataset_id in ("serum_protocol_comparison", "cspp_serum"):
            dataset_df = connection.execute(
                """
                SELECT *
                FROM dataset_domain_context
                WHERE dataset_id = ?
                """,
                [dataset_id],
            ).fetchdf()
            subclass_df = connection.execute(
                """
                SELECT *
                FROM subclass_domain_context
                WHERE dataset_id = ?
                ORDER BY subclass_label
                """,
                [dataset_id],
            ).fetchdf()

            chunk_text = (
                "Dataset-level serum context:\n"
                + dataset_df.to_string(index=False)
                + "\n\nSubclass-level serum context:\n"
                + subclass_df.to_string(index=False)
            )

            document_id = f"gaira_serum_context_{dataset_id}_pass1"
            title = f"{dataset_id} pass-1 serum context"
            notes = (
                "Pass-1 serum-context addition derived directly from dataset_domain_context and "
                "subclass_domain_context after integrating new serum archives."
            )
            documents_df, chunks_df = build_document(
                document_id=document_id,
                source_dataset_id=dataset_id,
                title=title,
                notes=notes,
                chunk_text=chunk_text,
            )

            connection.execute(
                "DELETE FROM domain_context_chunks WHERE document_id = ?",
                [document_id],
            )
            connection.execute(
                "DELETE FROM domain_context_documents WHERE document_id = ?",
                [document_id],
            )
            insert_dataframe(connection, "domain_context_documents", documents_df)
            insert_dataframe(connection, "domain_context_chunks", chunks_df)
            print(f"Inserted serum-context addition for {dataset_id}.")


if __name__ == "__main__":
    main()
