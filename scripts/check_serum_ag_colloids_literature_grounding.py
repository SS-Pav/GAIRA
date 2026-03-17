from pathlib import Path

import duckdb


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / "data" / "gaira.duckdb"
    dataset_id = "serum_ag_colloids_literature_grounding"

    with duckdb.connect(str(db_path), read_only=True) as connection:
        for table_name in [
            "grounding_support_documents",
            "grounding_support_chunks",
            "grounding_support_spectra",
            "grounding_support_spectrum_points",
        ]:
            count = connection.execute(
                f"SELECT COUNT(*) FROM {table_name} WHERE dataset_id = ?",
                [dataset_id],
            ).fetchone()[0]
            print(f"{table_name} count: {count}")

        print()
        print("Documents by evidence_family and support_type:")
        print(
            connection.execute(
                """
                SELECT evidence_family, support_type, COUNT(*) AS n
                FROM grounding_support_documents
                WHERE dataset_id = ?
                GROUP BY evidence_family, support_type
                ORDER BY evidence_family, support_type
                """,
                [dataset_id],
            ).fetchdf().to_string(index=False)
        )

        print()
        print("Chunk counts by evidence family:")
        print(
            connection.execute(
                """
                SELECT d.evidence_family, COUNT(c.chunk_id) AS n_chunks
                FROM grounding_support_documents d
                LEFT JOIN grounding_support_chunks c
                  ON d.document_id = c.document_id
                 AND d.dataset_id = c.dataset_id
                WHERE d.dataset_id = ?
                GROUP BY d.evidence_family
                ORDER BY d.evidence_family
                """,
                [dataset_id],
            ).fetchdf().to_string(index=False)
        )

        print()
        print("Digitized support spectra summary:")
        print(
            connection.execute(
                """
                SELECT
                  COUNT(*) AS n_spectra,
                  MIN(x_min) AS min_x,
                  MAX(x_max) AS max_x,
                  MIN(n_points) AS min_points,
                  MAX(n_points) AS max_points
                FROM grounding_support_spectra
                WHERE dataset_id = ?
                """,
                [dataset_id],
            ).fetchdf().to_string(index=False)
        )

        print()
        print("First documents:")
        print(
            connection.execute(
                """
                SELECT document_id, evidence_family, support_type, citation_label, title, source_file
                FROM grounding_support_documents
                WHERE dataset_id = ?
                ORDER BY document_id
                LIMIT 10
                """,
                [dataset_id],
            ).fetchdf().to_string(index=False)
        )


if __name__ == "__main__":
    main()
