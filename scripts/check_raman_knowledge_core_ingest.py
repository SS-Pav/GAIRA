from pathlib import Path

import duckdb


def preview_table(connection: duckdb.DuckDBPyConnection, query: str, params: list) -> None:
    """Print a small DuckDB preview cleanly."""
    df = connection.execute(query, params).fetchdf()
    if df.empty:
        print("  (no rows)")
    else:
        print(df.to_string(index=False))


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(project_root / "src"))
    from gaira.config import get_database_path, require_data_root_exists

    require_data_root_exists()
    db_path = get_database_path()
    dataset_id = "raman_knowledge_core"

    with duckdb.connect(str(db_path), read_only=True) as connection:
        counts = {
            "knowledge_sources": connection.execute(
                "SELECT COUNT(*) FROM knowledge_sources WHERE dataset_id = ?",
                [dataset_id],
            ).fetchone()[0],
            "peak_assignments": connection.execute(
                "SELECT COUNT(*) FROM peak_assignments WHERE dataset_id = ?",
                [dataset_id],
            ).fetchone()[0],
            "biomarker_claims": connection.execute(
                "SELECT COUNT(*) FROM biomarker_claims WHERE dataset_id = ?",
                [dataset_id],
            ).fetchone()[0],
            "confounder_notes": connection.execute(
                "SELECT COUNT(*) FROM confounder_notes WHERE dataset_id = ?",
                [dataset_id],
            ).fetchone()[0],
            "knowledge_chunks": connection.execute(
                "SELECT COUNT(*) FROM knowledge_chunks WHERE dataset_id = ?",
                [dataset_id],
            ).fetchone()[0],
            "semantic_regions": connection.execute(
                "SELECT COUNT(*) FROM semantic_regions WHERE dataset_id = ?",
                [dataset_id],
            ).fetchone()[0],
            "dataset_context": connection.execute(
                "SELECT COUNT(*) FROM dataset_context WHERE dataset_id = ?",
                [dataset_id],
            ).fetchone()[0],
        }

        print("Raman knowledge core row counts:")
        for table_name, count in counts.items():
            print(f"  {table_name}: {count}")

        print("\nknowledge_sources preview:")
        preview_table(
            connection,
            """
            SELECT source_id, source_type, title, year
            FROM knowledge_sources
            WHERE dataset_id = ?
            ORDER BY source_id
            LIMIT 5
            """,
            [dataset_id],
        )

        print("\npeak_assignments preview:")
        preview_table(
            connection,
            """
            SELECT assignment_id, peak_cm, assigned_molecule, assigned_group
            FROM peak_assignments
            WHERE dataset_id = ?
            ORDER BY peak_cm
            LIMIT 5
            """,
            [dataset_id],
        )

        print("\nbiomarker_claims preview:")
        preview_table(
            connection,
            """
            SELECT claim_id, biomarker_name, disease_context, sample_type
            FROM biomarker_claims
            WHERE dataset_id = ?
            ORDER BY claim_id
            LIMIT 5
            """,
            [dataset_id],
        )

        print("\nconfounder_notes preview:")
        preview_table(
            connection,
            """
            SELECT confounder_id, confounder_name, applies_to
            FROM confounder_notes
            WHERE dataset_id = ?
            ORDER BY confounder_id
            LIMIT 5
            """,
            [dataset_id],
        )

        print("\nknowledge_chunks preview:")
        preview_table(
            connection,
            """
            SELECT chunk_id, section, page_label
            FROM knowledge_chunks
            WHERE dataset_id = ?
            ORDER BY chunk_order
            LIMIT 5
            """,
            [dataset_id],
        )

        print("\nsemantic_regions preview:")
        preview_table(
            connection,
            """
            SELECT region_id, region_label, region_min_cm, region_max_cm, dominant_group
            FROM semantic_regions
            WHERE dataset_id = ?
            ORDER BY region_min_cm
            LIMIT 10
            """,
            [dataset_id],
        )

        print("\ndataset_context preview:")
        preview_table(
            connection,
            """
            SELECT context_id, target_dataset_id, modality, sample_type, enhancement_mode
            FROM dataset_context
            WHERE dataset_id = ?
            ORDER BY context_id
            LIMIT 10
            """,
            [dataset_id],
        )


if __name__ == "__main__":
    main()
