from __future__ import annotations

import sys
from pathlib import Path

import duckdb


REQUIRED_FILES = [
    "method_definitions.md",
    "version_coverage_summary.csv",
    "version_creation_log.md",
    "physics_metrics.csv",
    "visual_comparison_report.md",
    "inference_comparison.csv",
    "inference_comparison_report.md",
    "structure_comparison.csv",
    "structure_comparison_report.md",
    "embedding_input_recommendation.md",
    "final_assessment.md",
]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from gaira.config import get_database_path, get_storage_paths

    storage_paths = get_storage_paths()
    base_dir = storage_paths["processed_data"] / "preprocessing_method_comparison"
    missing = [str(base_dir / name) for name in REQUIRED_FILES if not (base_dir / name).exists()]
    if missing:
        raise FileNotFoundError("Missing preprocessing comparison outputs:\n" + "\n".join(missing))

    db_path = get_database_path()
    with duckdb.connect(str(db_path), read_only=True) as connection:
        asls_summary_count = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM grounding_class_summary
                WHERE processing_version LIKE 'v3_%asls_vector'
                """
            ).fetchone()[0]
        ) + int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM biosample_class_summary
                WHERE processing_version LIKE 'v3_%asls_vector'
                """
            ).fetchone()[0]
        )
        airpls_summary_count = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM grounding_class_summary
                WHERE processing_version LIKE 'v3_%airpls_vector'
                """
            ).fetchone()[0]
        ) + int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM biosample_class_summary
                WHERE processing_version LIKE 'v3_%airpls_vector'
                """
            ).fetchone()[0]
        )

    print("preprocessing_method_comparison outputs: ok")
    print(f"v3 asls class summaries present: {asls_summary_count}")
    print(f"v3 airpls class summaries present: {airpls_summary_count}")


if __name__ == "__main__":
    main()
