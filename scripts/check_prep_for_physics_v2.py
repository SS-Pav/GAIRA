from __future__ import annotations

import sys
from pathlib import Path

import duckdb


REQUIRED_PREP_FILES = [
    "diabetes_ev_label_audit.md",
    "diabetes_ev_label_update_summary.md",
    "amino_acid_dataset_inspection.md",
    "amino_acid_ingest_summary.md",
    "amino_acid_validation.txt",
]

REQUIRED_PHYSICS_FILES = [
    "current_processing_audit.csv",
    "current_processing_audit.md",
    "v2_processing_recipe.md",
    "v2_processed_counts.csv",
    "v2_processed_coverage_summary.md",
    "holdout_v2_processing_summary.md",
    "before_after_metrics.csv",
    "before_after_examples.md",
    "cca_holdout_baseline_impact.csv",
    "cca_holdout_baseline_impact.md",
    "final_assessment.md",
]

REQUIRED_PREP_FINAL_FILES = [
    "final_prep_summary.md",
]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from gaira.config import get_database_path, get_storage_paths

    storage_paths = get_storage_paths()
    prep_dir = storage_paths["processed_data"] / "prep_for_physics_v2"
    physics_dir = storage_paths["processed_data"] / "physics_standardization_v2"

    missing = [str(prep_dir / name) for name in REQUIRED_PREP_FILES if not (prep_dir / name).exists()]
    missing += [str(prep_dir / name) for name in REQUIRED_PREP_FINAL_FILES if not (prep_dir / name).exists()]
    missing += [str(physics_dir / name) for name in REQUIRED_PHYSICS_FILES if not (physics_dir / name).exists()]
    if missing:
        raise FileNotFoundError("Missing required prep/physics outputs:\n" + "\n".join(missing))

    db_path = get_database_path()
    with duckdb.connect(str(db_path), read_only=True) as connection:
        aa_counts = connection.execute(
            """
            SELECT
              COUNT(*) FILTER (WHERE table_name = 'grounding_metadata') AS grounding_metadata,
              COUNT(*) FILTER (WHERE table_name = 'grounding_processed_spectra') AS grounding_processed
            FROM (
              SELECT 'grounding_metadata' AS table_name FROM grounding_metadata WHERE dataset_id = 'amino_acid_raman_grounding'
              UNION ALL
              SELECT 'grounding_processed_spectra' AS table_name FROM grounding_processed_spectra WHERE dataset_id = 'amino_acid_raman_grounding'
            )
            """
        ).fetchone()
        diabetes_context_count = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM domain_context_chunks
                WHERE document_id LIKE 'gaira_ev_context_diabetes_%'
                  AND (lower(chunk_text) LIKE '%overweight%' OR lower(chunk_text) LIKE '%bmi%')
                """
            ).fetchone()[0]
        )
        v2_count = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM biosample_processed_spectra
                WHERE processing_version LIKE 'v2_%vector'
                """
            ).fetchone()[0]
        )

    print("prep_for_physics_v2 outputs: ok")
    print("physics_standardization_v2 outputs: ok")
    print(f"amino_acid_raman_grounding metadata rows: {int(aa_counts[0])}")
    print(f"amino_acid_raman_grounding processed rows: {int(aa_counts[1])}")
    print(f"diabetes EV updated context chunks mentioning BMI/overweight: {diabetes_context_count}")
    print(f"live v2 biosample processed rows: {v2_count}")


if __name__ == "__main__":
    main()
