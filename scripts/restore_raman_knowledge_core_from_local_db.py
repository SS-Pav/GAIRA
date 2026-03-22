import sys
from pathlib import Path

import duckdb
import pandas as pd


DATASET_ID = "raman_knowledge_core"
TABLE_EXPORTS = {
    "knowledge_sources": "sources.csv",
    "peak_assignments": "peak_assignments.csv",
    "biomarker_claims": "biomarker_claims.csv",
    "confounder_notes": "confounder_notes.csv",
    "knowledge_chunks": "knowledge_chunks.csv",
    "semantic_regions": "semantic_regions.csv",
    "dataset_context": "dataset_context.csv",
}


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import require_data_root_exists, get_storage_paths

    require_data_root_exists()
    storage_paths = get_storage_paths()
    source_db_path = project_root / "data" / "gaira.duckdb"
    target_root = storage_paths["raw_data"] / DATASET_ID
    target_root.mkdir(parents=True, exist_ok=True)

    if not source_db_path.exists():
        raise FileNotFoundError(f"Recovery source DB not found: {source_db_path}")

    with duckdb.connect(str(source_db_path), read_only=True) as connection:
        for table_name, file_name in TABLE_EXPORTS.items():
            df = connection.execute(
                f"SELECT * FROM {table_name} WHERE dataset_id = ?",
                [DATASET_ID],
            ).fetchdf()
            output_path = target_root / file_name
            if df.empty:
                if output_path.exists():
                    output_path.unlink()
                print(f"No rows found for {table_name}; skipped {file_name}")
                continue
            df.to_csv(output_path, index=False)
            print(f"Restored {file_name}: {len(df)} rows -> {output_path}")


if __name__ == "__main__":
    main()
