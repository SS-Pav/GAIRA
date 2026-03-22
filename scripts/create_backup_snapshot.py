import shutil
import sys
from pathlib import Path


def copy_with_versioning(source_path: Path, target_path: Path) -> Path:
    if not source_path.exists():
        raise FileNotFoundError(f"Backup source is missing: {source_path}")

    destination = target_path
    counter = 1
    while destination.exists():
        destination = target_path.with_name(f"{target_path.stem}_v{counter}{target_path.suffix}")
        counter += 1

    shutil.copy2(source_path, destination)
    return destination


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_database_path, get_storage_paths, require_data_root_exists

    storage_paths = require_data_root_exists()
    db_path = get_database_path()
    exports_dir = storage_paths["exports"]
    backups_dir = storage_paths["data_root"] / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)

    summary_path = exports_dir / "rebuild_dataset_summary.csv"
    db_backup_path = backups_dir / "gaira_post_normalization.duckdb"
    summary_backup_path = backups_dir / "rebuild_dataset_summary_post_normalization.csv"

    print("Creating canonical GAIRA backup snapshot.")
    print("DuckDB should be copied only after sequential writes have completed.")

    db_destination = copy_with_versioning(db_path, db_backup_path)
    print(f"Database backup created: {db_destination}")

    if summary_path.exists():
        summary_destination = copy_with_versioning(summary_path, summary_backup_path)
        print(f"Summary backup created: {summary_destination}")
    else:
        print(f"Summary CSV not found; skipped backup: {summary_path}")


if __name__ == "__main__":
    main()
