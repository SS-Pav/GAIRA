import argparse
import sys
from pathlib import Path


def main() -> None:
    # Make the src package importable when running from the project root.
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import ensure_storage_dirs, resolve_storage_path
    from gaira.parsers.ramanbiolib_parser import RamanBioLibParser

    parser = argparse.ArgumentParser(description="Run a dataset ingestion scaffold for GAIRA.")
    parser.add_argument("dataset_id", help="Dataset identifier to ingest")
    args = parser.parse_args()

    print(f"Preparing ingestion scaffold for dataset: {args.dataset_id}")

    if args.dataset_id != "ramanbiolib":
        print(f"Parser for '{args.dataset_id}' is not implemented yet.")
        return

    storage_config = ensure_storage_dirs()
    raw_data_path = resolve_storage_path(storage_config.get("raw_data"))
    db_path = resolve_storage_path(storage_config.get("database"))

    if raw_data_path is None:
        print("The storage config is missing the 'raw_data' path.")
        return

    if db_path is None:
        print("The storage config is missing the 'database' path.")
        return

    dataset_root = raw_data_path / args.dataset_id

    print("Storage paths in use:")
    print(f"  raw_data: {raw_data_path}")
    print(f"  processed_data: {resolve_storage_path(storage_config.get('processed_data'))}")
    print(f"  cache: {resolve_storage_path(storage_config.get('cache'))}")
    print(f"  database: {db_path}")
    print(f"Using dataset folder: {dataset_root}")

    parser_instance = RamanBioLibParser(
        dataset_id=args.dataset_id,
        dataset_root=dataset_root,
        db_path=db_path,
    )
    print("Running full RamanBioLib ingestion into DuckDB.")
    parser_instance.ingest()


if __name__ == "__main__":
    main()
