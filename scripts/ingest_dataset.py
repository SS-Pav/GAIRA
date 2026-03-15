import argparse
import sys
from pathlib import Path


def main() -> None:
    # Make the src package importable when running from the project root.
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_storage_config
    from gaira.parsers.ramanbiolib_parser import RamanBioLibParser

    parser = argparse.ArgumentParser(description="Run a dataset ingestion scaffold for GAIRA.")
    parser.add_argument("dataset_id", help="Dataset identifier to ingest")
    args = parser.parse_args()

    print(f"Preparing ingestion scaffold for dataset: {args.dataset_id}")

    if args.dataset_id != "ramanbiolib":
        print(f"Parser for '{args.dataset_id}' is not implemented yet.")
        return

    storage_config = get_storage_config()
    raw_data_path = storage_config.get("raw_data")

    if not raw_data_path:
        print("The storage config is missing the 'raw_data' path.")
        return

    dataset_root = project_root / raw_data_path / args.dataset_id
    db_path = project_root / "data" / "gaira.duckdb"

    print(f"Using dataset folder: {dataset_root}")
    print(f"Using database: {db_path}")

    parser_instance = RamanBioLibParser(
        dataset_id=args.dataset_id,
        dataset_root=dataset_root,
        db_path=db_path,
    )
    parser_instance.audit()
    print("Ingestion scaffold finished. Full parsing is still a TODO.")


if __name__ == "__main__":
    main()
