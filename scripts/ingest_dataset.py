import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    # Make the src package importable when running from the project root.
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import ensure_storage_dirs, resolve_storage_path
    from gaira.parsers.biosample.diabetes_plasma_ev_sers_parser import DiabetesPlasmaEVSERSParser
    from gaira.parsers.biosample.hcc_serum_parser import HCCSerumParser
    from gaira.parsers.biosample.small2023_ev_parser import Small2023EVParser
    from gaira.parsers.biosample.shine_ev_sers_parser import ShineEVSERSParser
    from gaira.parsers.knowledge.raman_knowledge_core_parser import RamanKnowledgeCoreParser
    from gaira.parsers.ramanbiolib_parser import RamanBioLibParser

    parser = argparse.ArgumentParser(description="Run a dataset ingestion scaffold for GAIRA.")
    parser.add_argument("dataset_id", help="Dataset identifier to ingest")
    args = parser.parse_args()

    print(f"Preparing ingestion scaffold for dataset: {args.dataset_id}")
    registry_path = project_root / "data" / "registry" / "datasets.csv"

    dataset_family = None
    if registry_path.exists():
        registry_df = pd.read_csv(registry_path)
        match_df = registry_df[registry_df["dataset_id"] == args.dataset_id].copy()
        if not match_df.empty and "dataset_family" in match_df.columns:
            dataset_family = str(match_df.iloc[0]["dataset_family"]).strip().lower()

    # Preserve the known RamanBioLib ingestion path exactly.
    if args.dataset_id == "ramanbiolib" or dataset_family == "reference":
        selected_family = "reference"
    elif dataset_family == "biosample":
        selected_family = "biosample"
    elif dataset_family == "knowledge":
        selected_family = "knowledge"
    else:
        selected_family = "unknown"

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

    if selected_family == "reference":
        if args.dataset_id != "ramanbiolib":
            print(
                "This reference dataset is registered, but only the RamanBioLib parser is "
                "implemented right now."
            )
            return

        parser_instance = RamanBioLibParser(
            dataset_id=args.dataset_id,
            dataset_root=dataset_root,
            db_path=db_path,
        )
        print("Running full RamanBioLib ingestion into DuckDB.")
        parser_instance.ingest()
        return

    if selected_family == "biosample":
        if args.dataset_id == "shine_ev_sers":
            parser_instance = ShineEVSERSParser(
                dataset_id=args.dataset_id,
                dataset_root=dataset_root,
                db_path=db_path,
            )
            print("Running SHINE EV SERS biosample ingestion into DuckDB.")
            parser_instance.audit()
            parser_instance.ingest()
            return

        if args.dataset_id == "small2023_ev":
            parser_instance = Small2023EVParser(
                dataset_id=args.dataset_id,
                dataset_root=dataset_root,
                db_path=db_path,
            )
            print("Running small2023_ev biosample ingestion into DuckDB.")
            parser_instance.audit()
            parser_instance.ingest()
            return

        if args.dataset_id == "diabetes_plasma_ev_sers":
            parser_instance = DiabetesPlasmaEVSERSParser(
                dataset_id=args.dataset_id,
                dataset_root=dataset_root,
                db_path=db_path,
            )
            print("Running diabetes_plasma_ev_sers biosample ingestion into DuckDB.")
            parser_instance.audit()
            parser_instance.ingest()
            return

        if args.dataset_id == "hcc_serum":
            parser_instance = HCCSerumParser(
                dataset_id=args.dataset_id,
                dataset_root=dataset_root,
                db_path=db_path,
            )
            print("Running hcc_serum biosample ingestion into DuckDB.")
            parser_instance.audit()
            parser_instance.ingest()
            return

        print(
            "Biosample parser scaffold exists, but no concrete dataset implementation has "
            "been added yet."
        )
        return

    if selected_family == "knowledge":
        if args.dataset_id == "raman_knowledge_core":
            parser_instance = RamanKnowledgeCoreParser(
                dataset_id=args.dataset_id,
                dataset_root=dataset_root,
                db_path=db_path,
            )
            print("Running Raman knowledge core ingestion into DuckDB.")
            parser_instance.audit()
            parser_instance.ingest()
            return

        print(
            "Knowledge parser scaffold exists, but no concrete dataset implementation has "
            "been added yet."
        )
        return

    print(
        "Dataset family could not be resolved from the registry. Add a registry row with "
        "dataset_family set to reference, biosample, or knowledge."
    )


if __name__ == "__main__":
    main()
