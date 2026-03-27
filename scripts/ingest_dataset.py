import argparse
import sys
from pathlib import Path

import pandas as pd


REQUIRED_RAW_FILES: dict[str, tuple[str, ...]] = {
    "metabolite_sers63_support": (
        "fit",
        "peaks",
    ),
    "cca_hcc_lm_serum_sers": (
        "Combination of label-free SERS-based nanosensor an.zip",
    ),
    "covid_serum_raman": (
        "readme.txt",
        "code.m",
        "wave_number.txt",
        "raw_COVID.txt",
        "raw_Helthy.txt",
        "raw_Suspected.txt",
        "raw_Tube.txt",
        "table2_data.txt",
    ),
    "serum_protocol_comparison": (
        "dataset_serum_spectra.zip",
        "Instructions.docx",
        "analysis.R",
    ),
    "cspp_serum": (
        "spectra.zip",
        "scripts.zip",
        "Figure-2_all-spectra-and-metadata.csv",
        "Figure-4_all-spectra-and-metadata.csv",
        "Figure-5_all-spectra-and-metadata.csv",
        "Figure-6_all-spectra-and-metadata.csv",
        "Figure-7_all-spectra-and-metadata.csv",
    ),
    "sers_fingerprint_workingpaper_support": (
        "record_14294417.json",
        "comparing1.pdf",
    ),
    "sers24_metabolite_support": (
        "pubmed_37918093.html",
        "crossref_10_1016_j_saa_2023_123587.json",
    ),
    "adenine_sers_control": (
        "1-s2.0-S0003267025009894-main.pdf",
        "LOD_opakovatelnost.xlsx",
        "ad1ng.CSV",
        "ad1ng_after_two_weeks.CSV",
        "ad1ug_Average.CSV",
        "Adenine_1ng_mL.CSV",
        "Adenine_bAgNPs_100nano.CSV",
        "Adenine_bAgNPs_100pg.CSV",
        "Adenine_bAgNPs_10micro.CSV",
        "Adenine_bAgNPs_10nano.CSV",
        "Adenine_bAgNPs_10pg.CSV",
        "Adenine_bAgNPs_1micro.CSV",
        "bAg-koloid_ad1ug_0.5mW_Average.CSV",
        "bAgNPs_Adenine_1ng_1.CSV",
        "bAgNPs_Adenine_1ng_2.CSV",
        "bAgNPs_Adenine_1ng_3.CSV",
        "bAgNPs_Adenine_1ng_4.CSV",
        "bAgNPs_Adenine_1ng_5.CSV",
        "bg.CSV",
    ),
    "amino_acid_raman_grounding": (
        "aa.xlsx",
    ),
}


def validate_dataset_root(dataset_id: str, dataset_root: Path) -> None:
    """Fail fast when the canonical raw dataset folder is missing or incomplete.

    DuckDB should be treated as a single-writer store. We do not allow ingest to
    proceed against missing raw data or alternate staging paths.
    """
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Canonical raw dataset folder is missing for '{dataset_id}': {dataset_root}"
        )

    if not any(dataset_root.iterdir()):
        raise FileNotFoundError(
            f"Canonical raw dataset folder is empty for '{dataset_id}': {dataset_root}"
        )

    expected_files = REQUIRED_RAW_FILES.get(dataset_id, ())
    missing_files = [file_name for file_name in expected_files if not (dataset_root / file_name).exists()]
    if missing_files:
        raise FileNotFoundError(
            "Canonical raw assets are incomplete for "
            f"'{dataset_id}'. Missing: {', '.join(missing_files)} in {dataset_root}"
        )


def main() -> None:
    # Make the src package importable when running from the project root.
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import ensure_storage_dirs, resolve_storage_path
    from gaira.parsers.biosample.diabetes_plasma_ev_sers_parser import DiabetesPlasmaEVSERSParser
    from gaira.parsers.biosample.cca_hcc_lm_serum_sers_parser import CCAHCCLMSerumSERSParser
    from gaira.parsers.biosample.covid_serum_raman_parser import COVIDSerumRamanParser
    from gaira.parsers.biosample.cspp_serum_parser import CSPPSerumParser
    from gaira.parsers.biosample.ergothioneine_serum_parser import ErgothioneineSerumParser
    from gaira.parsers.biosample.hcc_serum_parser import HCCSerumParser
    from gaira.parsers.biosample.serum_protocol_comparison_parser import SerumProtocolComparisonParser
    from gaira.parsers.biosample.serum_ag_colloids_parser import SerumAgColloidsParser
    from gaira.parsers.biosample.small2023_ev_parser import Small2023EVParser
    from gaira.parsers.grounding.document_support_parser import DocumentSupportParser
    from gaira.parsers.grounding.adenine_sers_control_parser import AdenineSERSControlParser
    from gaira.parsers.grounding.amino_acid_raman_parser import AminoAcidRamanParser
    from gaira.parsers.grounding.metabolite_sers_parser import MetaboliteSERSParser
    from gaira.parsers.grounding.serum_ag_colloids_grounding_parser import (
        SerumAgColloidsGroundingParser,
    )
    from gaira.parsers.grounding.serum_ag_colloids_literature_grounding_parser import (
        SerumAgColloidsLiteratureGroundingParser,
    )
    from gaira.parsers.biosample.shine_ev_sers_parser import ShineEVSERSParser
    from gaira.parsers.knowledge.raman_knowledge_core_parser import RamanKnowledgeCoreParser
    from gaira.parsers.ramanbiolib_parser import RamanBioLibParser

    parser = argparse.ArgumentParser(description="Run a dataset ingestion scaffold for GAIRA.")
    parser.add_argument("dataset_id", help="Dataset identifier to ingest")
    parser.add_argument(
        "--allow-holdout",
        action="store_true",
        help="Explicitly allow ingestion for a dataset marked as a holdout in the registry.",
    )
    args = parser.parse_args()

    print(f"Preparing ingestion scaffold for dataset: {args.dataset_id}")
    registry_path = project_root / "data" / "registry" / "datasets.csv"

    dataset_family = None
    if registry_path.exists():
        registry_df = pd.read_csv(registry_path)
        match_df = registry_df[registry_df["dataset_id"] == args.dataset_id].copy()
        if not match_df.empty and "dataset_family" in match_df.columns:
            dataset_family = str(match_df.iloc[0]["dataset_family"]).strip().lower()
            dataset_status = str(match_df.iloc[0].get("status", "")).strip().lower()
            if dataset_status == "holdout" and not args.allow_holdout:
                print(
                    f"Dataset '{args.dataset_id}' is marked as a holdout in the registry and is "
                    "skipped by default. Re-run with --allow-holdout to ingest it intentionally."
                )
                raise SystemExit(2)

    # Preserve the known RamanBioLib ingestion path exactly.
    if args.dataset_id == "ramanbiolib" or dataset_family == "reference":
        selected_family = "reference"
    elif dataset_family == "biosample":
        selected_family = "biosample"
    elif dataset_family == "grounding":
        selected_family = "grounding"
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
    if args.dataset_id in {"serum_ag_colloids_grounding", "serum_ag_colloids_literature_grounding"}:
        dataset_root = raw_data_path / "serum_ag_colloids"

    print("Storage paths in use:")
    print(f"  raw_data: {raw_data_path}")
    print(f"  processed_data: {resolve_storage_path(storage_config.get('processed_data'))}")
    print(f"  cache: {resolve_storage_path(storage_config.get('cache'))}")
    print(f"  database: {db_path}")
    print(f"Using dataset folder: {dataset_root}")

    validate_dataset_root(args.dataset_id, dataset_root)
    print(f"Starting sequential ingest for dataset: {args.dataset_id}")
    print("DuckDB is single-writer; do not parallelize GAIRA write operations.")

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

        if args.dataset_id == "serum_ag_colloids":
            parser_instance = SerumAgColloidsParser(
                dataset_id=args.dataset_id,
                dataset_root=dataset_root,
                db_path=db_path,
            )
            print("Running serum_ag_colloids biosample ingestion into DuckDB.")
            parser_instance.audit()
            parser_instance.ingest()
            return

        if args.dataset_id == "serum_protocol_comparison":
            parser_instance = SerumProtocolComparisonParser(
                dataset_id=args.dataset_id,
                dataset_root=dataset_root,
                db_path=db_path,
            )
            print("Running serum_protocol_comparison biosample ingestion into DuckDB.")
            parser_instance.audit()
            parser_instance.ingest()
            return

        if args.dataset_id == "cspp_serum":
            parser_instance = CSPPSerumParser(
                dataset_id=args.dataset_id,
                dataset_root=dataset_root,
                db_path=db_path,
            )
            print("Running cspp_serum biosample ingestion into DuckDB.")
            parser_instance.audit()
            parser_instance.ingest()
            return

        if args.dataset_id == "ergothioneine_serum":
            parser_instance = ErgothioneineSerumParser(
                dataset_id=args.dataset_id,
                dataset_root=dataset_root,
                db_path=db_path,
            )
            print("Running ergothioneine_serum biosample ingestion into DuckDB.")
            parser_instance.audit()
            parser_instance.ingest()
            return

        if args.dataset_id == "covid_serum_raman":
            parser_instance = COVIDSerumRamanParser(
                dataset_id=args.dataset_id,
                dataset_root=dataset_root,
                db_path=db_path,
            )
            print("Running covid_serum_raman biosample ingestion into DuckDB.")
            parser_instance.audit()
            parser_instance.ingest()
            return

        if args.dataset_id == "cca_hcc_lm_serum_sers":
            parser_instance = CCAHCCLMSerumSERSParser(
                dataset_id=args.dataset_id,
                dataset_root=dataset_root,
                db_path=db_path,
            )
            print("Running cca_hcc_lm_serum_sers biosample ingestion into DuckDB.")
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

    if selected_family == "grounding":
        if args.dataset_id == "serum_ag_colloids_grounding":
            parser_instance = SerumAgColloidsGroundingParser(
                dataset_id=args.dataset_id,
                dataset_root=dataset_root,
                db_path=db_path,
            )
            print("Running serum_ag_colloids_grounding ingestion into DuckDB.")
            parser_instance.audit()
            parser_instance.ingest()
            return

        if args.dataset_id == "adenine_sers_control":
            parser_instance = AdenineSERSControlParser(
                dataset_id=args.dataset_id,
                dataset_root=dataset_root,
                db_path=db_path,
            )
            print("Running adenine_sers_control grounding ingestion into DuckDB.")
            parser_instance.audit()
            parser_instance.ingest()
            return

        if args.dataset_id == "metabolite_sers63_support":
            parser_instance = MetaboliteSERSParser(
                dataset_id=args.dataset_id,
                dataset_root=dataset_root,
                db_path=db_path,
            )
            print("Running metabolite_sers63_support grounding ingestion into DuckDB.")
            parser_instance.audit()
            parser_instance.ingest()
            return

        if args.dataset_id == "amino_acid_raman_grounding":
            parser_instance = AminoAcidRamanParser(
                dataset_id=args.dataset_id,
                dataset_root=dataset_root,
                db_path=db_path,
            )
            print("Running amino_acid_raman_grounding ingestion into DuckDB.")
            parser_instance.audit()
            parser_instance.ingest()
            return

        if args.dataset_id == "serum_ag_colloids_literature_grounding":
            parser_instance = SerumAgColloidsLiteratureGroundingParser(
                dataset_id=args.dataset_id,
                dataset_root=dataset_root,
                db_path=db_path,
            )
            print("Running serum_ag_colloids_literature_grounding ingestion into DuckDB.")
            parser_instance.audit()
            parser_instance.ingest()
            return

        if args.dataset_id in {"sers_fingerprint_workingpaper_support", "sers24_metabolite_support"}:
            parser_instance = DocumentSupportParser(
                dataset_id=args.dataset_id,
                dataset_root=dataset_root,
                db_path=db_path,
            )
            print(f"Running {args.dataset_id} support-only grounding ingestion into DuckDB.")
            parser_instance.audit()
            parser_instance.ingest()
            return

        print(
            "Grounding parser scaffold exists, but no concrete dataset implementation has "
            "been added yet."
        )
        return

    print(
        "Dataset family could not be resolved from the registry. Add a registry row with "
        "dataset_family set to reference, biosample, grounding, or knowledge."
    )


if __name__ == "__main__":
    main()
