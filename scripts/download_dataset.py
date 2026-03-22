import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile, BadZipFile

import pandas as pd


def extract_zip_file(zip_path: Path, target_folder: Path) -> list[str]:
    """Extract a zip archive and return the top-level folder names."""
    try:
        with ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(target_folder)
            top_level_names = {
                Path(member).parts[0]
                for member in zip_file.namelist()
                if Path(member).parts
            }
    except BadZipFile:
        print(f"The downloaded file is not a valid zip archive: {zip_path}")
        return []

    return sorted(top_level_names)


def main() -> None:
    # Make the src package importable when running from the project root.
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import ensure_storage_dirs, resolve_storage_path
    from gaira.github_utils import download_github_repo_zip

    parser = argparse.ArgumentParser(description="Prepare a raw dataset folder for GAIRA.")
    parser.add_argument("dataset_id", help="Dataset identifier from data/registry/datasets.csv")
    parser.add_argument(
        "--allow-holdout",
        action="store_true",
        help="Explicitly allow download for a dataset marked as a holdout in the registry.",
    )
    args = parser.parse_args()

    registry_path = project_root / "data" / "registry" / "datasets.csv"

    if not registry_path.exists():
        print(f"Dataset registry not found: {registry_path}")
        return

    datasets_df = pd.read_csv(registry_path)
    match_df = datasets_df[datasets_df["dataset_id"] == args.dataset_id]

    if match_df.empty:
        print(f"Dataset '{args.dataset_id}' was not found in the registry.")
        return

    storage_config = ensure_storage_dirs()
    raw_data_path = resolve_storage_path(storage_config.get("raw_data"))

    if raw_data_path is None:
        print("The storage config is missing the 'raw_data' path.")
        return

    dataset_row = match_df.iloc[0]
    dataset_status = str(dataset_row.get("status", "")).strip().lower()
    if dataset_status == "holdout" and not args.allow_holdout:
        print(
            f"Dataset '{args.dataset_id}' is marked as a holdout in the registry and is skipped by "
            "default. Re-run with --allow-holdout to download it intentionally."
        )
        raise SystemExit(2)

    target_folder = raw_data_path / args.dataset_id
    target_folder.mkdir(parents=True, exist_ok=True)

    print(f"Dataset found: {dataset_row['name']}")
    print(f"Source URL: {dataset_row['source_url']}")
    print(f"Target folder: {target_folder}")

    if args.dataset_id == "shine_ev_sers":
        zip_url = (
            "https://zenodo.org/records/14768753/files/"
            "SERS-Hepatotoxicity_DATA_CODE_FIGURE.zip?download=1"
        )
        zip_path = target_folder / "SERS-Hepatotoxicity_DATA_CODE_FIGURE.zip"
        print(f"Downloading Zenodo archive: {zip_url}")
        urlretrieve(zip_url, zip_path)
        print(f"Downloaded archive to: {zip_path}")
        top_level_folders = extract_zip_file(zip_path, target_folder)
        print("Extracted top-level folders:")
        for folder_name in top_level_folders:
            print(f"  {folder_name}")
        print("SHINE EV SERS download and extraction step is complete.")
        return

    if args.dataset_id == "small2023_ev":
        download_targets = [
            (
                "Readme.docx",
                "https://zenodo.org/api/records/7011380/files/Readme.docx/content",
            ),
            (
                "NormedProbe1.mat",
                "https://zenodo.org/api/records/7011380/files/NormedProbe1.mat/content",
            ),
            (
                "Main_Text.zip",
                "https://zenodo.org/api/records/7011380/files/Main_Text.zip/content",
            ),
        ]

        print(
            "Downloading the grounded first-ingest assets for small2023_ev. "
            "This includes the README, the normalized Probe1 matrix, and the Main_Text archive "
            "that contains the released Calx axis."
        )

        for file_name, file_url in download_targets:
            output_path = target_folder / file_name
            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"Skipping existing file: {output_path}")
                continue

            print(f"Downloading: {file_url}")
            urlretrieve(file_url, output_path)
            print(f"Saved: {output_path}")

        print("small2023_ev download step is complete.")
        return

    if args.dataset_id == "diabetes_plasma_ev_sers":
        download_targets = [
            (
                "Diabetes_Raw_Data_Codes.zip",
                "https://zenodo.org/api/records/18945379/files/Diabetes%20-%20Raw%20Data%20-%20Codes.zip/content",
            ),
        ]

        print(
            "Downloading the grounded diabetes plasma EV multimodal archive. "
            "The initial GAIRA onboarding targets the released Figure 3 SERS MAT assets inside this zip."
        )

        for file_name, file_url in download_targets:
            output_path = target_folder / file_name
            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"Skipping existing file: {output_path}")
                continue

            print(f"Downloading: {file_url}")
            urlretrieve(file_url, output_path)
            print(f"Saved: {output_path}")

        print("diabetes_plasma_ev_sers download step is complete.")
        return

    if args.dataset_id == "hcc_serum":
        download_targets = [
            (
                "dataset.zip",
                "https://zenodo.org/api/records/4277797/files/dataset.zip/content",
            ),
            (
                "data.csv",
                "https://zenodo.org/api/records/4277797/files/data.csv/content",
            ),
            (
                "R_code.R",
                "https://zenodo.org/api/records/4277797/files/R_code.R/content",
            ),
        ]

        print(
            "Downloading the grounded HCC serum SERS release assets. "
            "GAIRA onboarding will use the 144-spectrum TXT archive as the primary raw path "
            "and keep data.csv plus R_code.R as grounded metadata and preprocessing references."
        )

        for file_name, file_url in download_targets:
            output_path = target_folder / file_name
            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"Skipping existing file: {output_path}")
                continue

            print(f"Downloading: {file_url}")
            urlretrieve(file_url, output_path)
            print(f"Saved: {output_path}")

        print("hcc_serum download step is complete.")
        return

    if args.dataset_id == "serum_ag_colloids":
        download_targets = [
            (
                "dataset_spectral_data.zip",
                "https://zenodo.org/api/records/17374939/files/Dataset%20spectral%20data.zip/content",
            ),
            (
                "scripts_spectral_data.zip",
                "https://zenodo.org/api/records/17374939/files/Scripts%20spectral%20data.zip/content",
            ),
            (
                "Instructions.docx",
                "https://zenodo.org/api/records/17374939/files/Instructions.docx/content",
            ),
        ]

        print(
            "Downloading the grounded serum Ag colloids release assets. "
            "GAIRA onboarding will keep the native BWtek TXT archive untouched and use the released "
            "script and instructions files as grounded metadata/context references."
        )

        for file_name, file_url in download_targets:
            output_path = target_folder / file_name
            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"Skipping existing file: {output_path}")
                continue

            print(f"Downloading: {file_url}")
            urlretrieve(file_url, output_path)
            print(f"Saved: {output_path}")

        print("serum_ag_colloids download step is complete.")
        return

    if args.dataset_id == "serum_protocol_comparison":
        download_targets = [
            (
                "dataset_serum_spectra.zip",
                "https://zenodo.org/api/records/11143059/files/Dataset%20serum%20spectra.zip/content",
            ),
            (
                "Instructions.docx",
                "https://zenodo.org/api/records/11143059/files/Instructions.docx/content",
            ),
            (
                "analysis.R",
                "https://zenodo.org/api/records/11143059/files/Script%20for%20spectra%20analysis.R/content",
            ),
        ]

        print(
            "Downloading the grounded serum protocol-comparison release assets. "
            "GAIRA onboarding will use the native BWtek TXT archive as the primary raw path and keep "
            "the instructions plus released R script as provenance and context references."
        )

        for file_name, file_url in download_targets:
            output_path = target_folder / file_name
            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"Skipping existing file: {output_path}")
                continue

            print(f"Downloading: {file_url}")
            urlretrieve(file_url, output_path)
            print(f"Saved: {output_path}")

        print("serum_protocol_comparison download step is complete.")
        return

    if args.dataset_id == "cspp_serum":
        download_targets = [
            (
                "spectra.zip",
                "https://zenodo.org/api/records/5644790/files/Spectra.zip/content",
            ),
            (
                "scripts.zip",
                "https://zenodo.org/api/records/5644790/files/Scripts.zip/content",
            ),
            (
                "Figure-2_all-spectra-and-metadata.csv",
                "https://zenodo.org/api/records/5644790/files/Figure-2_all-spectra-and-metadata.csv/content",
            ),
            (
                "Figure-4_all-spectra-and-metadata.csv",
                "https://zenodo.org/api/records/5644790/files/Figure-4_all-spectra-and-metadata.csv/content",
            ),
            (
                "Figure-5_all-spectra-and-metadata.csv",
                "https://zenodo.org/api/records/5644790/files/Figure-5_all-spectra-and-metadata.csv/content",
            ),
            (
                "Figure-6_all-spectra-and-metadata.csv",
                "https://zenodo.org/api/records/5644790/files/Figure-6_all-spectra-and-metadata.csv/content",
            ),
            (
                "Figure-7_all-spectra-and-metadata.csv",
                "https://zenodo.org/api/records/5644790/files/Figure-7_all-spectra-and-metadata.csv/content",
            ),
        ]

        print(
            "Downloading the grounded CSPP serum methodology archive assets. "
            "GAIRA onboarding will use the figure-level metadata CSVs as the primary raw ingest path "
            "and retain the matching TXT archive plus released scripts for provenance."
        )

        for file_name, file_url in download_targets:
            output_path = target_folder / file_name
            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"Skipping existing file: {output_path}")
                continue

            print(f"Downloading: {file_url}")
            urlretrieve(file_url, output_path)
            print(f"Saved: {output_path}")

        print("cspp_serum download step is complete.")
        return

    if args.dataset_id == "ergothioneine_serum":
        download_targets = [
            (
                "ERG_calibration.csv",
                "https://zenodo.org/api/records/13791050/files/ERG_calibration.csv/content",
            ),
        ]

        print(
            "Downloading the grounded ergothioneine-in-serum calibration archive. "
            "GAIRA onboarding will use the released calibration CSV as the primary raw ingest path."
        )

        for file_name, file_url in download_targets:
            output_path = target_folder / file_name
            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"Skipping existing file: {output_path}")
                continue

            print(f"Downloading: {file_url}")
            urlretrieve(file_url, output_path)
            print(f"Saved: {output_path}")

        print("ergothioneine_serum download step is complete.")
        return

    if args.dataset_id in {"serum_ag_colloids_grounding", "serum_ag_colloids_literature_grounding"}:
        download_targets = [
            (
                "dataset_spectral_data.zip",
                "https://zenodo.org/api/records/17374939/files/Dataset%20spectral%20data.zip/content",
            ),
            (
                "scripts_spectral_data.zip",
                "https://zenodo.org/api/records/17374939/files/Scripts%20spectral%20data.zip/content",
            ),
            (
                "Instructions.docx",
                "https://zenodo.org/api/records/17374939/files/Instructions.docx/content",
            ),
        ]

        shared_folder = raw_data_path / "serum_ag_colloids"
        shared_folder.mkdir(parents=True, exist_ok=True)

        print(
            "Downloading the shared serum Ag colloids release assets for the grounding-layer asset. "
            "The grounding ingest reuses the same grounded archive as the serum biosample dataset but "
            "targets only selected non-biosample support folders."
        )

        for file_name, file_url in download_targets:
            output_path = shared_folder / file_name
            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"Skipping existing file: {output_path}")
                continue

            print(f"Downloading: {file_url}")
            urlretrieve(file_url, output_path)
            print(f"Saved: {output_path}")

        print(f"{args.dataset_id} download step is complete.")
        return

    if args.dataset_id == "sers_fingerprint_workingpaper_support":
        download_targets = [
            (
                "record_14294417.json",
                "https://zenodo.org/api/records/14294417",
            ),
            (
                "comparing1.pdf",
                "https://zenodo.org/api/records/14294417/files/comparing1.pdf/content",
            ),
        ]

        print(
            "Downloading the PDF-only metabolite SERS fingerprint working-paper support assets. "
            "GAIRA will treat this as support-only grounding evidence because the inspected Zenodo "
            "record does not expose a clean numeric spectral package."
        )

        for file_name, file_url in download_targets:
            output_path = target_folder / file_name
            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"Skipping existing file: {output_path}")
                continue

            print(f"Downloading: {file_url}")
            urlretrieve(file_url, output_path)
            print(f"Saved: {output_path}")

        print("sers_fingerprint_workingpaper_support download step is complete.")
        return

    if args.dataset_id == "sers24_metabolite_support":
        download_targets = [
            (
                "pubmed_37918093.html",
                "https://pubmed.ncbi.nlm.nih.gov/37918093/",
            ),
            (
                "crossref_10_1016_j_saa_2023_123587.json",
                "https://api.crossref.org/works/10.1016/j.saa.2023.123587",
            ),
        ]

        print(
            "Downloading the grounded metadata assets for the 24-metabolite SERS database paper. "
            "No downloadable numeric spectral package was found in this pass, so GAIRA will ingest "
            "this resource as support-only grounding context."
        )

        for file_name, file_url in download_targets:
            output_path = target_folder / file_name
            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"Skipping existing file: {output_path}")
                continue

            print(f"Downloading: {file_url}")
            urlretrieve(file_url, output_path)
            print(f"Saved: {output_path}")

        print("sers24_metabolite_support download step is complete.")
        return

    if args.dataset_id != "ramanbiolib":
        print("Downloader scaffold is ready, but automatic downloading is only implemented for RamanBioLib right now.")
        return

    provenance_url = str(dataset_row.get("provenance_url", "")).strip()
    raw_source_url = str(dataset_row.get("raw_source_url", "")).strip()

    if raw_source_url.lower() in {"", "nan"}:
        print("The RamanBioLib registry row does not include a raw_source_url yet.")
        return

    print(f"Provenance URL: {provenance_url or 'Not provided'}")
    print(f"Raw source URL: {raw_source_url}")

    if "github.com" not in raw_source_url.lower():
        print("The raw_source_url is not a GitHub repository. No specialized downloader is available.")
        return

    zip_path, branch_name = download_github_repo_zip(raw_source_url, target_folder)
    if zip_path is None or branch_name is None:
        print("Could not download the GitHub repository archive from either the main or master branch.")
        print("Please verify the repository URL or download the repository manually.")
        return

    print(f"Downloaded GitHub archive from branch: {branch_name}")
    print(f"Zip file path: {zip_path}")

    top_level_folders = extract_zip_file(zip_path, target_folder)
    print("Extracted top-level folders:")
    if not top_level_folders:
        print("  None detected after extraction.")
    else:
        for folder_name in top_level_folders:
            print(f"  {folder_name}")

    print("GitHub repository download and extraction step is complete.")


if __name__ == "__main__":
    main()
