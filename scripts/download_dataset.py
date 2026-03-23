import argparse
import shutil
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


def copy_local_file(source_path: Path, target_path: Path) -> None:
    if target_path.exists() and target_path.stat().st_size > 0:
        print(f"Skipping existing file: {target_path}")
        return
    shutil.copy2(source_path, target_path)
    print(f"Copied: {source_path} -> {target_path}")


def copy_local_tree(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in sorted(source_dir.iterdir()):
        destination = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
            print(f"Copied directory: {item} -> {destination}")
        else:
            copy_local_file(item, destination)


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

    local_source_root = Path.home() / "Downloads" / "New_Set_SERS_Papers_Data"

    if args.dataset_id == "metabolite_sers63_support":
        source_dir = local_source_root / "Metabolite SERS fingerprints Fityk .fit and .peaks files" / "Supplementary-material"
        if not source_dir.exists():
            raise FileNotFoundError(f"Local metabolite archive folder not found: {source_dir}")
        copy_local_tree(source_dir / "fit", target_folder / "fit")
        copy_local_tree(source_dir / "peaks", target_folder / "peaks")
        print("metabolite_sers63_support local copy step is complete.")
        return

    if args.dataset_id == "cca_hcc_lm_serum_sers":
        source_path = local_source_root / "Combination of label-free SERS-based nanosensor an.zip"
        if not source_path.exists():
            raise FileNotFoundError(f"Local cholangio zip not found: {source_path}")
        copy_local_file(source_path, target_folder / source_path.name)
        print("cca_hcc_lm_serum_sers local copy step is complete.")
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

    if args.dataset_id == "covid_serum_raman":
        download_targets = [
            (
                "readme.txt",
                "https://ndownloader.figshare.com/files/22386447",
            ),
            (
                "code.m",
                "https://ndownloader.figshare.com/files/22386408",
            ),
            (
                "wave_number.txt",
                "https://ndownloader.figshare.com/files/22386453",
            ),
            (
                "raw_COVID.txt",
                "https://ndownloader.figshare.com/files/22386432",
            ),
            (
                "raw_Helthy.txt",
                "https://ndownloader.figshare.com/files/22386435",
            ),
            (
                "raw_Suspected.txt",
                "https://ndownloader.figshare.com/files/22386438",
            ),
            (
                "raw_Tube.txt",
                "https://ndownloader.figshare.com/files/22386441",
            ),
            (
                "table2_data.txt",
                "https://ndownloader.figshare.com/files/22386450",
            ),
        ]

        print(
            "Downloading the grounded COVID serum Raman cohort archive. "
            "GAIRA onboarding will use the shared wave_number vector plus raw cohort matrices as the "
            "canonical biosample ingest path and retain the readme, MATLAB code, and Table 2 export "
            "as provenance/context references."
        )

        for file_name, file_url in download_targets:
            output_path = target_folder / file_name
            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"Skipping existing file: {output_path}")
                continue

            print(f"Downloading: {file_url}")
            urlretrieve(file_url, output_path)
            print(f"Saved: {output_path}")

        print("covid_serum_raman download step is complete.")
        return

    if args.dataset_id == "adenine_sers_control":
        download_targets = [
            (
                "1-s2.0-S0003267025009894-main.pdf",
                "https://zenodo.org/api/records/17035751/files/1-s2.0-S0003267025009894-main.pdf/content",
            ),
            (
                "LOD_opakovatelnost.xlsx",
                "https://zenodo.org/api/records/17035751/files/LOD,%20opakovatelnost.xlsx/content",
            ),
            (
                "ad1ng.CSV",
                "https://zenodo.org/api/records/17035751/files/ad1ng.CSV/content",
            ),
            (
                "ad1ng_after_two_weeks.CSV",
                "https://zenodo.org/api/records/17035751/files/ad1ng_after_two_weeks.CSV/content",
            ),
            (
                "ad1ug_Average.CSV",
                "https://zenodo.org/api/records/17035751/files/ad1ug_Average.CSV/content",
            ),
            (
                "Adenine_1ng_mL.CSV",
                "https://zenodo.org/api/records/17035751/files/Adenine_1ng_mL.CSV/content",
            ),
            (
                "Adenine_bAgNPs_100nano.CSV",
                "https://zenodo.org/api/records/17035751/files/Adenine_bAgNPs_100nano.CSV/content",
            ),
            (
                "Adenine_bAgNPs_100pg.CSV",
                "https://zenodo.org/api/records/17035751/files/Adenine_bAgNPs_100pg.CSV/content",
            ),
            (
                "Adenine_bAgNPs_10micro.CSV",
                "https://zenodo.org/api/records/17035751/files/Adenine_bAgNPs_10micro.CSV/content",
            ),
            (
                "Adenine_bAgNPs_10nano.CSV",
                "https://zenodo.org/api/records/17035751/files/Adenine_bAgNPs_10nano.CSV/content",
            ),
            (
                "Adenine_bAgNPs_10pg.CSV",
                "https://zenodo.org/api/records/17035751/files/Adenine_bAgNPs_10pg.CSV/content",
            ),
            (
                "Adenine_bAgNPs_1micro.CSV",
                "https://zenodo.org/api/records/17035751/files/Adenine_bAgNPs_1micro.CSV/content",
            ),
            (
                "bAg-koloid_ad1ug_0.5mW_Average.CSV",
                "https://zenodo.org/api/records/17035751/files/bAg-koloid_ad1ug_0.5mW_Average.CSV/content",
            ),
            (
                "bAgNPs_Adenine_1ng_1.CSV",
                "https://zenodo.org/api/records/17035751/files/bAgNPs_Adenine_1ng_1.CSV/content",
            ),
            (
                "bAgNPs_Adenine_1ng_2.CSV",
                "https://zenodo.org/api/records/17035751/files/bAgNPs_Adenine_1ng_2.CSV/content",
            ),
            (
                "bAgNPs_Adenine_1ng_3.CSV",
                "https://zenodo.org/api/records/17035751/files/bAgNPs_Adenine_1ng_3.CSV/content",
            ),
            (
                "bAgNPs_Adenine_1ng_4.CSV",
                "https://zenodo.org/api/records/17035751/files/bAgNPs_Adenine_1ng_4.CSV/content",
            ),
            (
                "bAgNPs_Adenine_1ng_5.CSV",
                "https://zenodo.org/api/records/17035751/files/bAgNPs_Adenine_1ng_5.CSV/content",
            ),
            (
                "bg.CSV",
                "https://zenodo.org/api/records/17035751/files/bg.CSV/content",
            ),
        ]

        print(
            "Downloading the grounded adenine SERS controlled-reference archive. "
            "GAIRA onboarding will keep the adenine-focused two-column CSV spectra as the primary raw "
            "grounding path and retain the article PDF plus LOD workbook as provenance and calibration references."
        )

        for file_name, file_url in download_targets:
            output_path = target_folder / file_name
            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"Skipping existing file: {output_path}")
                continue

            print(f"Downloading: {file_url}")
            urlretrieve(file_url, output_path)
            print(f"Saved: {output_path}")

        print("adenine_sers_control download step is complete.")
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
