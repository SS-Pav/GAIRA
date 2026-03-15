import argparse
import sys
from pathlib import Path
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
    target_folder = raw_data_path / args.dataset_id
    target_folder.mkdir(parents=True, exist_ok=True)

    print(f"Dataset found: {dataset_row['name']}")
    print(f"Source URL: {dataset_row['source_url']}")
    print(f"Target folder: {target_folder}")

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
