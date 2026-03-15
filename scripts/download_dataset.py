import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    # Make the src package importable when running from the project root.
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_storage_config

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

    storage_config = get_storage_config()
    raw_data_path = storage_config.get("raw_data")

    if not raw_data_path:
        print("The storage config is missing the 'raw_data' path.")
        return

    dataset_row = match_df.iloc[0]
    target_folder = project_root / raw_data_path / args.dataset_id
    target_folder.mkdir(parents=True, exist_ok=True)

    print(f"Dataset found: {dataset_row['name']}")
    print(f"Source URL: {dataset_row['source_url']}")
    print(f"Target folder: {target_folder}")
    print("Downloader scaffold is ready. Actual downloading is not implemented yet.")


if __name__ == "__main__":
    main()
