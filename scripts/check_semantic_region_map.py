import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_storage_config, resolve_storage_path

    storage_config = get_storage_config()
    processed_root = resolve_storage_path(storage_config.get("processed_data"))

    if processed_root is None:
        print("The storage config is missing processed_data.")
        return

    region_map_path = processed_root / "knowledge" / "semantic_region_map.csv"
    if not region_map_path.exists():
        print(f"Semantic region map not found: {region_map_path}")
        return

    region_df = pd.read_csv(region_map_path)
    print(f"Semantic region rows: {len(region_df)}")
    print(region_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
