import sys
from pathlib import Path


def main() -> None:
    # Make the src package importable when running from the project root.
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_storage_config, resolve_storage_path

    storage_config = get_storage_config()

    print("Configured GAIRA storage paths:")
    print(f"  raw_data: {resolve_storage_path(storage_config.get('raw_data'))}")
    print(f"  processed_data: {resolve_storage_path(storage_config.get('processed_data'))}")
    print(f"  cache: {resolve_storage_path(storage_config.get('cache'))}")
    print(f"  database: {resolve_storage_path(storage_config.get('database'))}")


if __name__ == "__main__":
    main()
