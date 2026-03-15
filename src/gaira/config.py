from pathlib import Path

import yaml


def get_storage_config() -> dict:
    """Load storage paths from config/storage.yaml."""
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "config" / "storage.yaml"

    # Raise a friendly error if the storage config has not been created yet.
    if not config_path.exists():
        raise FileNotFoundError(
            f"Storage config not found: {config_path}. Create config/storage.yaml first."
        )

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
