from pathlib import Path

import yaml


def get_project_root() -> Path:
    """Return the GAIRA project root path."""
    return Path(__file__).resolve().parents[2]


def get_storage_config() -> dict:
    """Load storage paths from config/storage.yaml."""
    project_root = get_project_root()
    config_path = project_root / "config" / "storage.yaml"

    # Raise a friendly error if the storage config has not been created yet.
    if not config_path.exists():
        raise FileNotFoundError(
            f"Storage config not found: {config_path}. Create config/storage.yaml first."
        )

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_storage_path(path_value: str | None) -> Path | None:
    """Resolve a storage path relative to the project root when needed."""
    if not path_value:
        return None

    path = Path(path_value)
    if path.is_absolute():
        return path

    return get_project_root() / path


def ensure_storage_dirs() -> dict:
    """Create configured storage folders if they do not exist yet."""
    storage_config = get_storage_config()

    for key in ("raw_data", "processed_data", "cache"):
        path = resolve_storage_path(storage_config.get(key))
        if path is None:
            continue

        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"Created storage directory: {path}")

    return storage_config
