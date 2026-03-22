from pathlib import Path

import yaml


DATA_ROOT_SUBDIR_KEYS = (
    "raw_data",
    "interim_data",
    "processed_data",
    "logs",
    "exports",
    "cache",
)


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


def get_storage_paths() -> dict[str, Path]:
    """Return resolved storage paths and validate the configured data root."""
    storage_config = get_storage_config()
    resolved = {
        key: resolve_storage_path(value)
        for key, value in storage_config.items()
    }

    data_root = resolved.get("data_root")
    if data_root is None:
        raise ValueError("config/storage.yaml is missing required key: data_root")

    return resolved


def require_data_root_exists() -> dict[str, Path]:
    """Fail fast when the configured GAIRA data root is not present."""
    storage_paths = get_storage_paths()
    data_root = storage_paths["data_root"]

    if not data_root.exists():
        raise FileNotFoundError(
            "Configured DATA_ROOT does not exist. "
            f"Expected: {data_root}. Create and mount the target volume first."
        )

    return storage_paths


def initialize_storage_root() -> dict[str, Path]:
    """Create the GAIRA storage layout under an existing mounted parent volume."""
    storage_paths = get_storage_paths()
    data_root = storage_paths["data_root"]
    parent_dir = data_root.parent

    if not parent_dir.exists():
        raise FileNotFoundError(
            "The configured external storage parent is not accessible. "
            f"Expected mount parent: {parent_dir}"
        )

    data_root.mkdir(parents=True, exist_ok=True)
    for key in DATA_ROOT_SUBDIR_KEYS:
        path = storage_paths.get(key)
        if path is not None:
            path.mkdir(parents=True, exist_ok=True)

    database_path = storage_paths.get("database")
    if database_path is not None:
        database_path.parent.mkdir(parents=True, exist_ok=True)

    return storage_paths


def get_database_path() -> Path:
    """Return the configured DuckDB path after verifying DATA_ROOT exists."""
    storage_paths = require_data_root_exists()
    database_path = storage_paths.get("database")
    if database_path is None:
        raise ValueError("config/storage.yaml is missing required key: database")
    return database_path


def ensure_storage_dirs() -> dict:
    """Verify the configured storage layout exists before data operations run."""
    storage_paths = require_data_root_exists()

    missing = [
        str(storage_paths[key])
        for key in DATA_ROOT_SUBDIR_KEYS
        if storage_paths.get(key) is not None and not storage_paths[key].exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Configured GAIRA storage subdirectories are missing. "
            "Run the storage initialization step first. Missing: "
            + ", ".join(missing)
        )

    return get_storage_config()
