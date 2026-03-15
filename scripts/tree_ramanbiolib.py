import sys
from pathlib import Path


def print_tree(root: Path, max_depth: int = 3, current_depth: int = 0) -> None:
    """Print a compact directory tree up to a limited depth."""
    if current_depth > max_depth or not root.exists():
        return

    entries = sorted(root.iterdir(), key=lambda path: (path.is_file(), path.name.lower()))
    for entry in entries:
        indent = "  " * current_depth
        marker = "[D]" if entry.is_dir() else "[F]"
        print(f"{indent}{marker} {entry.name}")
        if entry.is_dir():
            print_tree(entry, max_depth=max_depth, current_depth=current_depth + 1)


def main() -> None:
    # Make the src package importable when running from the project root.
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_storage_config, resolve_storage_path

    storage_config = get_storage_config()
    raw_data_path = resolve_storage_path(storage_config.get("raw_data"))

    if raw_data_path is None:
        print("The storage config is missing the 'raw_data' path.")
        return

    dataset_root = raw_data_path / "ramanbiolib"
    print(f"RamanBioLib folder: {dataset_root}")

    if not dataset_root.exists():
        print("The RamanBioLib folder does not exist yet. Run the download script first.")
        return

    print("Compact tree view (depth 3):")
    print_tree(dataset_root, max_depth=3)


if __name__ == "__main__":
    main()
