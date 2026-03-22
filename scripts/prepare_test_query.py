import shutil
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    source_file = (
        Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/ramanbiolib")
        / "ramanbiolib-main"
        / "examples"
        / "search"
        / "collagen_example.csv"
    )
    target_dir = project_root / "data" / "processed" / "test_queries"
    target_file = target_dir / "collagen_example_query.csv"

    if not source_file.exists():
        print(f"Source query file not found: {source_file}")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_file, target_file)
    print(f"Copied test query file to: {target_file}")


if __name__ == "__main__":
    main()
