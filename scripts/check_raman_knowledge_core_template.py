from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    template_path = project_root / "docs" / "raman_knowledge_core_template.md"
    dataset_root = Path("/Volumes/SSD_SPG/GAIRA_DATA/raw/raman_knowledge_core")
    expected_files = [
        "sources.csv",
        "peak_assignments.csv",
        "biomarker_claims.csv",
        "confounder_notes.csv",
        "knowledge_chunks.csv (optional)",
        "semantic_regions.csv (recommended)",
        "dataset_context.csv (recommended)",
    ]

    print(f"Template file present: {template_path.exists()} -> {template_path}")
    print(f"Expected local raw folder: {dataset_root}")
    print("Expected knowledge package files:")
    for file_name in expected_files:
        print(f"  - {file_name}")

    if dataset_root.exists():
        print("Current files found in the raw folder:")
        for path in sorted(dataset_root.glob("*.csv")):
            print(f"  - {path.name}")
    else:
        print("Raw knowledge folder is not present yet.")


if __name__ == "__main__":
    main()
