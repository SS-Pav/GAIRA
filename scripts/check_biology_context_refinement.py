from pathlib import Path
import sys

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_storage_paths, require_data_root_exists

    storage_paths = require_data_root_exists()
    base_dir = storage_paths["processed_data"] / "biology_context_refinement"

    required_paths = [
        base_dir / "report" / "biology_context_refinement_report.md",
        base_dir / "report" / "biology_context_refinement_report.pdf",
        base_dir / "raw_outputs" / "theme_before_after_outputs.csv",
        base_dir / "raw_outputs" / "query_before_after_summary.csv",
        base_dir / "tables" / "ev_dataset_understanding_map.csv",
        storage_paths["processed_data"] / "context_extraction" / "diabetes_ev_structured_notes.md",
        storage_paths["processed_data"] / "context_extraction" / "spectra_shine_structured_notes.md",
    ]
    for path in required_paths:
        print(f"{path.name}: {path.exists()}")

    theme_path = base_dir / "raw_outputs" / "theme_before_after_outputs.csv"
    if not theme_path.exists():
        return

    theme_df = pd.read_csv(theme_path)
    print(f"Theme rows: {len(theme_df)}")
    print("Versions:")
    print(theme_df["version"].value_counts().to_string())
    print("Query groups:")
    print(theme_df["query_group"].value_counts().to_string())


if __name__ == "__main__":
    main()
