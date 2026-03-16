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

    report_path = processed_root / "shine_class_reference_matches" / "shine_class_explanation_report_v6.csv"
    if not report_path.exists():
        print(f"Explanation report v6 not found: {report_path}")
        return

    report_df = pd.read_csv(report_path)
    non_empty_context_rows = int(
        (
            report_df["context_modality"].fillna("").str.strip().ne("")
            & report_df["context_sample_type"].fillna("").str.strip().ne("")
            & report_df["context_enhancement_mode"].fillna("").str.strip().ne("")
        ).sum()
    )

    print(f"Explanation report v6 rows: {len(report_df)}")
    print(f"Rows with non-empty core context fields: {non_empty_context_rows}")
    print("\nFirst 10 rows:")
    print(
        report_df[
            [
                "class_label",
                "subclass_label",
                "context_modality",
                "context_sample_type",
                "context_region_caution_1",
                "context_region_caution_2",
                "confidence_tier",
                "explanation_text",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
