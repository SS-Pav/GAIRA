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

    report_path = (
        processed_root
        / "shine_class_reference_matches"
        / "shine_class_explanation_report.csv"
    )
    if not report_path.exists():
        print(f"Explanation report not found: {report_path}")
        return

    report_df = pd.read_csv(report_path)
    non_empty_chunk_count = int(
        report_df["supporting_chunk_1"].fillna("").str.strip().ne("").sum()
    )

    print(f"Explanation report rows: {len(report_df)}")
    print(f"Rows with non-empty supporting_chunk_1: {non_empty_chunk_count}")
    print(
        report_df[
            [
                "class_label",
                "subclass_label",
                "supporting_chunk_1",
                "supporting_chunk_2",
                "supporting_chunk_3",
                "explanation_text",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
