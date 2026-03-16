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

    interpreted_path = (
        processed_root
        / "shine_class_reference_matches"
        / "shine_class_consensus_interpreted.csv"
    )
    if not interpreted_path.exists():
        print(f"Interpreted SHINE consensus file not found: {interpreted_path}")
        return

    interpreted_df = pd.read_csv(interpreted_path)
    print(f"Interpreted SHINE consensus rows: {len(interpreted_df)}")
    print(
        interpreted_df[
            [
                "class_label",
                "subclass_label",
                "region_semantic_label_1",
                "region_semantic_label_2",
                "knowledge_supported_groups",
                "cautious_interpretation",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
