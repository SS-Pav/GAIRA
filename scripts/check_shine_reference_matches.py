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

    matches_dir = processed_root / "shine_class_reference_matches"
    if not matches_dir.exists():
        print(f"Match output folder not found: {matches_dir}")
        return

    match_files = sorted(matches_dir.glob("class_*_matches.csv"))
    if not match_files:
        print(f"No class match files were found in: {matches_dir}")
        return

    molecule_counter: dict[str, int] = {}
    class_counter: dict[str, int] = {}
    per_class_rows: list[dict] = []

    for match_file in match_files:
        match_df = pd.read_csv(match_file)
        if match_df.empty:
            continue

        top_component = str(match_df.iloc[0]["component"])
        top_class = str(match_df.iloc[0]["biochemical_class"])
        molecule_counter[top_component] = molecule_counter.get(top_component, 0) + 1
        class_counter[top_class] = class_counter.get(top_class, 0) + 1

        per_class_rows.append(
            {
                "file": match_file.name,
                "class_label": match_df.iloc[0].get("class_label"),
                "subclass_label": match_df.iloc[0].get("subclass_label"),
                "top_component": top_component,
                "top_biochemical_class": top_class,
            }
        )

    top_molecules_df = (
        pd.DataFrame(
            [{"component": key, "count": value} for key, value in molecule_counter.items()]
        )
        .sort_values(["count", "component"], ascending=[False, True])
        .reset_index(drop=True)
    )
    top_classes_df = (
        pd.DataFrame(
            [{"biochemical_class": key, "count": value} for key, value in class_counter.items()]
        )
        .sort_values(["count", "biochemical_class"], ascending=[False, True])
        .reset_index(drop=True)
    )
    per_class_df = pd.DataFrame(per_class_rows)

    print(f"Number of classes processed: {len(match_files)}")
    print("\nTop molecule frequencies across classes:")
    print(top_molecules_df.head(10).to_string(index=False))
    print("\nTop biochemical classes across classes:")
    print(top_classes_df.head(10).to_string(index=False))
    print("\nPer-class top hits:")
    print(per_class_df.to_string(index=False))


if __name__ == "__main__":
    main()
