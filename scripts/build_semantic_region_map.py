import json
import sys
from pathlib import Path

import duckdb
import pandas as pd


DATASET_ID = "raman_knowledge_core"
REGION_GAP_CM = 25.0


def build_region_label(region_df: pd.DataFrame) -> str:
    """Choose a cautious region label from the dominant assigned group."""
    top_group_df = (
        region_df.groupby("assigned_group", dropna=False)["source_id"]
        .nunique()
        .reset_index(name="source_count")
        .sort_values(["source_count", "assigned_group"], ascending=[False, True])
        .reset_index(drop=True)
    )
    if top_group_df.empty:
        return "mixed_region"

    top_group = str(top_group_df.iloc[0]["assigned_group"]).strip()
    if top_group == "" or top_group.lower() == "nan":
        return "mixed_region"

    if len(top_group_df) > 1 and int(top_group_df.iloc[0]["source_count"]) == int(top_group_df.iloc[1]["source_count"]):
        return "mixed_region"

    return f"{top_group}_enriched_region"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import ensure_storage_dirs, resolve_storage_path

    storage_config = ensure_storage_dirs()
    processed_root = resolve_storage_path(storage_config.get("processed_data"))
    db_path = project_root / "data" / "gaira.duckdb"

    if processed_root is None:
        print("The storage config is missing processed_data.")
        return

    output_dir = processed_root / "knowledge"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "semantic_region_map.csv"

    with duckdb.connect(str(db_path), read_only=True) as connection:
        assignments_df = connection.execute(
            """
            SELECT source_id, peak_cm, assigned_molecule, assigned_group, evidence_text
            FROM peak_assignments
            WHERE dataset_id = ?
            ORDER BY peak_cm
            """,
            [DATASET_ID],
        ).fetchdf()

    if assignments_df.empty:
        print("No peak_assignments rows were found for raman_knowledge_core.")
        print("Ingest the local knowledge package first.")
        return

    assignments_df["peak_cm"] = pd.to_numeric(assignments_df["peak_cm"], errors="coerce")
    assignments_df = assignments_df.dropna(subset=["peak_cm"]).sort_values("peak_cm").reset_index(drop=True)

    region_ids: list[int] = []
    current_region_id = 0
    previous_peak = None

    for peak_cm in assignments_df["peak_cm"]:
        if previous_peak is None or float(peak_cm) - float(previous_peak) > REGION_GAP_CM:
            current_region_id += 1
        region_ids.append(current_region_id)
        previous_peak = peak_cm

    assignments_df["region_id"] = region_ids

    region_rows: list[dict] = []
    for region_id, region_df in assignments_df.groupby("region_id", sort=True):
        assigned_groups = sorted({str(value) for value in region_df["assigned_group"].dropna().tolist() if str(value).strip()})
        example_molecules = sorted({str(value) for value in region_df["assigned_molecule"].dropna().tolist() if str(value).strip()})[:5]
        notes = sorted({str(value) for value in region_df["evidence_text"].dropna().tolist() if str(value).strip()})[:3]

        region_rows.append(
            {
                "region_id": f"region_{int(region_id):03d}",
                "region_min_cm": float(region_df["peak_cm"].min()),
                "region_max_cm": float(region_df["peak_cm"].max()),
                "region_label": build_region_label(region_df),
                "assigned_groups": json.dumps(assigned_groups),
                "example_molecules": json.dumps(example_molecules),
                "source_count": int(region_df["source_id"].nunique()),
                "notes": " | ".join(notes),
            }
        )

    region_map_df = pd.DataFrame(region_rows)
    region_map_df.to_csv(output_path, index=False)

    print(f"Semantic region map written to: {output_path}")
    print(f"Semantic regions created: {len(region_map_df)}")
    print(region_map_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
