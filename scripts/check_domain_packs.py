from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.domain_pack_registry import (
        find_packs_for_dataset,
        get_pack_default_embedding,
        list_domain_packs,
        list_shared_datasets,
    )

    pack_rows = list_domain_packs()
    pack_df = pd.DataFrame(pack_rows)
    if not pack_df.empty:
        pack_df["datasets"] = pack_df["datasets"].apply(lambda values: ", ".join(values))
        pack_df["intended_sample_types"] = pack_df["intended_sample_types"].apply(lambda values: ", ".join(values))
        pack_df["intended_matrices"] = pack_df["intended_matrices"].apply(lambda values: ", ".join(values))
        if "holdout_datasets" in pack_df.columns:
            pack_df["holdout_datasets"] = pack_df["holdout_datasets"].apply(
                lambda values: ", ".join(values) if isinstance(values, list) else ""
            )

    print("Registered domain packs:")
    print(pack_df.to_string(index=False))

    print()
    print("Pack defaults:")
    for pack_id in [row["pack_id"] for row in pack_rows]:
        print(f"  {pack_id}: {get_pack_default_embedding(pack_id)}")

    print()
    print("Dataset membership lookup:")
    for dataset_id in [
        "ramanbiolib",
        "serum_ag_colloids_grounding",
        "serum_ag_colloids_literature_grounding",
        "sers_fingerprint_workingpaper_support",
        "sers24_metabolite_support",
        "small2023_ev",
        "shine_ev_sers",
        "diabetes_plasma_ev_sers",
        "hcc_serum",
        "serum_ag_colloids",
        "serum_protocol_comparison",
        "cspp_serum",
    ]:
        print(f"  {dataset_id}: {', '.join(find_packs_for_dataset(dataset_id))}")

    shared_rows = list_shared_datasets()
    print()
    print("Shared datasets across packs:")
    if not shared_rows:
        print("  none")
    else:
        shared_df = pd.DataFrame(shared_rows)
        shared_df["pack_ids"] = shared_df["pack_ids"].apply(lambda values: ", ".join(values))
        print(shared_df.to_string(index=False))


if __name__ == "__main__":
    main()
