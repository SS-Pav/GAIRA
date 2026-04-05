from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


LOCAL_ANCHOR_TABLE_PATH = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_anchor_audit/embedding_anchor_table_v1.csv")
REMOTE_ANCHOR_TABLE_PATH = Path.home() / "projects" / "GAIRA" / "data" / "processed" / "embedding_anchor_audit" / "embedding_anchor_table_v1.csv"


def resolve_anchor_table_path(path: str | Path | None = None) -> Path:
    if path:
        return Path(path).expanduser().resolve()
    env_value = os.environ.get("GAIRAM_ANCHOR_TABLE_PATH")
    if env_value:
        return Path(env_value).expanduser().resolve()
    if LOCAL_ANCHOR_TABLE_PATH.exists():
        return LOCAL_ANCHOR_TABLE_PATH.resolve()
    return REMOTE_ANCHOR_TABLE_PATH.resolve()


def load_anchor_table(path: str | Path | None = None) -> pd.DataFrame:
    anchor_path = resolve_anchor_table_path(path)
    return pd.read_csv(anchor_path)


def anchor_map_by_sample_key(path: str | Path | None = None) -> dict[str, dict[str, object]]:
    df = load_anchor_table(path)
    return {
        str(row["sample_key"]): {
            "sample_type": row["sample_type"],
            "dataset_id": row["dataset_id"],
            "proposed_harmonized_anchor": row["proposed_harmonized_anchor"],
            "anchor_type": row["anchor_type"],
            "anchor_confidence": row["anchor_confidence"],
            "cross_dataset_usable": bool(row["cross_dataset_usable"]),
            "notes": row["notes"],
        }
        for row in df.to_dict(orient="records")
    }


def aligned_anchor_arrays(
    sample_keys: np.ndarray | list[str],
    path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    anchor_map = anchor_map_by_sample_key(path)

    harmonized_anchor = []
    anchor_confidence = []
    cross_dataset_usable = []
    anchor_type = []
    for sample_key in map(str, sample_keys):
        row = anchor_map.get(sample_key, {})
        harmonized_anchor.append(str(row.get("proposed_harmonized_anchor", "")))
        anchor_confidence.append(str(row.get("anchor_confidence", "")))
        cross_dataset_usable.append(bool(row.get("cross_dataset_usable", False)))
        anchor_type.append(str(row.get("anchor_type", "")))

    return {
        "harmonized_anchor": np.asarray(harmonized_anchor, dtype=object),
        "anchor_confidence": np.asarray(anchor_confidence, dtype=object),
        "cross_dataset_usable": np.asarray(cross_dataset_usable, dtype=bool),
        "anchor_type": np.asarray(anchor_type, dtype=object),
    }
