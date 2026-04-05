from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gaira.embedding.branch_metadata import EV_STRESS_DATASETS, SMALL2023_DATASETS, branch_label_arrays


def branch_mask(
    dataset_ids: np.ndarray,
    sample_types: np.ndarray,
    *,
    branch_mode: str,
) -> np.ndarray:
    dataset_ids = dataset_ids.astype(str)
    sample_types = sample_types.astype(str)
    if branch_mode == "none":
        return np.ones(len(dataset_ids), dtype=bool)
    if branch_mode == "ev_stress":
        return (sample_types == "ev") & np.isin(dataset_ids, list(EV_STRESS_DATASETS))
    if branch_mode in {"small2023_specialized", "small2023_cellline", "small2023_mixture"}:
        return (sample_types == "ev") & np.isin(dataset_ids, list(SMALL2023_DATASETS))
    raise ValueError(f"Unsupported branch_mode: {branch_mode}")


def filtered_dataset_dict(dataset: np.lib.npyio.NpzFile, *, branch_mode: str) -> dict[str, np.ndarray]:
    sample_keys = dataset["sample_keys"].astype(str) if "sample_keys" in dataset.files else np.asarray([str(i) for i in range(len(dataset["X"]))], dtype=object)
    dataset_ids = dataset["dataset_ids"].astype(str)
    sample_types = dataset["sample_types"].astype(str)
    labels_optional = dataset["labels_optional"].astype(str) if "labels_optional" in dataset.files else np.asarray([""] * len(dataset_ids), dtype=object)
    subclasses = dataset["subclasses"].astype(str) if "subclasses" in dataset.files else np.asarray([""] * len(dataset_ids), dtype=object)

    keep_mask = branch_mask(dataset_ids, sample_types, branch_mode=branch_mode)
    branch_arrays = branch_label_arrays(sample_keys, dataset_ids, labels_optional, subclasses, branch_mode=branch_mode)
    if branch_mode == "ev_stress":
        keep_mask = keep_mask & (branch_arrays["branch_state_label"].astype(str) != "")
    if branch_mode in {"small2023_specialized", "small2023_cellline", "small2023_mixture"}:
        keep_mask = keep_mask & (branch_arrays["branch_primary_label"].astype(str) != "")

    result: dict[str, np.ndarray] = {}
    n_samples = len(dataset["X"])
    for key in dataset.files:
        value = dataset[key]
        if getattr(value, "shape", None) is not None and len(value.shape) > 0 and value.shape[0] == n_samples:
            result[key] = value[keep_mask]
        else:
            result[key] = value
    for key, value in branch_arrays.items():
        result[key] = value[keep_mask]
    result["branch_mode"] = np.asarray([branch_mode] * int(keep_mask.sum()), dtype=object)
    return result


def write_filtered_dataset(filtered: dict[str, np.ndarray], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **filtered)


def branch_sample_manifest(filtered: dict[str, np.ndarray]) -> pd.DataFrame:
    n = len(filtered["X"])
    frame = pd.DataFrame(
        {
            "sample_key": filtered.get("sample_keys", np.asarray([str(i) for i in range(n)], dtype=object)).astype(str),
            "dataset_id": filtered["dataset_ids"].astype(str),
            "sample_type": filtered["sample_types"].astype(str),
            "label_optional": filtered.get("labels_optional", np.asarray([""] * n, dtype=object)).astype(str),
            "subclass_label": filtered.get("subclasses", np.asarray([""] * n, dtype=object)).astype(str),
            "branch_mode": filtered.get("branch_mode", np.asarray(["none"] * n, dtype=object)).astype(str),
            "branch_primary_label": filtered.get("branch_primary_label", np.asarray([""] * n, dtype=object)).astype(str),
            "branch_secondary_label": filtered.get("branch_secondary_label", np.asarray([""] * n, dtype=object)).astype(str),
            "branch_state_label": filtered.get("branch_state_label", np.asarray([""] * n, dtype=object)).astype(str),
            "branch_label_weight": filtered.get("branch_label_weight", np.asarray([0.0] * n, dtype=np.float32)).astype(float),
        }
    )
    return frame


def branch_dataset_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_id, group in manifest.groupby("dataset_id", sort=True):
        primary = group.loc[group["branch_primary_label"].astype(str) != "", "branch_primary_label"]
        secondary = group.loc[group["branch_secondary_label"].astype(str) != "", "branch_secondary_label"]
        state = group.loc[group["branch_state_label"].astype(str) != "", "branch_state_label"]
        rows.append(
            {
                "dataset_id": dataset_id,
                "n_samples": int(len(group)),
                "sample_type": str(group["sample_type"].mode().iloc[0]),
                "branch_primary_labels": int(primary.nunique()),
                "branch_primary_label_values": "|".join(sorted(primary.astype(str).unique().tolist())) if not primary.empty else "",
                "branch_secondary_labels": int(secondary.nunique()),
                "branch_secondary_label_values": "|".join(sorted(secondary.astype(str).unique().tolist())) if not secondary.empty else "",
                "branch_state_labels": int(state.nunique()),
                "branch_state_label_values": "|".join(sorted(state.astype(str).unique().tolist())) if not state.empty else "",
            }
        )
    return pd.DataFrame(rows)
