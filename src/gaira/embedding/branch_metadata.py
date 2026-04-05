from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


EV_STRESS_DATASETS = {"shine_ev_sers", "diabetes_plasma_ev_sers"}
SMALL2023_DATASETS = {"small2023_ev"}
SMALL2023_CELLLINE_LABELS = {"Hec", "Hela", "Ht", "Mef", "Thp"}
SMALL2023_MIXTURE_LABELS = {"c00", "c01", "c10", "c25", "c50", "c100"}
LOCAL_EV_STATE_TABLE = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/v8_ev_stress_prep/ev_state_mapping_table.csv")
LEGACY_EV_STATE_TABLE = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/v8_ev_stress_analysis_v1/harmonized_state_table.csv")


def resolve_ev_state_table_path() -> Path | None:
    if LOCAL_EV_STATE_TABLE.exists():
        return LOCAL_EV_STATE_TABLE
    if LEGACY_EV_STATE_TABLE.exists():
        return LEGACY_EV_STATE_TABLE
    return None


def _shine_fallback(sample_key: str, label: str) -> str:
    text = " ".join([sample_key, label]).lower()
    if re.search(r"\bd0\b", text) or re.search(r"(low|0u?m|1u?m|5u?m|10u?m)", text):
        return "control_like"
    if re.search(r"\bd[2-9]\b", text) or re.search(r"(20u?m|40u?m|80u?m|high|toxic|stress)", text):
        return "stress_or_toxicity_like"
    return "intermediate_or_ambiguous"


def fallback_ev_state(sample_key: str, dataset_id: str, label_optional: str) -> str:
    label = str(label_optional or "")
    if dataset_id == "diabetes_plasma_ev_sers":
        if label == "Strong-D":
            return "control_like"
        if label == "Impact":
            return "stress_or_toxicity_like"
    if dataset_id == "shine_ev_sers":
        return _shine_fallback(str(sample_key), label)
    return ""


def load_ev_state_map() -> dict[str, str]:
    path = resolve_ev_state_table_path()
    if path is None:
        return {}
    df = pd.read_csv(path)
    if "sample_key" not in df.columns:
        return {}
    state_col = "ev_state" if "ev_state" in df.columns else "harmonized_state"
    if state_col not in df.columns:
        return {}
    return {
        str(row["sample_key"]): str(row[state_col]).replace("low_metabolic_stress", "control_like").replace("high_metabolic_stress", "stress_or_toxicity_like")
        for row in df.to_dict(orient="records")
    }


def branch_label_arrays(
    sample_keys: np.ndarray,
    dataset_ids: np.ndarray,
    label_optional: np.ndarray,
    subclass_labels: np.ndarray,
    *,
    branch_mode: str,
) -> dict[str, np.ndarray]:
    sample_keys = sample_keys.astype(str)
    dataset_ids = dataset_ids.astype(str)
    label_optional = label_optional.astype(str)
    subclass_labels = subclass_labels.astype(str)
    primary = np.asarray([""] * len(sample_keys), dtype=object)
    secondary = np.asarray([""] * len(sample_keys), dtype=object)
    state = np.asarray([""] * len(sample_keys), dtype=object)
    weights = np.asarray([0.0] * len(sample_keys), dtype=np.float32)

    if branch_mode == "ev_stress":
        ev_state_map = load_ev_state_map()
        for i, (sample_key, dataset_id, label) in enumerate(zip(sample_keys, dataset_ids, label_optional, strict=False)):
            mapped = ev_state_map.get(sample_key) or fallback_ev_state(sample_key, dataset_id, label)
            state[i] = mapped
            primary[i] = mapped
            if mapped == "intermediate_or_ambiguous":
                weights[i] = 0.35
            elif mapped:
                weights[i] = 1.0
    elif branch_mode == "small2023_specialized":
        for i, (dataset_id, label, subclass) in enumerate(zip(dataset_ids, label_optional, subclass_labels, strict=False)):
            if dataset_id == "small2023_ev":
                primary[i] = label
                secondary[i] = subclass
                weights[i] = 1.0 if label else 0.0
    elif branch_mode == "small2023_cellline":
        for i, (dataset_id, label, subclass) in enumerate(zip(dataset_ids, label_optional, subclass_labels, strict=False)):
            if dataset_id == "small2023_ev" and label in SMALL2023_CELLLINE_LABELS:
                primary[i] = label
                secondary[i] = subclass
                weights[i] = 1.0
    elif branch_mode == "small2023_mixture":
        for i, (dataset_id, label, subclass) in enumerate(zip(dataset_ids, label_optional, subclass_labels, strict=False)):
            if dataset_id == "small2023_ev" and label in SMALL2023_MIXTURE_LABELS:
                primary[i] = label
                secondary[i] = subclass
                weights[i] = 1.0

    return {
        "branch_primary_label": primary,
        "branch_secondary_label": secondary,
        "branch_state_label": state,
        "branch_label_weight": weights,
    }
