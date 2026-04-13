"""
Spectral dataset registry — explicit target demo datasets for GAIRA spectral query.

Only datasets with validated cohort structure and scientific relevance
are exposed in the demo selector.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


NPZ_PATH = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_v5_full/embedding_dataset.npz")
HCC_HOLDOUT_CSV = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/hcc_serum/data.csv")

# Explicit target demo datasets — only these appear in the selector
TARGET_DATASETS = [
    {
        "id": "hcc_holdout_vornoli2020",
        "display": "HCC Holdout — Serum SERS (Vornoli 2020, Au)",
        "source": "csv",
        "family": "serum_liver_hepatobiliary",
        "note": "72 HCC + 72 healthy. Gold nanoparticle substrate.",
    },
    {
        "id": "cca_hcc_lm_serum_sers",
        "display": "CCA / HCC / LM — Serum SERS (Lin et al., AgNP)",
        "source": "npz",
        "family": "serum_liver_hepatobiliary",
        "note": "96 CCA + 89 HCC + 81 LM + 88 healthy. Silver nanoparticle substrate.",
    },
    {
        "id": "diabetes_plasma_ev_sers",
        "display": "Diabetes Plasma EV SERS (BMI stratified)",
        "source": "npz",
        "family": "ev_disease_or_stress",
        "note": "222 BMI>25 + 130 BMI≤25. EV SERS from diabetic patients.",
    },
]


@dataclass
class DatasetInfo:
    dataset_id: str
    display_name: str
    n_spectra: int
    cohorts: dict[str, int]
    family: str = ""
    note: str = ""


def discover_datasets() -> list[DatasetInfo]:
    """Return target demo datasets that are available on disk."""
    results = []

    for td in TARGET_DATASETS:
        did = td["id"]

        if did == "hcc_holdout_vornoli2020":
            if HCC_HOLDOUT_CSV.exists():
                results.append(DatasetInfo(
                    dataset_id=did, display_name=td["display"],
                    n_spectra=144, cohorts={"hcc": 72, "healthy_control": 72},
                    family=td["family"], note=td["note"],
                ))
            continue

        # NPZ-based
        if not NPZ_PATH.exists():
            continue

        npz = np.load(NPZ_PATH, allow_pickle=True)
        mask = npz["dataset_ids"] == did
        if mask.sum() == 0:
            continue

        n = int(mask.sum())
        conds = [s.split("::")[-1] for s in npz["semantic_groups"][mask]]
        cohort_counts = {k: v for k, v in Counter(conds).items() if k}

        if len(cohort_counts) < 2:
            continue

        results.append(DatasetInfo(
            dataset_id=did, display_name=td["display"],
            n_spectra=n, cohorts=cohort_counts,
            family=td["family"], note=td["note"],
        ))

    return results
