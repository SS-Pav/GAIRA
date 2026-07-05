"""
Spectral dataset loader — loads spectra and cohort labels from NPZ or CSV.

Axes-phase loaders (HCC holdout, Pilot 2b/3 DuckDB rows, NPZ-ingested
pools) are grandfathered on their current ``numpy.interp`` path: changing
them would break byte-for-byte reproducibility of the frozen
``gaira_build_axes_v1`` outputs, which the project policy forbids.

**Forward-looking rule (M3+):** new motif-layer code paths must use the
canonical helpers exposed by :mod:`gaira.spectral.crop_and_interp`:

    from gaira.spectral import crop_before_interpolate, crop_and_interp_batch

These enforce crop-first, deterministic, no-extrapolation behaviour and
surface partial-coverage explicitly. See
``canonical_signal_support_rule_v1`` doc for the rule and
``pipeline_fix_crop_before_interp_v1`` report for the migration plan.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gaira.spectral.dataset_registry import NPZ_PATH


HCC_HOLDOUT_CSV = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/hcc_serum/data.csv")

# Cohort label mapping for HCC holdout
_HCC_HOLDOUT_LABELS = {"H0T": "hcc", "CTR": "healthy_control"}


@dataclass
class SpectralDataset:
    """Loaded spectral dataset with metadata."""
    dataset_id: str
    X: np.ndarray            # (n_spectra, n_wavenumbers)
    wavenumbers: np.ndarray   # (n_wavenumbers,)
    cohorts: np.ndarray       # (n_spectra,) string labels
    n_spectra: int
    cohort_names: list[str]
    cohort_counts: dict[str, int]


def _load_hcc_holdout(master_x: np.ndarray) -> SpectralDataset:
    """Load HCC holdout (Vornoli 2020) from raw CSV, interpolated to master axis.

    GRANDFATHERED axes-phase loader. The np.interp clamp-at-boundaries
    behaviour here is preserved byte-for-byte so that
    ``gaira_build_axes_v1`` (Pilot 1) remains reproducible. M3+ motif-
    layer code must not depend on this loader's output; use
    :func:`gaira.spectral.crop_before_interpolate` instead.
    """
    import pandas as pd

    df = pd.read_csv(HCC_HOLDOUT_CSV)
    meta_cols = ["acquisition_date", "substrate_batch", "class", "sample_code"]
    wn_cols = [c for c in df.columns if c not in meta_cols]

    raw_wn = np.array([float(c) for c in wn_cols])
    raw_X = df[wn_cols].values.astype(np.float64)

    # Filter to fingerprint region and interpolate to master wavenumber axis
    fp_mask = (raw_wn >= master_x.min()) & (raw_wn <= master_x.max())
    raw_wn_fp = raw_wn[fp_mask]
    raw_X_fp = raw_X[:, fp_mask]

    # Interpolate each spectrum to the master axis
    X_interp = np.zeros((raw_X_fp.shape[0], len(master_x)))
    for i in range(raw_X_fp.shape[0]):
        X_interp[i] = np.interp(master_x, raw_wn_fp, raw_X_fp[i])

    # Map cohort labels
    cohorts = np.array([_HCC_HOLDOUT_LABELS.get(c, c) for c in df["class"]])
    cohort_names = sorted(set(cohorts))
    cohort_counts = {c: int((cohorts == c).sum()) for c in cohort_names}

    return SpectralDataset(
        dataset_id="hcc_holdout_vornoli2020",
        X=X_interp,
        wavenumbers=master_x,
        cohorts=cohorts,
        n_spectra=len(X_interp),
        cohort_names=cohort_names,
        cohort_counts=cohort_counts,
    )


def load_dataset(dataset_id: str) -> SpectralDataset:
    """Load a spectral dataset from the embedding NPZ or CSV source."""
    if dataset_id == "hcc_holdout_vornoli2020":
        npz = np.load(NPZ_PATH, allow_pickle=True)
        master_x = npz["master_x"].astype(np.float64)
        return _load_hcc_holdout(master_x)

    npz = np.load(NPZ_PATH, allow_pickle=True)
    mask = npz["dataset_ids"] == dataset_id
    X = npz["X"][mask].astype(np.float64)
    wavenumbers = npz["master_x"].astype(np.float64)
    cohorts = np.array([s.split("::")[-1] for s in npz["semantic_groups"][mask]])

    cohort_names = sorted(set(cohorts))
    cohort_counts = {c: int((cohorts == c).sum()) for c in cohort_names}

    return SpectralDataset(
        dataset_id=dataset_id,
        X=X,
        wavenumbers=wavenumbers,
        cohorts=cohorts,
        n_spectra=len(X),
        cohort_names=cohort_names,
        cohort_counts=cohort_counts,
    )
