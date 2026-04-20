"""Calibration dataset loaders — read raw spectra directly from disk.

Each loader returns (X, wavenumbers, cohorts, meta_df) for one calibration
dataset. Loaders are kept independent of the GAIRA ingestion / DuckDB state
so calibration evaluation can run on fresh checkouts.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


DATA_ROOT = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw")


@dataclass
class CalibrationRaw:
    """Raw spectra + cohort labels for a calibration dataset."""
    loader_id: str
    X: np.ndarray            # (n_spectra, n_wavenumbers) raw intensities
    wavenumbers: np.ndarray   # (n_wavenumbers,)
    cohorts: np.ndarray       # (n_spectra,) string labels matching registry cohorts
    meta: pd.DataFrame        # one row per spectrum, at minimum: cohort, sample_id, replicate_id
    source: str               # human-readable source path(s)


# ─────────────────────────────────────────────────────────────────────
# CSPP Figure 7 — metabolite spiking
# ─────────────────────────────────────────────────────────────────────

_CSPP_META_COLS = {
    "Unnamed: 0", "num", "method", "serum_typ", "metabolite",
    "conc", "acc", "t_mes", "pw", "rep",
}


def _load_cspp_fig7() -> CalibrationRaw:
    path = DATA_ROOT / "cspp_serum" / "Figure-7_all-spectra-and-metadata.csv"
    df = pd.read_csv(path)
    wn_cols = [c for c in df.columns if c not in _CSPP_META_COLS]
    wn = np.asarray([float(c) for c in wn_cols], dtype=float)
    X = df[wn_cols].to_numpy(dtype=float)

    cohorts = df["metabolite"].astype(str).to_numpy()
    meta = df[[c for c in df.columns if c in _CSPP_META_COLS]].copy()
    meta = meta.rename(columns={"metabolite": "cohort"})
    meta["sample_id"] = df["num"].astype(str).values
    meta["replicate_id"] = df["rep"].astype(str).values

    return CalibrationRaw(
        loader_id="cspp_fig7",
        X=X, wavenumbers=wn, cohorts=cohorts,
        meta=meta, source=str(path),
    )


# ─────────────────────────────────────────────────────────────────────
# Serum Ag colloids — uricase depletion experiment
# ─────────────────────────────────────────────────────────────────────

_URICASE_FOLDER = "dataset uricase"


def _read_bwtek_txt_bytes(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Read a BWtek-format txt file body and return (wavenumbers, intensities).

    Matches the ingestion parser: skiprows=90, sep=';', decimal=',',
    wavenumbers in column 3, intensities in column 7 (raw).
    """
    df = pd.read_csv(
        io.BytesIO(raw), sep=";", decimal=",", skiprows=90, header=None,
    )
    if df.shape[1] < 8:
        raise ValueError("BWtek file has fewer columns than expected.")
    wn = df.iloc[:, 3].to_numpy(dtype=float)
    it = df.iloc[:, 7].to_numpy(dtype=float)
    return wn, it


def _load_serum_ag_uricase() -> CalibrationRaw:
    path = DATA_ROOT / "serum_ag_colloids" / "dataset_spectral_data.zip"
    records = []
    ref_wn: np.ndarray | None = None

    with ZipFile(path, "r") as z:
        members = sorted(
            m for m in z.namelist()
            if m.startswith(_URICASE_FOLDER + "/") and m.lower().endswith(".txt")
        )
        for m in members:
            raw = z.read(m)
            wn, it = _read_bwtek_txt_bytes(raw)
            if ref_wn is None:
                ref_wn = wn
            elif not np.allclose(ref_wn, wn, atol=0.05):
                raise ValueError(f"Non-matching axis in {m}")

            # Filename: NNN_RR_<cohort>_ProtN.txt (e.g. 006_01_Serumspiked+Enzyme_Prot1.txt)
            stem = Path(m).stem
            parts = stem.split("_")
            if len(parts) < 4:
                continue
            spec_num, rep, cohort, protocol = parts[:4]
            records.append({
                "intensity": it,
                "cohort": cohort,
                "sample_id": spec_num,
                "replicate_id": rep,
                "protocol": protocol,
                "source_file": m,
            })

    if not records:
        raise ValueError(f"No uricase spectra found in {path}")

    X = np.stack([r["intensity"] for r in records])
    cohorts = np.array([r["cohort"] for r in records])
    meta = pd.DataFrame([{k: v for k, v in r.items() if k != "intensity"} for r in records])

    return CalibrationRaw(
        loader_id="serum_ag_uricase",
        X=X, wavenumbers=ref_wn, cohorts=cohorts,
        meta=meta, source=f"{path}::{_URICASE_FOLDER}/",
    )


# ─────────────────────────────────────────────────────────────────────
# Ergothioneine titration — concentration series in serum
# ─────────────────────────────────────────────────────────────────────

_ERG_META_COLS = {"laser", "power", "substrate", "c"}


def _load_ergothioneine_titration() -> CalibrationRaw:
    path = DATA_ROOT / "ergothioneine_serum" / "ERG_calibration.csv"
    df = pd.read_csv(path)
    wn_cols = [c for c in df.columns if c not in _ERG_META_COLS]
    wn = np.asarray([float(c) for c in wn_cols], dtype=float)
    X = df[wn_cols].to_numpy(dtype=float)

    # Cohort label matches parser convention: erg_{conc}_uM with dots → 'p'
    def _cohort_for(c_val: float) -> str:
        return f"erg_{c_val:.1f}_uM".replace(".", "p")

    cohorts = np.array([_cohort_for(float(c)) for c in df["c"]])
    meta = df[list(_ERG_META_COLS)].copy()
    meta["cohort"] = cohorts
    # stable replicate index per concentration
    rep = []
    counter: dict[str, int] = {}
    for c in cohorts:
        counter[c] = counter.get(c, 0) + 1
        rep.append(f"rep{counter[c]}")
    meta["replicate_id"] = rep
    meta["sample_id"] = meta["cohort"]

    return CalibrationRaw(
        loader_id="ergothioneine_titration",
        X=X, wavenumbers=wn, cohorts=cohorts,
        meta=meta, source=str(path),
    )


# ─────────────────────────────────────────────────────────────────────
# Public dispatch
# ─────────────────────────────────────────────────────────────────────

_LOADERS = {
    "cspp_fig7": _load_cspp_fig7,
    "serum_ag_uricase": _load_serum_ag_uricase,
    "ergothioneine_titration": _load_ergothioneine_titration,
}


def load_calibration_raw(loader_id: str) -> CalibrationRaw:
    if loader_id not in _LOADERS:
        raise KeyError(
            f"Unknown calibration loader '{loader_id}'. Available: {sorted(_LOADERS)}"
        )
    return _LOADERS[loader_id]()


def list_loaders() -> list[str]:
    return sorted(_LOADERS)
