"""GAIRA V5 Phase 2 Stage A — canonical representation input (785 nm grounding).

Builds the audited Phase-2 input: reuses src/gaira/data (loaders) and
src/gaira/preprocessing (pipelines). Applies the Phase-2 role corrections:
  * EXCLUDE adenine_sers_control (6-point concentration series = controlled
    perturbation evaluation, NOT independent grounding).
  * EXCLUDE non-785, metabolite-63 (633 nm), ORC-Ag peak-only.
Adenine remains grounded via Gobbato (Raman + Ag-SERS).

No cross-modality averaging happens here. Deterministic, read-only.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..data import loader, gobbato
from ..data.synonyms import canonical
from ..preprocessing import pipeline as pp

GRID = pp.common_grid(520.0, 1750.0, 2.0)

# preprocessing candidates for Stage A
PREPROCS = {
    "A1_asls_savgol_l2":  dict(baseline="asls", smooth="savgol", norm="l2"),
    "A2_asls_savgol_snv": dict(baseline="asls", smooth="savgol", norm="snv"),
    "A3_deriv_l2":        dict(baseline="asls", smooth="savgol", norm="l2", derivative=True),
}

EXCLUDE_SOURCES_FROM_REPR = {"adenine_sers_control"}  # perturbation conc-series


@dataclass
class Row:
    spectrum_id: str
    analyte: str            # canonical
    modality: str           # raman / sers
    source: str
    acquisition_domain: str
    replicate: str
    vector: np.ndarray      # processed intensity on GRID


def _preprocess(wn, y, cfg):
    deriv = cfg.get("derivative", False)
    base = {k: cfg[k] for k in ("baseline", "smooth", "norm")}
    v = pp.preprocess(wn, y, base, GRID)
    if deriv:
        d = np.gradient(np.nan_to_num(v, nan=0.0))
        n = np.linalg.norm(d)
        v = d / n if n > 1e-12 else d
    return v


def build_phase2_input(preproc="A1_asls_savgol_l2"):
    """Return (rows, excluded) for the audited 785-nm grounding corpus."""
    cfg = PREPROCS[preproc]
    rows, excluded = [], []

    def emit(spec, note_ok):
        v = _preprocess(spec.wavenumber, spec.intensity, cfg)
        if not np.isfinite(v).any():
            excluded.append((spec.record.spectrum_id, "excluded:empty_after_preproc")); return
        rows.append(Row(spec.record.spectrum_id, canonical(spec.record.canonical_analyte_name),
                        spec.record.modality.value, spec.record.source_dataset,
                        spec.record.acquisition_domain, spec.record.replicate, v))

    for s in loader.load_ramanbiolib():
        (emit(s, "785") if s.record.excitation_nm == 785.0
         else excluded.append((s.record.spectrum_id, f"excluded:non-785({s.record.excitation_nm})")))
    for s in loader.load_metabolite63():
        excluded.append((s.record.spectrum_id, "excluded:633nm"))
    for s in loader.load_adenine():
        excluded.append((s.record.spectrum_id, "excluded:adenine_conc_series(perturbation_eval)"))
    for s in gobbato.load_gobbato_785():
        emit(s, "785")
    for s in loader.load_orc_ag_peaks():
        excluded.append((s.record.spectrum_id, "excluded:peak_only(MSS)"))
    return rows, excluded


def matrix(rows):
    """Return (X, meta_df-like dict). NaN in vectors -> 0 (outside-range)."""
    import pandas as pd
    X = np.vstack([np.nan_to_num(r.vector, nan=0.0) for r in rows])
    meta = pd.DataFrame([dict(spectrum_id=r.spectrum_id, analyte=r.analyte, modality=r.modality,
                              source=r.source, replicate=r.replicate) for r in rows])
    return X, meta
