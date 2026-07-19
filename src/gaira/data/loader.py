"""GAIRA V5 — the ONE canonical spectrum loader (Phase 0 / V5.0).

Reads the loadable direct-grounding sources into a common `Spectrum` form with
full provenance. No preprocessing, no scoring, no fitting here — loading only.

Sources implemented (those with accessible full spectra or peak evidence):
  * RamanBioLib        (grounding_molecule_spectra.parquet + index)   — Raman
  * metabolite-63      (NIHMS1547448-supplement-2.xlsx)               — Ag-SERS 633 nm
  * adenine_sers_control (Adenine_bAgNPs_*.CSV)                       — Ag-SERS
  * ORC-Ag metabolite peaks (v4 registry)                            — peak-only

Deterministic, read-only.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import (Spectrum, SpectrumRecord, Modality, SubstrateGeometry,
                     IntendedRole)

REPO = Path("/Users/surajpg/projects/GAIRA")
DATA = Path("/Volumes/SSD_Rad/GAIRA_DATA")
RAMANBIOLIB_PARQUET = REPO / "streamlit_apps/gaira_demo/data/grounding_molecule_spectra.parquet"
RAMANBIOLIB_INDEX = REPO / "streamlit_apps/gaira_demo/data/grounding_molecule_index.csv"
METABOLITE63_XLSX = DATA / "raw/sers_metabolite_63/NIHMS1547448-supplement-2.xlsx"
ADENINE_DIR = DATA / "raw/adenine_sers_control"
ORC_PEAKS = REPO / "data_audit/v4_ag_flakes_metabolite24_peak_registry.csv"


def _dom(modality: Modality, geom: SubstrateGeometry, exc) -> str:
    return f"{modality.value}|{geom.value}|{exc if exc else 'na'}"


# ── RamanBioLib (spontaneous Raman) ──────────────────────────────────
def load_ramanbiolib():
    if not RAMANBIOLIB_PARQUET.exists():
        return []
    sp = pd.read_parquet(RAMANBIOLIB_PARQUET)
    idx = pd.read_csv(RAMANBIOLIB_INDEX)
    idx = idx.set_index("id")
    out = []
    for sid, g in sp.groupby("id"):
        g = g.sort_values("wavenumber")
        meta = idx.loc[sid] if sid in idx.index else None
        name = str(meta["component"]) if meta is not None else str(sid)
        substrate = str(meta["sample_substrate"]) if meta is not None else ""
        laser = meta["laser_wavelength"] if meta is not None else None
        try:
            exc = float(laser)
        except (TypeError, ValueError):
            exc = None
        wn = g["wavenumber"].to_numpy(float)
        y = g["intensity_norm"].to_numpy(float)
        rec = SpectrumRecord(
            spectrum_id=f"ramanbiolib::{sid}", analyte_id=f"rbl::{name.lower()}",
            canonical_analyte_name=name, source_dataset="RamanBioLib",
            raw_path=str(RAMANBIOLIB_PARQUET), modality=Modality.RAMAN, sers_or_raman="raman",
            substrate_material=substrate, substrate_geometry=SubstrateGeometry.NONE,
            colloid_or_planar="none", excitation_nm=exc, instrument="digitized(literature)",
            matrix="powder/standard", concentration="n/a", replicate="1",
            independent_measurement=True, raw_or_processed="processed(digitized,norm)",
            wavenumber_min=float(wn.min()), wavenumber_max=float(wn.max()), point_count=len(wn),
            quality_flags="", intended_role=IntendedRole.GROUNDING,
            allowed_for_representation_learning=True, allowed_for_evaluation_only=False,
            acquisition_domain=_dom(Modality.RAMAN, SubstrateGeometry.NONE, exc))
        out.append(Spectrum(record=rec, wavenumber=wn, intensity=y))
    return out


# ── metabolite-63 (Ag citrate colloid SERS, 633 nm) ──────────────────
def load_metabolite63():
    if not METABOLITE63_XLSX.exists():
        return []
    df = pd.read_excel(METABOLITE63_XLSX, sheet_name=0)
    cols = list(df.columns)
    # Interleaved layout: a "Raman Shift" column precedes each block of analyte
    # intensity columns; each analyte pairs with the nearest preceding shift col.
    out = []
    cur_shift = None
    for c in cols:
        if re.match(r"Raman Shift", str(c)):
            cur_shift = c
            continue
        if cur_shift is None:
            continue
        wn = pd.to_numeric(df[cur_shift], errors="coerce").to_numpy(float)
        y = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        m = np.isfinite(wn) & np.isfinite(y)
        wn, y = wn[m], y[m]
        if len(wn) < 50:
            continue
        order = np.argsort(wn); wn, y = wn[order], y[order]
        name = str(c).replace(" Intensity", "").strip()
        rec = SpectrumRecord(
            spectrum_id=f"metabolite63::{name.lower()}", analyte_id=f"m63::{name.lower()}",
            canonical_analyte_name=name, source_dataset="sers_metabolite_63(PMC6989628)",
            raw_path=str(METABOLITE63_XLSX), modality=Modality.SERS, sers_or_raman="sers",
            substrate_material="Ag citrate colloid (Lee-Meisel)", substrate_geometry=SubstrateGeometry.COLLOID,
            colloid_or_planar="colloid", excitation_nm=633.0, instrument="Acton SP2300 / PIXIS",
            matrix="aqueous", concentration="physiological", replicate="avg(3 scans)",
            independent_measurement=True, raw_or_processed="processed(avg,bg-subtracted)",
            wavenumber_min=float(wn.min()), wavenumber_max=float(wn.max()), point_count=len(wn),
            quality_flags="", intended_role=IntendedRole.GROUNDING,
            allowed_for_representation_learning=True, allowed_for_evaluation_only=False,
            acquisition_domain=_dom(Modality.SERS, SubstrateGeometry.COLLOID, 633.0))
        out.append(Spectrum(record=rec, wavenumber=wn, intensity=y))
    return out


# ── adenine Ag-SERS control (bAgNPs) ─────────────────────────────────
def load_adenine():
    files = sorted(ADENINE_DIR.glob("Adenine_bAgNPs_*.CSV")) if ADENINE_DIR.exists() else []
    out = []
    for p in files:
        try:
            df = pd.read_csv(p, sep=";", decimal=",", encoding="cp1252", header=None, names=["wn", "y"])
            wn = df["wn"].to_numpy(float); y = df["y"].to_numpy(float)
        except Exception:
            continue
        order = np.argsort(wn); wn, y = wn[order], y[order]
        label = p.stem.replace("Adenine_bAgNPs_", "")
        rec = SpectrumRecord(
            spectrum_id=f"adenine::{label}", analyte_id="adenine", canonical_analyte_name="Adenine",
            source_dataset="adenine_sers_control", raw_path=str(p), modality=Modality.SERS,
            sers_or_raman="sers", substrate_material="bAgNPs", substrate_geometry=SubstrateGeometry.COLLOID,
            colloid_or_planar="colloid", excitation_nm=785.0, instrument="portable Raman",
            matrix="aqueous", concentration=label, replicate="1", independent_measurement=True,
            raw_or_processed="raw", wavenumber_min=float(wn.min()), wavenumber_max=float(wn.max()),
            point_count=len(wn), quality_flags="", intended_role=IntendedRole.GROUNDING,
            allowed_for_representation_learning=True, allowed_for_evaluation_only=False,
            acquisition_domain=_dom(Modality.SERS, SubstrateGeometry.COLLOID, 785.0))
        out.append(Spectrum(record=rec, wavenumber=wn, intensity=y))
    return out


# ── ORC-Ag peak evidence (peak-only; excluded from full-spectrum analysis) ──
def load_orc_ag_peaks():
    if not ORC_PEAKS.exists():
        return []
    df = pd.read_csv(ORC_PEAKS)
    out = []
    for name, g in df.groupby("metabolite"):
        peaks = [(float(r["peak_cm1"]), str(r["intensity_code"]), bool(r["exclusive_characteristic"]))
                 for _, r in g.iterrows()]
        rec = SpectrumRecord(
            spectrum_id=f"orc_ag::{str(name).lower()}", analyte_id=f"orc::{str(name).lower()}",
            canonical_analyte_name=str(name), source_dataset="ag_flakes_metabolites_24",
            raw_path=str(ORC_PEAKS), modality=Modality.SERS, sers_or_raman="sers",
            substrate_material="ORC-roughened Ag", substrate_geometry=SubstrateGeometry.ORC,
            colloid_or_planar="planar", excitation_nm=None, instrument="unknown(SI)",
            matrix="aqueous", concentration="unknown", replicate="spatial-avg", independent_measurement=True,
            raw_or_processed="peak_table", wavenumber_min=None, wavenumber_max=None, point_count=len(peaks),
            quality_flags="peak_only;excitation_unknown;conc_unknown", intended_role=IntendedRole.PEAK_EVIDENCE,
            allowed_for_representation_learning=False, allowed_for_evaluation_only=True,
            acquisition_domain=_dom(Modality.SERS, SubstrateGeometry.ORC, None))
        out.append(Spectrum(record=rec, wavenumber=None, intensity=None, peaks=peaks))
    return out


def load_all():
    return (load_ramanbiolib() + load_metabolite63() + load_adenine() + load_orc_ag_peaks())
