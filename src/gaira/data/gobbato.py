"""GAIRA V5 — Gobbato/Bonifacio pure-metabolite loader (Phase 1.5).

Loads pure Raman powder + pure Ag-SERS metabolite spectra (B&WTek BWS465-785S,
785 nm) from serum_ag_colloids/dataset_spectral_data.zip. Extracts the zip to a
stable temp cache (never into source dirs). Deterministic.

File format: cp1252, ';'-delimited, comma decimal, 88-line header; columns include
'Raman Shift' (wavenumber) and 'Dark Subtracted #1' (intensity).
"""
from __future__ import annotations
import re, zipfile, tempfile
from pathlib import Path

import numpy as np

from .schema import Spectrum, SpectrumRecord, Modality, SubstrateGeometry, IntendedRole
from .synonyms import GOBBATO_ABBREV, canonical

ZIP = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/serum_ag_colloids/dataset_spectral_data.zip")
CACHE = Path(tempfile.gettempdir()) / "gaira_v5_gobbato_cache"


def _ensure_extracted() -> Path | None:
    if not ZIP.exists():
        return None
    marker = CACHE / ".extracted"
    if not marker.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ZIP) as z:
            z.extractall(CACHE)
        marker.write_text("ok")
    return CACHE


def _parse_bwtek(path: Path):
    lines = path.read_text(encoding="cp1252", errors="replace").splitlines()
    hdr = next((i for i, l in enumerate(lines)
                if "Raman Shift" in l and "Dark Subtracted" in l), None)
    if hdr is None:
        return None
    cols = [c.strip() for c in lines[hdr].split(";")]
    try:
        i_wn = cols.index("Raman Shift")
        i_y = next(i for i, c in enumerate(cols) if c.startswith("Dark Subtracted"))
    except (ValueError, StopIteration):
        return None
    wn, y = [], []
    for l in lines[hdr + 1:]:
        p = l.split(";")
        if len(p) <= max(i_wn, i_y):
            continue
        try:
            wn.append(float(p[i_wn].replace(",", ".")))
            y.append(float(p[i_y].replace(",", ".")))
        except ValueError:
            continue
    if len(wn) < 100:
        return None
    wn = np.array(wn); y = np.array(y)
    o = np.argsort(wn)
    return wn[o], y[o]


def load_gobbato_785():
    root = _ensure_extracted()
    if root is None:
        return []
    out = []
    # pure Ag-SERS metabolites
    sers_dir = next((p for p in root.rglob("SERS metabolites")
                     if p.is_dir() and "fitting" not in p.name.lower()), None)
    if sers_dir:
        for f in sorted(sers_dir.glob("*.txt")):
            m = re.match(r"SERS_met_(.+?)_(.+?)_(\d+)\.txt", f.name)
            if not m:
                continue
            abbrev, conc, rep = m.groups()
            name = GOBBATO_ABBREV.get(abbrev, abbrev.lower())
            parsed = _parse_bwtek(f)
            if parsed is None:
                continue
            wn, y = parsed
            out.append(_mk(f, name, abbrev, Modality.SERS, "Ag colloid (Gobbato)",
                           SubstrateGeometry.COLLOID, "colloid", conc, rep, wn, y,
                           "gobbato_sers_metabolites"))
    # pure Raman powders
    raman_dir = next((p for p in root.rglob("Raman metabolites") if p.is_dir()), None)
    if raman_dir:
        for f in sorted(raman_dir.glob("*.txt")):
            m = re.match(r"Raman_pwd_(.+?)_s_(\d+)\.txt", f.name)
            if not m:
                continue
            abbrev, rep = m.groups()
            name = GOBBATO_ABBREV.get(abbrev, abbrev.lower())
            parsed = _parse_bwtek(f)
            if parsed is None:
                continue
            wn, y = parsed
            out.append(_mk(f, name, abbrev, Modality.RAMAN, "powder", SubstrateGeometry.NONE,
                           "none", "neat", rep, wn, y, "gobbato_raman_metabolites"))
    return out


def _mk(f, name, abbrev, modality, sub_mat, geom, cp, conc, rep, wn, y, source):
    return Spectrum(record=SpectrumRecord(
        spectrum_id=f"{source}::{abbrev}::{rep}", analyte_id=canonical(name),
        canonical_analyte_name=name, source_dataset=source, raw_path=str(f),
        modality=modality, sers_or_raman=modality.value, substrate_material=sub_mat,
        substrate_geometry=geom, colloid_or_planar=cp, excitation_nm=785.0,
        instrument="B&WTek BWS465-785S", matrix="aqueous" if modality == Modality.SERS else "powder",
        concentration=conc, replicate=rep, independent_measurement=True,
        raw_or_processed="raw(bwtek dark-subtracted)", wavenumber_min=float(wn.min()),
        wavenumber_max=float(wn.max()), point_count=len(wn), quality_flags="",
        intended_role=IntendedRole.GROUNDING, allowed_for_representation_learning=True,
        allowed_for_evaluation_only=False,
        acquisition_domain=f"{modality.value}|{geom.value}|785.0"),
        wavenumber=wn, intensity=y)
