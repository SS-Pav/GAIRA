"""Stage B0 — assemble the RAW frozen Stage-B corpus (no preprocessing applied).

Reuses gaira.data loaders and the frozen Phase-2 input manifest; this module only
selects and aligns, it does not re-implement loading. Raw (wavenumber, intensity)
pairs are returned so the AutoResearch search can apply its own stage-1 pipelines.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from ..data import loader, gobbato
from ..data.synonyms import canonical
from ..evidence.families import family_of

MANIFEST = Path("/Users/surajpg/projects/GAIRA/results/v5_rebuild/phase2_stage_a/"
                "tables/phase2_input_manifest.csv")


def load_raw_frozen(manifest_path=MANIFEST):
    """Return (raw, meta) for exactly the 479 spectra of the frozen Stage-B corpus.
    raw: list of (spectrum_id, wavenumber, intensity, modality) in meta order."""
    mf = pd.read_csv(manifest_path)
    ent = mf[mf.entered_representation].copy()

    specs = {}
    for s in loader.load_ramanbiolib():
        specs[s.record.spectrum_id] = s
    for s in gobbato.load_gobbato_785():
        specs[s.record.spectrum_id] = s

    rows, raw = [], []
    for r in ent.itertuples():
        s = specs.get(r.spectrum_id)
        if s is None:
            continue
        a = canonical(s.record.canonical_analyte_name)
        rows.append({"spectrum_id": r.spectrum_id, "analyte": a,
                     "modality": s.record.modality.value, "source": s.record.source_dataset,
                     "replicate": str(s.record.replicate),
                     "replicate_group": f"{a}|{s.record.modality.value}|{s.record.source_dataset}",
                     "acquisition_domain": f"{s.record.modality.value}|{s.record.source_dataset}",
                     "family": family_of(a), "data_role": "grounding"})
        raw.append((r.spectrum_id, np.asarray(s.wavenumber, float),
                    np.asarray(s.intensity, float), s.record.modality.value))
    meta = pd.DataFrame(rows)
    ram = set(meta[meta.modality == "raman"].analyte)
    ser = set(meta[meta.modality == "sers"].analyte)
    matched = ram & ser
    meta["matched"] = meta.analyte.isin(matched)
    ms = meta[meta.modality == "raman"].groupby("analyte").source.nunique()
    meta["raman_multi_source"] = meta.analyte.map(ms).fillna(0).astype(float) > 1
    return raw, meta


def matched_analytes(meta):
    return sorted(set(meta[(meta.modality == "raman")].analyte) &
                  set(meta[(meta.modality == "sers")].analyte))
