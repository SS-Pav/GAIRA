"""GAIRA V5 Foundation — the frozen RAMAN-ONLY reference corpus.

Canonical observation domain = Raman. Ag-SERS, Au-SERS and DART are explicitly
EXCLUDED and remain future observation domains.

Reference corpus (pure analytes only):
  * RamanBioLib          — 202 spectra, 141 compounds, 9 excitations, 450-1800 grid
  * Gobbato Raman powders— 153 spectra, 51 metabolites, 785 nm
  * amino-acid grounding —  20 pure amino-acid Raman references

External (never used for representation learning):
  * covid_serum_raman    — biological SERUM RAMAN (COVID / suspected / healthy / tube)

Reuses gaira.data loaders and gaira.preprocessing (grid/resample/baseline/norm);
only the not-yet-wired amino-acid and serum-Raman readers are added here.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from ..data import loader, gobbato
from ..data.synonyms import canonical
from ..preprocessing import pipeline as pp

RAW = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw")
AA_XLSX = RAW / "amino_acid_raman_grounding/aa.xlsx"
COVID_DIR = RAW / "covid_serum_raman"
KNOWLEDGE = RAW / "raman_knowledge_core/peak_assignments.csv"

# Raman fingerprint window. Wider than the Stage-B 520-1750 because we are no
# longer constrained by Ag-SERS overlap; all sources cover 450-1800.
WINDOW = (450.0, 1800.0)
GRID = pp.common_grid(*WINDOW, 2.0)

# canonical preprocessing for the reference corpus (deterministic, reused primitives)
PREPROC = dict(baseline="asls", smooth="savgol", norm="l2")

AA_NAME_FIX = {
    "l-glu": "glutamate", "glutamic acid": "glutamate", "leu": "leucine", "phe": "phenylalanine",
    "pro": "proline", "ala": "alanine", "arg": "arginine", "asp": "aspartate", "gly": "glycine",
    "his": "histidine", "ile": "isoleucine", "lys": "lysine", "met": "methionine",
    "ser": "serine", "thr": "threonine", "trp": "tryptophan", "tyr": "tyrosine",
    "val": "valine", "cys": "cysteine", "asn": "asparagine", "gln": "glutamine",
}


@dataclass
class RamanCorpus:
    X: np.ndarray          # (n_spectra, n_bins) preprocessed on GRID
    grid: np.ndarray
    meta: pd.DataFrame     # analyte, source, excitation, replicate, spectrum_id
    preproc: dict

    @property
    def analytes(self):
        return sorted(self.meta.analyte.unique())

    def subset(self, mask):
        m = np.asarray(mask)
        return RamanCorpus(self.X[m], self.grid, self.meta[m].reset_index(drop=True), self.preproc)


def _prep(wn, y):
    return pp.preprocess(np.asarray(wn, float), np.asarray(y, float), PREPROC, GRID, WINDOW)


def _load_amino_acids():
    """20 pure amino-acid Raman references (column-per-analyte spreadsheet)."""
    if not AA_XLSX.exists():
        return [], []
    df = pd.read_excel(AA_XLSX)
    wn = df.iloc[:, 0].values.astype(float)
    rows, recs = [], []
    for col in df.columns[1:]:
        name = canonical(AA_NAME_FIX.get(str(col).strip().lower(), str(col).strip().lower()))
        y = df[col].values.astype(float)
        ok = np.isfinite(y)
        if ok.sum() < 100:
            continue
        v = _prep(wn[ok], y[ok])
        if not np.isfinite(v).any():
            continue
        rows.append(v)
        recs.append({"spectrum_id": f"amino_acid_raman::{name}", "analyte": name,
                     "source": "amino_acid_raman_grounding", "excitation_nm": 785.0,
                     "replicate": "1", "modality": "raman"})
    return rows, recs


def load_reference_corpus() -> RamanCorpus:
    """The frozen Raman-only pure-analyte reference corpus."""
    rows, recs = [], []
    for s in loader.load_ramanbiolib():                     # all excitations
        v = _prep(s.wavenumber, s.intensity)
        if not np.isfinite(v).any():
            continue
        rows.append(v)
        recs.append({"spectrum_id": s.record.spectrum_id,
                     "analyte": canonical(s.record.canonical_analyte_name),
                     "source": "RamanBioLib", "excitation_nm": float(s.record.excitation_nm),
                     "replicate": str(s.record.replicate), "modality": "raman"})
    for s in gobbato.load_gobbato_785():
        if s.record.modality.value != "raman":              # Ag-SERS EXCLUDED
            continue
        v = _prep(s.wavenumber, s.intensity)
        if not np.isfinite(v).any():
            continue
        rows.append(v)
        recs.append({"spectrum_id": s.record.spectrum_id,
                     "analyte": canonical(s.record.canonical_analyte_name),
                     "source": "gobbato_raman_metabolites", "excitation_nm": 785.0,
                     "replicate": str(s.record.replicate), "modality": "raman"})
    r, m = _load_amino_acids()
    rows += r; recs += m

    X = np.vstack(rows)
    meta = pd.DataFrame(recs)
    meta["replicate_group"] = meta.analyte + "|" + meta.source + "|" + meta.excitation_nm.astype(str)
    assert (meta.modality == "raman").all(), "non-Raman spectrum leaked into the reference corpus"
    return RamanCorpus(X, GRID, meta, dict(PREPROC))


def load_serum_raman():
    """External BIOLOGICAL serum Raman (COVID study). Never used for fitting."""
    if not COVID_DIR.exists():
        return None, None
    wn = np.loadtxt(COVID_DIR / "wave_number.txt")
    groups = {"COVID": "raw_COVID.txt", "Suspected": "raw_Suspected.txt",
              "Healthy": "raw_Helthy.txt", "Tube": "raw_Tube.txt"}
    rows, recs = [], []
    for g, fn in groups.items():
        p = COVID_DIR / fn
        if not p.exists():
            continue
        A = np.loadtxt(p)                                    # (n_wavenumbers, n_spectra)
        if A.shape[0] != len(wn):
            A = A.T
        n = min(A.shape[0], len(wn))
        for j in range(A.shape[1]):
            v = _prep(wn[:n], A[:n, j])
            if not np.isfinite(v).any():
                continue
            rows.append(v)
            recs.append({"spectrum_id": f"covid_serum_raman::{g}::{j:03d}", "group": g,
                         "source": "covid_serum_raman", "modality": "raman",
                         "sample_type": "serum" if g != "Tube" else "tube_blank"})
    if not rows:
        return None, None
    return np.vstack(rows), pd.DataFrame(recs)


def load_peak_assignments():
    """Literature Raman peak assignments (for axis interpretation, C3)."""
    if not KNOWLEDGE.exists():
        return pd.DataFrame()
    return pd.read_csv(KNOWLEDGE)


def dataset_card(c: RamanCorpus) -> dict:
    m = c.meta
    reps = m.groupby("replicate_group").size()
    multi = m.groupby("analyte").excitation_nm.nunique()
    return {
        "domain": "Raman only (canonical observation domain)",
        "excluded_domains": ["Ag-SERS", "Au-SERS", "DART", "serum Ag-colloid",
                             "metabolite-63 (633 nm Ag-SERS)", "adenine Ag-SERS series",
                             "european_multi_instrument_adenine (cAg/sAg/cAu substrates)"],
        "n_spectra": int(len(m)), "n_analytes": int(m.analyte.nunique()),
        "n_bins": int(c.X.shape[1]),
        "window_cm": list(WINDOW), "grid_step_cm": 2.0,
        "preprocessing": c.preproc,
        "sources": {k: int(v) for k, v in m.source.value_counts().items()},
        "excitations": {str(k): int(v) for k, v in m.excitation_nm.value_counts().items()},
        "n_analytes_multi_excitation": int((multi > 1).sum()),
        "replicate_groups": {"n": int(reps.shape[0]), "median_size": float(reps.median()),
                             "max_size": int(reps.max())},
        "analytes_with_replicates": int((m.groupby("analyte").size() > 1).sum()),
        "provenance": {
            "RamanBioLib": "digitized reference library, 141 compounds, 9 excitations",
            "gobbato_raman_metabolites": "B&WTek 785 nm pure metabolite powders",
            "amino_acid_raman_grounding": "20 pure amino-acid Raman references",
        },
        "external_validation_sets": {
            "covid_serum_raman": "biological serum Raman (COVID/suspected/healthy/tube) — "
                                 "projection only, never used for fitting",
        },
        "notes": [
            "Excitation is tracked as a nuisance factor and tested for leakage; all sources "
            "share the 450-1800 cm-1 window.",
            "The Ag-SERS-constrained 520-1750 window of Stage A/B is not used here.",
        ],
    }
