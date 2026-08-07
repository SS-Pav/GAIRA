"""GAIRA V7 Phase 11 — display-only reference data.

The demo needs the LSM and CSM basis SPECTRA in order to draw a motif when a user clicks one.
Those are reference *data*, not results: the engine reads the same arrays to project, and the
demo reads them to plot. Nothing here computes.

The files are the ones pinned by `gaira.v7.runtime.freeze`, and their digests are verified
against that ledger before use — so the motif a user sees is provably the motif the engine used.
"""
from __future__ import annotations

import json

import numpy as np

from gaira.v7.io import frozen_root
from gaira.v7.runtime import freeze as FREEZE


def load_reference_motifs() -> dict:
    """Grid, LSM and CSM dictionaries plus the CSM registry — verified, read-only."""
    FREEZE.verify(strict=True)          # refuse to display data from an unverified atlas
    F = frozen_root()
    z1 = np.load(F / "phase01/artifacts/lsm_dictionary_v1.npz", allow_pickle=True)
    z2 = np.load(F / "phase02/artifacts/csm_dictionary_v1.npz", allow_pickle=True)
    br = np.load(F / "phase01/artifacts/balanced_references_v1.npz", allow_pickle=True)
    reg = json.loads((F / "phase02/artifacts/csm_registry_v1.json").read_text())
    return {
        "grid": np.asarray(br["grid"], float),
        "H_lsm": np.asarray(z1["H"], float),
        "lsm_ids": [str(s) for s in z1["motif_ids"]],
        "CSM": np.asarray(z2["CSM"], float),
        "csm_ids": [str(s) for s in z2["csm_ids"]],
        "csm_records": {c["csm_id"]: c for c in reg["csms"]},
    }


def load_reference_spectra() -> dict:
    """The 375 corpus spectra keyed by molecule, for query-vs-reference overlays."""
    F = frozen_root()
    br = np.load(F / "phase01/artifacts/balanced_references_v1.npz", allow_pickle=True)
    X = np.asarray(br["X"], float)
    y = [str(s) for s in br["canonical_id"]]
    by_molecule: dict[str, np.ndarray] = {}
    for m in sorted(set(y)):
        rows = X[[i for i, v in enumerate(y) if v == m]]
        by_molecule[m] = rows.mean(axis=0)
    return by_molecule


DEMO_SPECTRA = {
    "— choose a built-in example —": None,
    "Cholesterol (sterol)": "cholesterol",
    "L-Cysteine (amino acid)": "cysteine",
    "Glucose (sugar)": "(+)-glucose",
    "Palmitic acid (fatty acid)": "palmitic acid",
    "Adenine (purine)": "adenine",
    "Albumin (protein)": "albumin",
    "Pyruvate (hard case — low explained variance)": "pyruvate",
}
