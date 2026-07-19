"""Direct spectral representation (Phase 2 Stage A).

The "direct representation" of a spectrum is simply its preprocessed intensity
vector on the common grid — no learned encoder, no chemical-feature engineering.
This module is the explicit entry point so callers name the representation they
are testing. Stage B (chemical features) and Stage C (learned embeddings) are
NOT implemented here and must not be started until Stage A is complete.
"""
from __future__ import annotations
import numpy as np
from . import datasets


def direct_representation(preproc="A1_asls_savgol_l2"):
    """Return (X, meta) — the direct spectral representation of the audited
    785 nm grounding corpus under the chosen preprocessing."""
    rows, _ = datasets.build_phase2_input(preproc)
    return datasets.matrix(rows)


def modality_split(X, meta):
    r = meta.modality.values == "raman"
    s = meta.modality.values == "sers"
    return (X[r], meta[r].reset_index(drop=True)), (X[s], meta[s].reset_index(drop=True))
