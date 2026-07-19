"""GAIRA engine — reference normalization against the frozen atlas (Part 10).

Replaces cohort-mean normalization. A query's component coordinates are z-scored
against the reference distribution (median/MAD over the 375 pure-analyte spectra),
and an out-of-distribution score is the cosine distance to the reference support.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
ARTIFACTS = REPO / "results/v5_rebuild/engine_v1/artifacts"


class ReferenceFrame:
    def __init__(self, path=None):
        path = Path(path) if path else ARTIFACTS / "reference_normalization_v1.json"
        d = json.loads(path.read_text())
        self.center = np.asarray(d["component_center"], float)
        self.spread = np.asarray(d["component_spread"], float)
        self.fingerprint = d["atlas_fingerprint"]
        self.n_reference = d["n_reference_spectra"]
        sup = np.load(ARTIFACTS / "reference_support.npz", allow_pickle=True)
        self._support = sup["support_unit"]

    def zscore(self, coord):
        """Robust z-score of each component vs the reference distribution."""
        return (np.nan_to_num(coord) - self.center) / self.spread

    def ood_score(self, coord, k=5):
        """1 - mean cosine to the k nearest reference spectra (higher = more OOD)."""
        v = np.clip(np.nan_to_num(np.atleast_2d(coord)), 0, None)
        u = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        s = u @ self._support.T
        topk = np.sort(s, axis=1)[:, -k:]
        out = 1.0 - topk.mean(axis=1)
        return float(out[0]) if out.shape[0] == 1 else out

    def display_scale(self, elevation, cap=3.0):
        """Bounded 0..1 display value from a signed elevation z (Part 10 bounded viz).
        Uses a saturating map so extreme values do not dominate the radar."""
        return float(np.clip(0.5 + 0.5 * np.tanh(elevation / cap), 0.0, 1.0))
