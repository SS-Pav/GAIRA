"""GAIRA V7 — Phase 02: the Consensus Spectral Motif object.

The canonical V7 evidence unit. A CSM is a spectrum plus the complete record of what produced
it and how much to trust it; the two are inseparable, which is why they live in one object.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.signal import find_peaks

GRID_BINS = 676
STATUSES = ("accepted", "rejected", "singleton")


@dataclass
class CSM:
    csm_id: str
    index: int
    contributing_lsms: list[str]
    contributing_lsm_weights: list[float]
    member_indices: list[int]
    supporting_classes: list[str]
    supporting_analytes: list[str]
    projected_support: list[str]
    n_lsms: int
    n_classes: int
    n_analytes: int
    spectrum: np.ndarray
    dominant_bands: list[float]
    cohesion: float
    uncertainty: float
    mean_edge_weight: float
    min_edge_weight: float
    max_external_weight: float
    min_coassignment: float
    lsm_types: list[str]
    is_singleton: bool
    is_anchored: bool
    is_cross_class: bool
    consensus_operator: str
    # filled by validation
    status: str = "accepted"
    rejection_reason: str | None = None
    bootstrap_confidence: float = float("nan")
    ev_delta_vs_lsms: float = float("nan")
    loco_survival: float = float("nan")
    source_robust: bool = True
    band_assignment: str = ""
    interpretation: str = ""
    diagnostic_status: str = ""          # generic | diagnostic
    anchor_justification: str | None = None

    def __post_init__(self):
        self.spectrum = np.asarray(self.spectrum, float)
        if self.spectrum.shape != (GRID_BINS,):
            raise ValueError(f"{self.csm_id}: spectrum must be ({GRID_BINS},), "
                             f"got {self.spectrum.shape}")
        if (self.spectrum < 0).any():
            raise ValueError(f"{self.csm_id}: CSM spectra must be non-negative (contract C-07)")
        if self.is_singleton != (self.n_lsms == 1):
            raise ValueError(f"{self.csm_id}: is_singleton must equal n_lsms == 1 (C-07)")
        if self.is_anchored and not self.anchor_justification:
            raise ValueError(f"{self.csm_id}: is_anchored requires an anchor_justification")

    def to_row(self) -> dict:
        d = asdict(self)
        d.pop("spectrum")
        for k in ("contributing_lsms", "supporting_classes", "supporting_analytes",
                  "projected_support", "lsm_types", "member_indices"):
            d[k] = ";".join(map(str, d[k]))
        d["contributing_lsm_weights"] = ";".join(f"{w:.4f}" for w in d["contributing_lsm_weights"])
        d["dominant_bands"] = ";".join(f"{b:.0f}" for b in d["dominant_bands"])
        return d


def dominant_bands(h: np.ndarray, grid: np.ndarray, prominence: float = 0.05,
                   max_bands: int = 10) -> list[float]:
    """Peak positions carrying the motif's identity, strongest first, returned sorted.

    Prominence rather than raw height: a shoulder on a strong band is not an independent
    diagnostic feature, and treating it as one inflates every position-based comparison.
    """
    x = h / (h.max() + 1e-12)
    idx, props = find_peaks(x, prominence=prominence)
    if idx.size == 0:
        return []
    order = np.argsort(props["prominences"])[::-1][:max_bands]
    return sorted(float(grid[i]) for i in idx[order])
