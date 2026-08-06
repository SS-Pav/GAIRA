"""GAIRA V7 — the Atlas Component Substructure (ACS) object.

WHAT AN ACS IS IN PHASE 01
--------------------------
An ACS is a **deterministic decomposition of one frozen atlas component into a reusable
spectral substructure**. It is NOT a new atlas component and it never enters projection.

    frozen atlas component  h_k  ∈ ℝ₊^676        (24 of these, immutable)
              │
              ├── ACS k.0    a subset of h_k's bands that co-occur across a coherent
              ├── ACS k.1    group of analytes activating component k
              └── …

The atlas, the projection and the fingerprint are unchanged. Only the interpretation layer
gains resolution: given a spectrum's activation of component k, the motif layer can say
*which substructure of k* that activation is carrying.

A NOTE ON TERMINOLOGY — read this before comparing with the architecture documents
----------------------------------------------------------------------------------
`GAIRA_v7_rebuild/architecture/LEARNING_MODE_ARCHITECTURE.md` defines an ACS as a row of
`H_c` from a class-local NMF over balanced references. That is a different construction
from the one implemented here, which decomposes the *existing frozen components*. Both are
"atlas component substructures" in the sense of local substructure, but they are not the same
object and must not be conflated. See `results/v7_rebuild/phase01/reports/PHASE_01_REPORT.md`
§"Divergence from the architecture documents".

INVARIANTS
    * every motif is non-negative and lives on the canonical 676-bin grid
    * every motif is a masked, renormalised restriction of exactly one parent component
    * every motif carries its full participation record and its rejection status
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

GRID_BINS = 676


@dataclass(frozen=True)
class Band:
    """One diagnostic band of a parent component."""
    index: int                 # bin index of the peak on the canonical grid
    center_cm: float
    lo_bin: int
    hi_bin: int                # inclusive
    prominence: float
    component_weight: float    # mass of the parent component inside this band

    def slice(self) -> slice:
        return slice(self.lo_bin, self.hi_bin + 1)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ACS:
    """A Atlas Component Substructure: a substructure of one frozen atlas component."""

    motif_id: str                      # "c07.m02" — stable, deterministic
    parent_component: int
    index_in_component: int

    # spectral content
    spectrum: np.ndarray               # ℝ₊^676, masked restriction of the parent
    band_indices: list[int]            # which of the parent's bands this motif carries
    band_centers_cm: list[float]
    band_weights: list[float]          # motif centroid profile over its own bands

    # participation
    analytes: list[str]                # canonical IDs
    n_analytes: int
    n_spectra: int
    fine_classes: dict[str, int]       # class -> count among participating analytes
    broad_classes: dict[str, int]
    sources: dict[str, int]

    # scores
    stability: float                   # jackknife co-assignment consistency, [0, 1]
    purity: float                      # dominant fine-class fraction, [0, 1]
    coverage_analytes: float           # share of the component's participants
    coverage_spectra: float
    dominant_class: str
    band_fidelity: float               # agreement with the parent inside the motif's bands
    redundancy_max: float              # max cosine to another motif of the same component

    # status
    retained: bool = True
    rejection_reason: str = ""
    provenance: dict = field(default_factory=dict)

    # ── derived ───────────────────────────────────────────────────────────────
    @property
    def n_bands(self) -> int:
        return len(self.band_indices)

    @property
    def is_singleton_class(self) -> bool:
        return len(self.fine_classes) == 1

    def normalised(self) -> np.ndarray:
        n = float(np.linalg.norm(self.spectrum))
        return self.spectrum / n if n > 0 else self.spectrum

    def cosine(self, other: "ACS") -> float:
        a, b = self.normalised(), other.normalised()
        return float(np.dot(a, b))

    # ── serialisation ─────────────────────────────────────────────────────────
    def to_record(self) -> dict:
        """Registry row — everything except the dense spectrum, which is stored separately."""
        return {
            "motif_id": self.motif_id,
            "parent_component": self.parent_component,
            "index_in_component": self.index_in_component,
            "n_bands": self.n_bands,
            "band_centers_cm": ";".join(f"{c:.0f}" for c in self.band_centers_cm),
            "band_indices": ";".join(str(i) for i in self.band_indices),
            "band_weights": ";".join(f"{w:.4f}" for w in self.band_weights),
            "n_analytes": self.n_analytes,
            "n_spectra": self.n_spectra,
            "analytes": ";".join(self.analytes),
            "dominant_class": self.dominant_class,
            "fine_classes": ";".join(f"{k}:{v}" for k, v in sorted(self.fine_classes.items())),
            "broad_classes": ";".join(f"{k}:{v}" for k, v in sorted(self.broad_classes.items())),
            "sources": ";".join(f"{k}:{v}" for k, v in sorted(self.sources.items())),
            "stability": round(self.stability, 4),
            "purity": round(self.purity, 4),
            "coverage_analytes": round(self.coverage_analytes, 4),
            "coverage_spectra": round(self.coverage_spectra, 4),
            "band_fidelity": round(self.band_fidelity, 4),
            "redundancy_max": round(self.redundancy_max, 4),
            "retained": self.retained,
            "rejection_reason": self.rejection_reason,
        }

    def validate(self) -> list[str]:
        """Structural invariants. Returns a list of violations (empty = valid)."""
        bad = []
        if self.spectrum.shape != (GRID_BINS,):
            bad.append(f"spectrum shape {self.spectrum.shape} != ({GRID_BINS},)")
        if not np.all(np.isfinite(self.spectrum)):
            bad.append("spectrum contains non-finite values")
        if np.any(self.spectrum < 0):
            bad.append("spectrum has negative values (non-negativity violated)")
        if self.n_bands == 0:
            bad.append("motif carries no bands")
        if len(self.band_centers_cm) != self.n_bands or len(self.band_weights) != self.n_bands:
            bad.append("band arrays disagree in length")
        if self.n_analytes != len(self.analytes):
            bad.append("n_analytes disagrees with the analyte list")
        if not (0.0 <= self.purity <= 1.0):
            bad.append(f"purity {self.purity} outside [0, 1]")
        if not (0.0 <= self.stability <= 1.0):
            bad.append(f"stability {self.stability} outside [0, 1]")
        if not self.retained and not self.rejection_reason:
            bad.append("rejected motif carries no rejection reason")
        if self.retained and self.rejection_reason:
            bad.append("retained motif carries a rejection reason")
        return bad


def build_motif_spectrum_representative(parent: np.ndarray, bands: list[Band],
                                        carried: list[int],
                                        weights: np.ndarray) -> np.ndarray:
    """Alternative construction: scale each band by the motif's own centroid weight.

    Benchmarked and NOT selected. It sounds more faithful — "the part of the component this
    group actually carries" — but measured on this corpus it makes the motifs of a component
    near-duplicates of one another, because they all inherit the parent's dominant peak:
    25 motif pairs above cosine 0.9 and a maximum of 0.979, against 0 pairs and a maximum of
    0.844 for the selected construction, with no gain in chemical alignment. Recorded here
    so the choice is auditable rather than asserted.
    """
    m = np.zeros_like(parent, dtype=float)
    if not carried:
        return m
    w = np.asarray(weights, float)
    w = w / (w.max() + 1e-12)
    for s, b in zip(w, carried):
        sl = bands[b].slice()
        m[sl] = parent[sl] * float(s)
    return m


def build_motif_spectrum(parent: np.ndarray, bands: list[Band], carried: list[int],
                         weights: np.ndarray) -> np.ndarray:
    """Mask the parent component down to the motif's bands, reweighted by the centroid.

    The motif spectrum is a **restriction of the parent**, never a new fit: outside the
    carried bands it is exactly zero, and inside them it is the parent's own shape scaled
    by how strongly the motif's analytes emphasise that band relative to the parent. This
    is what keeps a motif traceable to its component and keeps the layer interpretive
    rather than generative.
    """
    m = np.zeros_like(parent, dtype=float)
    if not carried:
        return m
    w = np.asarray(weights, float)
    w = w / (w.sum() + 1e-12)
    # parent's own share of mass across the carried bands
    par = np.array([parent[bands[b].slice()].sum() for b in carried], float)
    par = par / (par.sum() + 1e-12)
    scale = w / (par + 1e-12)
    scale = scale / (scale.max() + 1e-12)          # keep in [0, 1]; preserves shape
    for s, b in zip(scale, carried):
        sl = bands[b].slice()
        m[sl] = parent[sl] * float(s)
    return m
