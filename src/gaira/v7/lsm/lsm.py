"""GAIRA V7 — the Local Spectral Motif object (canonical definition).

An LSM is a **row of `H_c` from an independent class-local non-negative factorisation**:

    X_c ≈ W_c H_c,    W_c ∈ ℝ₊^{n_c×k_c},   H_c ∈ ℝ₊^{k_c×D},   D = 676

retained only if it is **stable** — recurrent across repeated fits under seed variation and
analyte-level resampling, above the pre-registered threshold, after Hungarian alignment.

`X_c` is the balanced reference block for chemistry class `c`. **The frozen V5 atlas is not
involved** (P-15): an LSM is a newly fitted basis vector, free to occupy any direction in
`ℝ₊^676` that its class's own data supports. That freedom is the whole point — it is what lets
a 4-molecule class receive capacity that a global fit would never have given it.

TYPING — required by the specification because Phase 02 depends on it

    class-shared               activates on most molecules of the class
    subfamily                  activates on a coherent proper subset
    molecule-discriminating    activates on very few, with high selectivity

Class-shared LSMs from *different* classes describing the same chemistry are exactly what the
consensus phase must merge; molecule-discriminating LSMs are exactly what it must not.

ANCHORS — classes too small to fit (`n_analytes < 2`) route here (Strategy F). An anchored
LSM is a single high-quality reference admitted directly, permanently flagged, and never
presented as having consensus support it does not have.

A NOTE ON `purity`
------------------
`purity` is the dominant broad-class fraction among an LSM's activating molecules. Inside a
class-local fit it is **structurally 1.0**: every molecule in a fine class shares that class's
broad superclass, so the field cannot discriminate and must not be read as evidence of
chemical coherence. It is retained because contract C-05 specifies it and because it becomes
informative in Phase 02, where LSMs from different classes are pooled. Within-class structure
is carried instead by `lsm_type` and `activation_sparsity`, which are label-free and do vary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GRID_BINS = 676

LSM_TYPES = ("class_shared", "subfamily", "molecule_discriminating")


@dataclass
class LSM:
    """One Local Spectral Motif: a stability-selected row of a class-local `H_c`."""

    motif_id: str                      # "protein.m02" — class-scoped, deterministic
    chemical_class: str                # the class whose local fit produced it
    index_in_class: int

    # spectral content — a NEWLY FITTED basis vector, not a restriction of anything
    spectrum: np.ndarray               # ℝ₊^676, row of H_c
    dominant_bands: list[dict]         # [{center_cm, prominence, weight}, ...]

    # participation within its class
    analytes: list[str]                # canonical IDs activating this LSM
    n_analytes: int
    n_spectra: int
    activation_share: float            # share of the class's total activation mass
    activation_sparsity: float         # fraction of class molecules NOT activating it

    # scores
    stability: float                   # recurrence across repeated fits, [0, 1]
    matched_similarity: float          # mean cosine of Hungarian-matched partners
    purity: float                      # dominant broad-class fraction among its analytes
                                       # DEGENERATE within a class-local fit — see note
    reconstruction_share: float        # share of the class's explained variance
    redundancy_max: float              # max cosine to a sibling LSM of the same class
    lsm_type: str                      # one of LSM_TYPES

    # class context
    k_c: int                           # the class's selected motif count
    n_class_analytes: int
    dominant_broad_class: str

    # status
    retained: bool = True
    rejection_reason: str = ""
    is_anchor: bool = False
    anchor_justification: str = ""
    provenance: dict = field(default_factory=dict)

    # ── derived ───────────────────────────────────────────────────────────────
    @property
    def n_bands(self) -> int:
        return len(self.dominant_bands)

    def normalised(self) -> np.ndarray:
        n = float(np.linalg.norm(self.spectrum))
        return self.spectrum / n if n > 0 else self.spectrum

    def cosine(self, other: "LSM") -> float:
        return float(np.dot(self.normalised(), other.normalised()))

    # ── serialisation ─────────────────────────────────────────────────────────
    def to_record(self) -> dict:
        return {
            "motif_id": self.motif_id,
            "chemical_class": self.chemical_class,
            "index_in_class": self.index_in_class,
            "k_c": self.k_c,
            "n_class_analytes": self.n_class_analytes,
            "n_bands": self.n_bands,
            "band_centers_cm": ";".join(f"{b['center_cm']:.0f}" for b in self.dominant_bands),
            "n_analytes": self.n_analytes,
            "n_spectra": self.n_spectra,
            "analytes": ";".join(self.analytes),
            "lsm_type": self.lsm_type,
            "dominant_broad_class": self.dominant_broad_class,
            "activation_share": round(self.activation_share, 4),
            "activation_sparsity": round(self.activation_sparsity, 4),
            "stability": round(self.stability, 4),
            "matched_similarity": round(self.matched_similarity, 4),
            "purity": round(self.purity, 4),
            "reconstruction_share": round(self.reconstruction_share, 4),
            "redundancy_max": round(self.redundancy_max, 4),
            "retained": self.retained,
            "rejection_reason": self.rejection_reason,
            "is_anchor": self.is_anchor,
            "anchor_justification": self.anchor_justification,
        }

    def validate(self) -> list[str]:
        """Structural invariants. Empty list = valid."""
        bad = []
        if self.spectrum.shape != (GRID_BINS,):
            bad.append(f"spectrum shape {self.spectrum.shape} != ({GRID_BINS},)")
        if not np.all(np.isfinite(self.spectrum)):
            bad.append("spectrum contains non-finite values")
        if np.any(self.spectrum < 0):
            bad.append("negative values — non-negativity is an architectural invariant")
        if float(self.spectrum.sum()) <= 0:
            bad.append("spectrum is all zero")
        if self.n_analytes != len(self.analytes):
            bad.append("n_analytes disagrees with the analyte list")
        if self.lsm_type not in LSM_TYPES:
            bad.append(f"unknown lsm_type {self.lsm_type!r}")
        for name, v in (("purity", self.purity), ("stability", self.stability),
                        ("activation_share", self.activation_share)):
            if not (0.0 <= v <= 1.0 + 1e-9):
                bad.append(f"{name} {v} outside [0, 1]")
        if not self.retained and not self.rejection_reason:
            bad.append("rejected LSM carries no rejection reason")
        if self.retained and self.rejection_reason:
            bad.append("retained LSM carries a rejection reason")
        if self.is_anchor and not self.anchor_justification:
            bad.append("anchored LSM carries no written chemical justification")
        if self.is_anchor and self.n_analytes != 1:
            bad.append("an anchor must declare exactly one supporting analyte")
        if not self.motif_id.startswith(f"{self.chemical_class}."):
            bad.append("motif_id does not encode its chemical class")
        return bad


def classify_type(activation: np.ndarray, threshold: float = 0.05,
                  shared_frac: float = 0.6, discriminating_frac: float = 0.25) -> str:
    """Type an LSM from its within-class activation pattern.

    `activation` is the LSM's column of `W_c`, normalised per molecule so the reading is
    "what share of this molecule's class-local evidence does this motif carry".
    """
    a = np.asarray(activation, float)
    on = float((a > threshold).mean()) if a.size else 0.0
    if on >= shared_frac:
        return "class_shared"
    if on <= discriminating_frac:
        return "molecule_discriminating"
    return "subfamily"


def dominant_bands(h: np.ndarray, grid: np.ndarray, prominence: float = 0.05,
                   max_bands: int = 8) -> list[dict]:
    """The diagnostic bands of a fitted LSM, strongest first."""
    from scipy.signal import find_peaks
    h = np.asarray(h, float)
    if float(h.max()) <= 0:
        return []
    peaks, props = find_peaks(h, prominence=prominence * float(h.max()))
    if len(peaks) == 0:
        p = int(np.argmax(h))
        return [{"center_cm": float(grid[p]), "prominence": float(h[p]),
                 "weight": 1.0}]
    order = np.argsort(-props["prominences"])[:max_bands]
    tot = float(props["prominences"][order].sum()) or 1.0
    out = [{"center_cm": float(grid[peaks[i]]),
            "prominence": float(props["prominences"][i]),
            "weight": float(props["prominences"][i] / tot)} for i in order]
    return sorted(out, key=lambda b: b["center_cm"])
