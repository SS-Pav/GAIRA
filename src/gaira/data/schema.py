"""GAIRA V5 — canonical spectrum data contract (Phase 0 / V5.0).

Defines the per-spectrum metadata contract every included spectrum must satisfy
before it may enter representation analysis. Read-only; no scoring, no fitting.

This is the governance layer: a spectrum without sufficient provenance +
acquisition-domain metadata is NOT admitted to joint analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

import numpy as np


class Modality(str, Enum):
    RAMAN = "raman"                 # spontaneous Raman
    SERS = "sers"                   # surface-enhanced
    UNKNOWN = "unknown"


class SubstrateGeometry(str, Enum):
    NONE = "none"                   # powder / neat / CaF2 slide (spontaneous)
    COLLOID = "colloid"             # nanoparticle colloid
    PLANAR = "planar"               # planar / film
    ORC = "orc_roughened"           # electrochemically roughened
    UNKNOWN = "unknown"


class IntendedRole(str, Enum):
    GROUNDING = "grounding"                       # pure-analyte reference
    PERTURBATION_EVAL = "controlled_perturbation" # held-out challenge
    BIOLOGICAL_CHALLENGE = "biological_challenge"  # mixtures / cohorts
    PEAK_EVIDENCE = "peak_evidence"                # peak tables only (no spectrum)


@dataclass(frozen=True)
class SpectrumRecord:
    """One measured (or peak-level) reference observation with full provenance."""
    spectrum_id: str
    analyte_id: str
    canonical_analyte_name: str
    source_dataset: str
    raw_path: str
    modality: Modality
    sers_or_raman: str
    substrate_material: str
    substrate_geometry: SubstrateGeometry
    colloid_or_planar: str
    excitation_nm: Optional[float]
    instrument: str
    matrix: str                      # aqueous / powder / serum / ...
    concentration: str
    replicate: str
    independent_measurement: bool
    raw_or_processed: str
    wavenumber_min: Optional[float]
    wavenumber_max: Optional[float]
    point_count: Optional[int]
    quality_flags: str
    intended_role: IntendedRole
    allowed_for_representation_learning: bool
    allowed_for_evaluation_only: bool
    # acquisition-domain key = (modality, substrate_geometry, excitation) — the
    # unit across which comparability must be established before joint analysis.
    acquisition_domain: str = ""

    def to_row(self) -> dict:
        d = asdict(self)
        for k in ("modality", "substrate_geometry", "intended_role"):
            d[k] = d[k].value if hasattr(d[k], "value") else d[k]
        return d


@dataclass
class Spectrum:
    """A loaded spectrum: record metadata + numeric arrays. Arrays are None for
    peak-only evidence (PEAK_EVIDENCE role)."""
    record: SpectrumRecord
    wavenumber: Optional[np.ndarray] = None
    intensity: Optional[np.ndarray] = None
    peaks: Optional[list] = None      # list[(cm1, intensity_code, exclusive)] for peak-only

    @property
    def has_spectrum(self) -> bool:
        return self.wavenumber is not None and self.intensity is not None


# ── Admission gate (Phase 0) ──────────────────────────────────────────
REQUIRED_METADATA = (
    "spectrum_id", "canonical_analyte_name", "source_dataset", "modality",
    "substrate_geometry", "intended_role",
)


def admits_to_joint_analysis(rec: SpectrumRecord) -> tuple[bool, str]:
    """Governance gate: may this spectrum enter representation/comparability
    analysis? Returns (ok, reason). Peak-only and non-grounding roles are
    excluded from full-spectrum joint analysis by construction."""
    missing = [f for f in REQUIRED_METADATA if not getattr(rec, f, None)]
    if missing:
        return False, f"missing metadata: {missing}"
    if rec.intended_role == IntendedRole.PEAK_EVIDENCE:
        return False, "peak-only evidence (no full spectrum) — excluded from full-spectrum joint analysis"
    if rec.intended_role != IntendedRole.GROUNDING:
        return False, f"role={rec.intended_role.value} — not molecular grounding"
    if rec.modality == Modality.UNKNOWN or rec.substrate_geometry == SubstrateGeometry.UNKNOWN:
        return False, "unknown modality/substrate — cannot assign acquisition domain"
    if not rec.allowed_for_representation_learning:
        return False, "flagged not-allowed for representation learning"
    return True, "ok"
