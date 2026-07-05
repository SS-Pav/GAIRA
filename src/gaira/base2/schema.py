"""Schema / dataclasses for gaira_base_2.

These are pure data containers. No logic lives here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Constants (locked from architecture docs)
# ──────────────────────────────────────────────────────────────────────

# 11 biology axes from axis design doc §2
BIOLOGY_AXES_V11: tuple[str, ...] = (
    "purine_nucleotide",
    "purine_metabolite",
    "pyrimidine_nucleotide",
    "phosphate_nucleic_adjacent",
    "glycan_carbohydrate",
    "protein_peptide_backbone",
    "aromatic_residue",
    "lipid_acyl_membrane",
    "sterol_neutral_lipid",
    "sulfur_thiol_redox",
    "metabolic_small_molecule",
)

# 1 ambiguity/artifact control lane
CONTROL_LANE: str = "ambiguity_artifact"

# gaira_base 8 axes (for backward-compat projection)
GAIRA_BASE_AXES_V8: tuple[str, ...] = (
    "membrane_lipid",
    "protein_backbone",
    "aromatic_amino_acid",
    "purine_nucleotide",
    "pyrimidine_nucleotide",
    "glycan_carbohydrate",
    "redox_metabolite",
    "nucleic_acid_backbone",
)

# 11-axis → 8-axis projection map (from axis design doc §5)
PROJECTION_V11_TO_V8: dict[str, tuple[str, ...]] = {
    "purine_nucleotide":        ("purine_nucleotide", "purine_metabolite"),
    "pyrimidine_nucleotide":    ("pyrimidine_nucleotide",),
    "nucleic_acid_backbone":    ("phosphate_nucleic_adjacent",),
    "glycan_carbohydrate":      ("glycan_carbohydrate",),
    "protein_backbone":         ("protein_peptide_backbone",),
    "aromatic_amino_acid":      ("aromatic_residue",),
    "membrane_lipid":           ("lipid_acyl_membrane", "sterol_neutral_lipid"),
    "redox_metabolite":         ("sulfur_thiol_redox", "metabolic_small_molecule"),
}

# Supporting-band multiplier (locked for v1 per scoring pressure test §4)
ALPHA_SUPPORTING: float = 0.3

# Floor for "band fires above noise" test (consistent with M3/M4)
BAND_FLOOR: float = 1e-3

# core_status → core_weight (from scoring architecture §3.3)
CORE_WEIGHT_BY_STATUS: dict[str, float] = {
    "CORE_GROUNDED":             1.00,
    "CORE_PARTIALLY_GROUNDED":   0.70,
    "CORE_AMBIGUITY_CONFIRMED":  0.80,
    "CORE_WEAK":                 0.40,
    "CORE_NOT_SUPPORTED":        0.00,
    "CORE_AMBIGUITY_WEAK":       0.20,
}

# calibration_status → calibration_weight (from scoring architecture §3.3)
CALIBRATION_WEIGHT_BY_STATUS: dict[str, float] = {
    "CALIBRATION_VALID":   1.00,
    "PARTIALLY_VALID":     0.80,
    "CONTEXT_ONLY":        0.50,
    "UNRELIABLE":          0.30,
    "NOT_RUN":             0.80,   # no substrate-specific data → no penalty
}

# mapping_type → mapping_weight
MAPPING_WEIGHT_BY_TYPE: dict[str, float] = {
    "PRIMARY":          1.00,
    "SECONDARY":        0.50,
    "CROSS_AXIS":       0.70,   # per target axis
    "AMBIGUITY_ONLY":   1.00,   # to ambiguity lane only; biology axes get 0
}


# ──────────────────────────────────────────────────────────────────────
# Band / primitive
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BandFamily:
    """A spectral band window from a motif definition."""
    family_id: str
    cm1_centre: float
    cm1_tolerance: float
    role: str            # "primary" | "supporting"
    vibrational_origin: str = ""

    @property
    def cm1_low(self) -> float:
        return self.cm1_centre - self.cm1_tolerance

    @property
    def cm1_high(self) -> float:
        return self.cm1_centre + self.cm1_tolerance


# ──────────────────────────────────────────────────────────────────────
# Motif specification + state
# ──────────────────────────────────────────────────────────────────────

@dataclass
class MotifSpec:
    """A motif loaded from the v1.2 registry, ready for scoring."""
    motif_id: str
    motif_family: str
    motif_type: str
    primary_bands: tuple[BandFamily, ...]
    supporting_bands: tuple[BandFamily, ...]
    co_band_requirement: str       # "REQUIRED" | "SUPPORTING"
    v1_active: bool
    ambiguity_class: str = ""
    exclusion_conditions: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class AxisMapping:
    """How a motif contributes to biology axes + ambiguity lane."""
    motif_id: str
    primary_axis: str
    secondary_axes: tuple[str, ...]
    mapping_type: str              # PRIMARY / SECONDARY / CROSS_AXIS / AMBIGUITY_ONLY
    # "active" = this mapping contributes weight > 0 in v1 scoring.
    # HELD_V2 motifs have active=False.
    active: bool


@dataclass(frozen=True)
class MotifDualStatus:
    """Dual-status row from M2.2 dual-status table."""
    motif_id: str
    core_status: str
    calibration_status: str
    final_v1_role: str


# ──────────────────────────────────────────────────────────────────────
# Output dataclasses
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MotifScore:
    """Per-motif output for one spectrum."""
    motif_id: str
    activation: float              # raw mean-normalised activation
    core_weight: float             # activation × core × mapping  (clipped [0,1])
    regime_weight: float           # core_weight × calibration     (clipped [0,1])
    contributes_to_ambiguity: bool


@dataclass(frozen=True)
class AxisScore:
    """Per-axis output (11-axis or 8-axis projection, one spectrum)."""
    axis_id: str
    core_evidence: float           # 11-axis: noisy-OR core weights;
                                    # 8-axis: MAX over contributing 11-axes core
    regime_evidence: float
    contributing_motifs: tuple[str, ...]


@dataclass(frozen=True)
class AmbiguityLane:
    """Ambiguity control-lane output."""
    core_evidence: float
    regime_evidence: float
    contributing_motifs: tuple[str, ...]


@dataclass(frozen=True)
class SpectrumResult:
    """Full gaira_base_2 output for one spectrum."""
    spectrum_id: str
    motif_scores: tuple[MotifScore, ...]
    axis11_scores: tuple[AxisScore, ...]
    axis8_projection: tuple[AxisScore, ...]
    ambiguity: AmbiguityLane
    metadata: dict = field(default_factory=dict)
