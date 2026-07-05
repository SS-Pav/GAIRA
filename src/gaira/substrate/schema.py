"""GAIRA Substrate Engine v1.1.1 — typed dataclasses (conflict-aware).

Mirrors the substrate-physics schema spec
(`/Volumes/SSD_Rad/GAIRA_BUILD/substrate_physics_v1/docs/substrate_physics_schema_v1.md`).

All dataclasses are frozen / deterministic. The engine never mutates a
loaded schema object after construction.

v1.1.1 patch:
  - `EffectChannel`: explicit "weighted" / "conflicting" / "insufficient" tag
  - `ResolvedEffect.effect_channel` and `weighting_applied` new fields
  - `ComposedOverlay` gains conflict-aware channels and flags
  - `ConflictReport`: structured per-overlay conflict summary
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


TargetLevel = Literal["band", "band_family", "motif", "axis", "global"]
ConvergenceStatus = Literal["CONVERGED", "EMERGING", "CONFLICTING", "INSUFFICIENT"]
EvidenceConfidence = Literal["strong", "moderate", "weak", "speculative"]

# v1.1.1: explicit channel for evidence routing.
EffectChannel = Literal["weighted", "conflicting", "insufficient"]


# ──────────────────────────────────────────────────────────────────────
# Static controlled-vocabulary objects
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SubstrateFamily:
    id: str
    display: str
    metal: str
    geometry_class: str
    fabrication_class: str
    biofluid_use_common: bool
    typical_excitation_nm: tuple[int, ...]
    typical_capping_or_surface_chemistry: tuple[str, ...]
    known_strengths: tuple[str, ...]
    known_weaknesses: tuple[str, ...]
    notes: str


@dataclass(frozen=True)
class SubstrateEffectType:
    id: str
    display: str
    semantics: str
    confidence_multiplier: float
    caution_flag: bool
    bias_direction: str            # "towards_enhancement" | "towards_suppression" | "none"
    interpretation_note: str


# ──────────────────────────────────────────────────────────────────────
# Seed-corpus entry
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EffectTarget:
    level: TargetLevel                       # band / band_family / motif / axis / global
    axis: str | None = None                  # one of BSV_COMPONENTS or "all"
    window_id: str | None = None             # atlas window_id, if applicable
    cm1_range: tuple[float, float] | None = None
    band_family: str | None = None           # free-text label (e.g. "nucleobase_ring")


@dataclass(frozen=True)
class SubstrateBandEffect:
    id: str
    substrate_family: str
    target: EffectTarget
    effect_type: str
    evidence_confidence: EvidenceConfidence
    convergence_status: ConvergenceStatus | None
    biochemical_target_class: str | None
    spectral_region_description: str | None
    evidence_types: tuple[str, ...]
    confidence_multiplier: float | None
    provenance_sources: tuple[str, ...]
    provenance_notes: str
    summary_of_effect: str
    notes: str

    def bias_direction(self, effect_types: dict[str, SubstrateEffectType]) -> str:
        et = effect_types.get(self.effect_type)
        return et.bias_direction if et else "none"


# ──────────────────────────────────────────────────────────────────────
# Engine outputs
# ──────────────────────────────────────────────────────────────────────

VisibilityTag = Literal[
    "enhanced",
    "suppressed",
    "biased",
    "variable",
    "non_biological",
    "uncertain",
    "neutral",
]

AbundanceTag = Literal[
    "may_underestimate_abundance",
    "may_overestimate_abundance",
    "abundance_not_directly_inferable",
    "relatively_neutral_visibility",
]


@dataclass(frozen=True)
class ResolvedEffect:
    """A single seed-corpus effect resolved for a specific query.

    v1.1.1 adds two channel-routing fields:
      - `effect_channel`: "weighted" | "conflicting" | "insufficient"
      - `weighting_applied`: True iff this effect contributed to the
        composed multiplier
    """
    effect_id: str
    effect_type: str
    bias_direction: str
    convergence_status: ConvergenceStatus | None
    evidence_confidence: EvidenceConfidence
    confidence_multiplier: float
    caution_flag: bool
    target_level: TargetLevel
    interpretation_note: str           # canonical (from effect-type)
    effect_summary: str                # from the seed entry itself
    provenance_sources: tuple[str, ...]
    # v1.1.1 additions
    effect_channel: EffectChannel = "weighted"
    weighting_applied: bool = True


@dataclass(frozen=True)
class ConflictReport:
    """Structured per-overlay conflict summary (v1.1.1).

    Generated whenever any CONFLICTING evidence resolves for the query.
    Used by report_overlay to surface conflict language deterministically.
    """
    has_conflict: bool
    conflicting_effect_ids: tuple[str, ...]
    conflict_notes: tuple[str, ...]
    candidate_assignment_classes: tuple[str, ...]   # competing biochem labels
    spectral_regions: tuple[str, ...]               # region descriptions


@dataclass(frozen=True)
class ComposedOverlay:
    """Composite result of resolving + composing effects for a (family, target) query.

    v1.1.1 additions (additive, default values preserve old call sites):
      - `weighted_effects` / `conflicting_effects` / `insufficient_effects`
      - `conflict_flag`, `unresolved_assignment_flag`
      - `conflict_report`
      - `weighted_multiplier_input_ids`: ids of effects that contributed
        to the composed multiplier (audit trail).
    """
    substrate_family: str
    target: EffectTarget
    resolved_effects: tuple[ResolvedEffect, ...]

    composed_confidence_multiplier: float     # clamped [0.4, 1.15]
    caution: bool

    # Visibility-vs-abundance semantics (first-class per v1.1 requirement)
    observed_signal_visibility: VisibilityTag
    biological_abundance_interpretation: AbundanceTag
    user_facing_caveat_lines: tuple[str, ...]

    convergence_labels: tuple[str, ...]       # e.g. ("CONVERGED", "EMERGING")
    provenance_sources: tuple[str, ...]
    substrate_blind: bool = False             # True when family is `unknown`

    # ── v1.1.1 conflict-aware additions ───────────────────────────────
    weighted_effects: tuple[ResolvedEffect, ...] = ()
    conflicting_effects: tuple[ResolvedEffect, ...] = ()
    insufficient_effects: tuple[ResolvedEffect, ...] = ()
    conflict_flag: bool = False
    unresolved_assignment_flag: bool = False
    conflict_report: ConflictReport | None = None
    weighted_multiplier_input_ids: tuple[str, ...] = ()
