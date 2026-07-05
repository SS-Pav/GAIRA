"""Motif activation + weight computation for gaira_base_2.

Implements the locked scoring pipeline from
``gaira_base_2_scoring_pressure_test_v1.md`` §7:

    primary_mean    = mean(band_intensity(f) for f in primary)
    supporting_mean = mean(band_intensity(f) for f in supporting)
    raw_activation  = primary_mean + 0.3 × supporting_mean

    if co_band_requirement == REQUIRED and any primary band fails floor:
        motif_activation = 0
    else:
        motif_activation = raw_activation

    core_weight   = clip(activation × status_core_weight × mapping_weight,   0, 1)
    regime_weight = clip(core_weight × calibration_weight,                    0, 1)

Ambiguity flagging: a motif contributes to the ambiguity lane iff
(a) its mapping_type is AMBIGUITY_ONLY, or
(b) its motif_type is AMBIGUITY_MOTIF or ARTIFACT_MOTIF, or
(c) it is a PRIMARY motif that is "demoted" by an exclusion condition
    (handled by the axis_engine through the exclusion-resolution pass).
"""
from __future__ import annotations

import numpy as np

from gaira.base2.primitives import band_fires, band_intensity
from gaira.base2.schema import (
    ALPHA_SUPPORTING,
    AxisMapping,
    BAND_FLOOR,
    CALIBRATION_WEIGHT_BY_STATUS,
    CORE_WEIGHT_BY_STATUS,
    MAPPING_WEIGHT_BY_TYPE,
    MotifDualStatus,
    MotifScore,
    MotifSpec,
)


# ──────────────────────────────────────────────────────────────────────
# Activation
# ──────────────────────────────────────────────────────────────────────

def compute_motif_activation(
    motif: MotifSpec, spectrum: np.ndarray, master_x: np.ndarray,
    floor: float = BAND_FLOOR,
) -> float:
    """Mean-normalised motif activation (pressure-test formula)."""
    if not motif.primary_bands:
        return 0.0

    # Co-band gating
    if motif.co_band_requirement == "REQUIRED":
        if not all(band_fires(f, spectrum, master_x, floor) for f in motif.primary_bands):
            return 0.0

    p_intensities = [band_intensity(f, spectrum, master_x) for f in motif.primary_bands]
    primary_mean = float(np.mean(p_intensities))

    supporting_mean = 0.0
    if motif.supporting_bands:
        s_intensities = [
            band_intensity(f, spectrum, master_x) for f in motif.supporting_bands
        ]
        supporting_mean = float(np.mean(s_intensities))

    return primary_mean + ALPHA_SUPPORTING * supporting_mean


# ──────────────────────────────────────────────────────────────────────
# Weights
# ──────────────────────────────────────────────────────────────────────

def resolve_status_core_weight(status: MotifDualStatus | None) -> float:
    if status is None:
        return 0.0
    return CORE_WEIGHT_BY_STATUS.get(status.core_status, 0.0)


def resolve_status_calibration_weight(status: MotifDualStatus | None) -> float:
    if status is None:
        return CALIBRATION_WEIGHT_BY_STATUS["NOT_RUN"]
    return CALIBRATION_WEIGHT_BY_STATUS.get(
        status.calibration_status,
        CALIBRATION_WEIGHT_BY_STATUS["NOT_RUN"],
    )


def resolve_mapping_weight(mapping: AxisMapping, target_axis: str) -> float:
    """Mapping_weight for this motif's contribution to target_axis.

    Rules:
      PRIMARY:         w = 1.00 on primary_axis;  w = 0 elsewhere
      SECONDARY:       w = 0.50 on primary_axis;  w = 0 elsewhere
      CROSS_AXIS:      w = 0.70 on primary_axis and each secondary_axis
      AMBIGUITY_ONLY:  w = 1.00 on the ambiguity lane; w = 0 on biology axes
    """
    if not mapping.active:
        return 0.0
    mt = mapping.mapping_type
    if mt == "AMBIGUITY_ONLY":
        return 0.0  # biology axes get zero; ambiguity handled in axis_engine
    if target_axis == mapping.primary_axis:
        if mt == "PRIMARY":
            return MAPPING_WEIGHT_BY_TYPE["PRIMARY"]
        if mt == "SECONDARY":
            return MAPPING_WEIGHT_BY_TYPE["SECONDARY"]
        if mt == "CROSS_AXIS":
            return MAPPING_WEIGHT_BY_TYPE["CROSS_AXIS"]
    if mt == "CROSS_AXIS" and target_axis in mapping.secondary_axes:
        return MAPPING_WEIGHT_BY_TYPE["CROSS_AXIS"]
    return 0.0


def motif_belongs_to_ambiguity_lane(
    motif: MotifSpec, mapping: AxisMapping | None,
) -> bool:
    """Does this motif contribute to the ambiguity_artifact lane?"""
    if mapping is None or not mapping.active:
        # HELD_V2 motifs do not contribute to the lane either
        return False
    if mapping.mapping_type == "AMBIGUITY_ONLY":
        return True
    if motif.motif_type in ("AMBIGUITY_MOTIF", "ARTIFACT_MOTIF"):
        return True
    # CROSS_AXIS motifs whose secondary axis includes ambiguity_artifact
    if "ambiguity_artifact" in (mapping.primary_axis, *mapping.secondary_axes):
        return True
    return False


def compute_motif_score(
    motif: MotifSpec,
    mapping: AxisMapping | None,
    status: MotifDualStatus | None,
    spectrum: np.ndarray,
    master_x: np.ndarray,
) -> MotifScore:
    """Compute activation + core/regime weights for a single motif.

    This produces per-motif primary weights (independent of target axis).
    The axis_engine will multiply by mapping_weight(axis) inside its
    noisy-OR combiner. Here we return the 'self weight' = activation ×
    core_status_weight, which the axis engine combines with mapping_weight.

    Returned core_weight and regime_weight are clipped to [0, 1].
    """
    activation = compute_motif_activation(motif, spectrum, master_x)
    core_status_w = resolve_status_core_weight(status)
    # Per-motif "self" contribution without target-axis mapping. Will be
    # multiplied by mapping_weight per-axis in the axis engine.
    self_core = float(np.clip(activation * core_status_w, 0.0, 1.0))
    cal_w = resolve_status_calibration_weight(status)
    self_regime = float(np.clip(self_core * cal_w, 0.0, 1.0))
    ambig = motif_belongs_to_ambiguity_lane(motif, mapping)
    return MotifScore(
        motif_id=motif.motif_id,
        activation=float(activation),
        core_weight=self_core,
        regime_weight=self_regime,
        contributes_to_ambiguity=ambig,
    )
