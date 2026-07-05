"""GAIRA spectral query pipeline — dataset-grounded biochemical composition analysis.

Public helpers exposed for downstream motif work (M3+):

    from gaira.spectral import (
        CANONICAL_SUPPORT_CM1, CANONICAL_N_POINTS, CANONICAL_STEP_CM1,
        canonical_master_axis, CoverageInfo, InsufficientOverlapError,
        crop_before_interpolate, crop_and_interp_batch,
    )

These enforce the crop-before-interpolate rule. See
``docs/canonical_signal_support_rule_v1.md``.
"""
from gaira.spectral.crop_and_interp import (
    CANONICAL_SUPPORT_CM1,
    CANONICAL_N_POINTS,
    CANONICAL_STEP_CM1,
    CoverageInfo,
    InsufficientOverlapError,
    canonical_master_axis,
    crop_and_interp_batch,
    crop_before_interpolate,
)

__all__ = [
    "CANONICAL_SUPPORT_CM1",
    "CANONICAL_N_POINTS",
    "CANONICAL_STEP_CM1",
    "CoverageInfo",
    "InsufficientOverlapError",
    "canonical_master_axis",
    "crop_and_interp_batch",
    "crop_before_interpolate",
]
