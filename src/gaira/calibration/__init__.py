"""GAIRA calibration evaluation — layer-1 validation on controlled perturbations.

Tests whether the direct spectral BSV pipeline recovers known biochemical
shifts when the chemistry is controlled (spiking, enzymatic depletion,
titration). Not a disease demo.
"""
from gaira.calibration.registry import (
    CALIBRATION_REGISTRY,
    CalibrationContrast,
    get_contrast,
    list_contrasts,
)
from gaira.calibration.eval import (
    CalibrationResult,
    run_calibration_eval,
)

__all__ = [
    "CALIBRATION_REGISTRY",
    "CalibrationContrast",
    "CalibrationResult",
    "get_contrast",
    "list_contrasts",
    "run_calibration_eval",
]
