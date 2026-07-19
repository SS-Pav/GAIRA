"""Stage B0 — derivative representations (re-exported; must follow denoising)."""
from __future__ import annotations
from .normalization import derivative, concat_intensity_derivative  # noqa: F401

ORDERS = [0, 1, 2, "concat"]


def apply_derivative(Y, spec):
    if spec == "concat":
        return concat_intensity_derivative(Y)
    return derivative(Y, int(spec))
