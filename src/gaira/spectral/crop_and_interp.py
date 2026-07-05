"""Crop-before-interpolate helper (canonical preprocessing rule).

Enforces the deterministic rule:

    raw spectrum
      → crop to canonical support [400, 1800] cm-1
      → interpolate to master axis (1401 points, 1 cm-1 step)
      → (downstream) AsLS + SG + L2

This module exists to prevent silent extrapolation at spectrum boundaries
(the default behaviour of ``numpy.interp`` when the target axis extends
beyond the measured support), and to surface partial-coverage cases
explicitly rather than hiding them behind clamped boundary values.

Public surface:

    CANONICAL_SUPPORT_CM1   — (400.0, 1800.0)
    CANONICAL_N_POINTS       — 1401
    CANONICAL_STEP_CM1       — 1.0
    canonical_master_axis()  — returns the locked master axis
    CoverageInfo             — per-spectrum audit record
    InsufficientOverlapError — raised when a spectrum cannot be usefully used
    crop_before_interpolate()      — single-spectrum deterministic op
    crop_and_interp_batch()        — vectorised wrapper for (n_spectra, n_wn)

The helper does NOT apply AsLS / SG / L2 — those are downstream steps in
``gaira.spectral.preprocessing._preprocess_raw`` and operate on the
master-axis output of this helper.

No learned weights, no random state, deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Canonical analysis support (LOCKED — do not change without a policy bump)
# ──────────────────────────────────────────────────────────────────────

CANONICAL_SUPPORT_CM1: tuple[float, float] = (400.0, 1800.0)
CANONICAL_N_POINTS:    int   = 1401
CANONICAL_STEP_CM1:    float = 1.0


def canonical_master_axis() -> np.ndarray:
    """Return the locked canonical master axis (400-1800 cm-1, 1401 pts, 1 cm-1)."""
    lo, hi = CANONICAL_SUPPORT_CM1
    return np.linspace(lo, hi, CANONICAL_N_POINTS)


# ──────────────────────────────────────────────────────────────────────
# Error + audit types
# ──────────────────────────────────────────────────────────────────────

class InsufficientOverlapError(ValueError):
    """Raised when a raw spectrum's measured support does not overlap the
    canonical analysis support above the minimum-coverage threshold."""


@dataclass(frozen=True)
class CoverageInfo:
    """Per-spectrum audit record for the crop-before-interpolate step."""
    raw_min_cm1:            float
    raw_max_cm1:            float
    cropped_min_cm1:        float
    cropped_max_cm1:        float
    master_min_cm1:         float
    master_max_cm1:         float
    partial_coverage:       bool      # True if measured support < master axis span
    coverage_fraction:      float     # fraction of master_x inside measured support
    n_interpolated_points:  int       # count of master_x points with finite interp
    n_masked_out_of_support: int      # count of master_x points set to NaN


# ──────────────────────────────────────────────────────────────────────
# Public API — single-spectrum
# ──────────────────────────────────────────────────────────────────────

def crop_before_interpolate(
    raw_wn: np.ndarray,
    raw_y:  np.ndarray,
    master_x: np.ndarray,
    *,
    partial_ok: bool = True,
    min_coverage: float = 0.5,
) -> tuple[np.ndarray, CoverageInfo]:
    """Deterministic crop-then-interpolate for a single spectrum.

    Steps:
      1. Sort ``raw_wn`` ascending; reorder ``raw_y`` accordingly.
      2. Crop raw support to ``[master_x.min(), master_x.max()]``.
      3. Determine measured-support overlap with ``master_x``.
      4. If coverage_fraction < ``min_coverage`` → raise
         :class:`InsufficientOverlapError`.
      5. If ``partial_ok=False`` and the measured support is partial
         (does not fully span master_x) → raise
         :class:`InsufficientOverlapError`.
      6. Interpolate ``raw_y`` onto the master-axis points that lie inside
         the measured support.
      7. Set master-axis points outside the measured support to ``NaN``
         (no extrapolation; caller decides how to handle missing bands).

    Parameters
    ----------
    raw_wn : (N,) array
        Raw wavenumber axis in any order, any range, any units consistent
        with ``master_x``.
    raw_y : (N,) array
        Raw intensities aligned index-wise with ``raw_wn``.
    master_x : (M,) array
        Target canonical master axis. Must be monotone increasing.
    partial_ok : bool, default True
        If False, partial coverage above ``min_coverage`` still raises.
    min_coverage : float in [0, 1], default 0.5
        Below this fraction of measured-support overlap, the spectrum is
        rejected with :class:`InsufficientOverlapError`.

    Returns
    -------
    y_interp : (M,) array
        Interpolated intensities on ``master_x``. ``NaN`` outside the
        measured support (never extrapolated).
    coverage : :class:`CoverageInfo`
        Audit record for this operation.

    Raises
    ------
    InsufficientOverlapError
        If the measured-support overlap with ``master_x`` is below
        ``min_coverage``, or if ``partial_ok=False`` and the measured
        support does not fully span ``master_x``.
    ValueError
        If inputs have mismatched shapes, ``master_x`` is not monotone
        increasing, or the raw arrays are empty.
    """
    raw_wn = np.asarray(raw_wn, dtype=np.float64)
    raw_y  = np.asarray(raw_y,  dtype=np.float64)
    master_x = np.asarray(master_x, dtype=np.float64)

    if raw_wn.ndim != 1 or raw_y.ndim != 1:
        raise ValueError("raw_wn and raw_y must be 1-D arrays")
    if raw_wn.shape != raw_y.shape:
        raise ValueError(
            f"raw_wn and raw_y must have the same length; got "
            f"{raw_wn.shape} and {raw_y.shape}"
        )
    if master_x.ndim != 1 or master_x.size < 2:
        raise ValueError("master_x must be a 1-D array of length >= 2")
    if not np.all(np.diff(master_x) > 0):
        raise ValueError("master_x must be strictly monotone increasing")
    if raw_wn.size == 0:
        raise ValueError("raw_wn is empty; no data to interpolate")

    # 1. sort raw_wn ascending (reorder raw_y accordingly)
    order = np.argsort(raw_wn, kind="stable")
    sorted_wn = raw_wn[order]
    sorted_y  = raw_y[order]

    # collapse exact duplicates (keep first occurrence; preserves determinism
    # without averaging because averaging would be a hidden decision).
    uniq_mask = np.concatenate(([True], np.diff(sorted_wn) > 0))
    sorted_wn = sorted_wn[uniq_mask]
    sorted_y  = sorted_y[uniq_mask]

    raw_min, raw_max = float(sorted_wn[0]), float(sorted_wn[-1])
    master_min, master_max = float(master_x[0]), float(master_x[-1])

    # 2. crop to canonical support (= [master_min, master_max])
    crop_mask = (sorted_wn >= master_min) & (sorted_wn <= master_max)
    cropped_wn = sorted_wn[crop_mask]
    cropped_y  = sorted_y[crop_mask]

    if cropped_wn.size == 0:
        raise InsufficientOverlapError(
            f"raw spectrum range [{raw_min}, {raw_max}] has no overlap with "
            f"master axis [{master_min}, {master_max}]"
        )

    cropped_min, cropped_max = float(cropped_wn[0]), float(cropped_wn[-1])

    # 3. determine in-support mask on master_x
    in_support = (master_x >= cropped_min) & (master_x <= cropped_max)
    n_in = int(in_support.sum())
    coverage_fraction = n_in / master_x.size
    partial = (cropped_min > master_min) or (cropped_max < master_max)

    # 4. coverage gate
    if coverage_fraction < min_coverage:
        raise InsufficientOverlapError(
            f"measured support [{cropped_min}, {cropped_max}] covers only "
            f"{coverage_fraction:.3f} of master axis [{master_min}, {master_max}]; "
            f"below threshold {min_coverage:.3f}"
        )
    if partial and not partial_ok:
        raise InsufficientOverlapError(
            f"measured support [{cropped_min}, {cropped_max}] does not fully "
            f"span master axis [{master_min}, {master_max}]; partial_ok=False"
        )

    # 5. interpolate on in-support points; NaN outside
    y_interp = np.full_like(master_x, np.nan, dtype=np.float64)
    if n_in > 0:
        # np.interp clamps at boundaries by default; by restricting the
        # output indices to in_support we ensure no extrapolation occurs.
        y_interp[in_support] = np.interp(master_x[in_support], cropped_wn, cropped_y)

    coverage = CoverageInfo(
        raw_min_cm1=raw_min,
        raw_max_cm1=raw_max,
        cropped_min_cm1=cropped_min,
        cropped_max_cm1=cropped_max,
        master_min_cm1=master_min,
        master_max_cm1=master_max,
        partial_coverage=partial,
        coverage_fraction=coverage_fraction,
        n_interpolated_points=n_in,
        n_masked_out_of_support=int(master_x.size - n_in),
    )
    return y_interp, coverage


# ──────────────────────────────────────────────────────────────────────
# Public API — batch wrapper
# ──────────────────────────────────────────────────────────────────────

def crop_and_interp_batch(
    raw_wn: np.ndarray,
    raw_X:  np.ndarray,
    master_x: np.ndarray,
    *,
    partial_ok: bool = True,
    min_coverage: float = 0.5,
) -> tuple[np.ndarray, list[CoverageInfo]]:
    """Vectorised wrapper for a single shared ``raw_wn`` with a batch of
    spectra in ``raw_X`` of shape ``(n_spectra, n_wn)``.

    Each spectrum is processed independently via
    :func:`crop_before_interpolate`; per-spectrum :class:`CoverageInfo`
    records are returned.

    The shared ``raw_wn`` case is the common one (e.g. the HCC holdout CSV
    where every spectrum is sampled on the same wavenumber grid). For
    per-spectrum wavenumber grids (e.g. the DuckDB row store used by
    Pilots 2b/3), call :func:`crop_before_interpolate` per spectrum
    directly.
    """
    raw_X = np.asarray(raw_X, dtype=np.float64)
    if raw_X.ndim != 2:
        raise ValueError("raw_X must be a 2-D array (n_spectra, n_wn)")
    if raw_X.shape[1] != np.asarray(raw_wn).size:
        raise ValueError(
            f"raw_X has {raw_X.shape[1]} columns but raw_wn has "
            f"{np.asarray(raw_wn).size} points"
        )
    n = raw_X.shape[0]
    master_x = np.asarray(master_x, dtype=np.float64)

    out = np.empty((n, master_x.size), dtype=np.float64)
    cov_list: list[CoverageInfo] = []
    for i in range(n):
        y_i, cov_i = crop_before_interpolate(
            raw_wn, raw_X[i], master_x,
            partial_ok=partial_ok, min_coverage=min_coverage,
        )
        out[i] = y_i
        cov_list.append(cov_i)
    return out, cov_list
