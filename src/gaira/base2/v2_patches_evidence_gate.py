"""gaira_base_2 engine evidence-gating repair (v_gatefix).

ADDITIVE wrapper around gaira_base_2.motif_engine. Provides stricter
band-firing logic + gated motif activation that:

  1. Raises absolute band-firing floor from BAND_FLOOR=1e-3 to 0.005
  2. Adds local-prominence check (peak in band must be N× local median
     in a wider neighborhood) — prevents bands from passing on noise that
     is uniformly elevated in a region
  3. Adds a relative-to-spectrum-max gate (peak must be at least P fraction
     of the spectrum's overall max) — prevents bands from passing on
     near-zero relative signal

The gated `compute_motif_activation_gated()` matches the original API and
behavior except the band-firing tests are stricter. The driver applies
this via runtime monkey-patch of `gaira.base2.motif_engine.compute_motif_activation`
so that the rescue → discriminative → rankfix → packet chain all use
the gated activation without modifying any downstream module.

Engine modules touched: NONE on disk. The monkey-patch is applied in
driver scope only.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from gaira.base2.schema import ALPHA_SUPPORTING, BandFamily, MotifSpec


# ─────────────────────────────────────────────────────────────────────
# Gating constants (final values — locked after design comparison)
# ─────────────────────────────────────────────────────────────────────

# Raised from BAND_FLOOR=1e-3 to filter out the bottom of the noise floor
ABSOLUTE_FLOOR_GATED: float = 0.005

# Local-prominence: peak in band must be at least this multiple of the
# local-neighborhood median. 1.30 means "30% above the local baseline".
PROMINENCE_FACTOR: float = 1.30

# Half-width of the local neighborhood (cm-1) on EACH side of the band
# window. So total neighborhood = band_window + 2*PROMINENCE_HALFWIDTH.
PROMINENCE_HALFWIDTH_CM1: float = 40.0

# Relative-to-spectrum: peak in band must be at least this fraction of
# the spectrum's overall max intensity. Filters bands that are nonzero
# but tiny in absolute scale.
RELATIVE_TO_SPECTRUM_MIN: float = 0.05  # 5% of spectrum max


# Per-motif gating overrides. For chemistry families where the global
# gate filters out genuine fires (e.g. pyrimidine bands tend to be
# narrow with smaller absolute intensity), allow per-motif relaxation.
# Empty by default; populated by drivers that need closure-pass overrides.
MOTIF_GATE_OVERRIDES: dict[str, dict[str, float]] = {}


# ─────────────────────────────────────────────────────────────────────
# Stricter band-firing test
# ─────────────────────────────────────────────────────────────────────

def _spectrum_max(spectrum: np.ndarray) -> float:
    if spectrum.size == 0:
        return 0.0
    fin = np.isfinite(spectrum)
    if not fin.any():
        return 0.0
    return float(np.max(spectrum[fin]))


def band_fires_gated(
    fam: BandFamily, spectrum: np.ndarray, master_x: np.ndarray,
    absolute_floor: float | None = None,
    prominence_factor: float | None = None,
    prominence_halfwidth_cm1: float | None = None,
    relative_to_spectrum_min: float | None = None,
    spectrum_max: float | None = None,
) -> bool:
    """Stricter band-firing: passes only if all of:

      (1) band peak >= absolute_floor
      (2) band peak >= prominence_factor × local-baseline (median in
          neighborhood window excluding the band itself)
      (3) band peak >= relative_to_spectrum_min × spectrum_max

    When called with None for any threshold, uses the *current* module-level
    constant (so design comparisons that override these constants take
    effect inside the function call).
    """
    if absolute_floor is None:
        absolute_floor = ABSOLUTE_FLOOR_GATED
    if prominence_factor is None:
        prominence_factor = PROMINENCE_FACTOR
    if prominence_halfwidth_cm1 is None:
        prominence_halfwidth_cm1 = PROMINENCE_HALFWIDTH_CM1
    if relative_to_spectrum_min is None:
        relative_to_spectrum_min = RELATIVE_TO_SPECTRUM_MIN
    mask_band = (master_x >= fam.cm1_low) & (master_x <= fam.cm1_high)
    if not mask_band.any():
        return False
    band_vals = spectrum[mask_band]
    fin = np.isfinite(band_vals)
    if not fin.any():
        return False
    band_max = float(np.max(band_vals[fin]))

    # (1) absolute floor
    if band_max < absolute_floor:
        return False

    # (2) local prominence
    nbhd_low  = fam.cm1_low - prominence_halfwidth_cm1
    nbhd_high = fam.cm1_high + prominence_halfwidth_cm1
    mask_nbhd = (master_x >= nbhd_low) & (master_x <= nbhd_high) & ~mask_band
    if mask_nbhd.any():
        nbhd_vals = spectrum[mask_nbhd]
        nbhd_fin = np.isfinite(nbhd_vals)
        if nbhd_fin.any():
            nbhd_baseline = float(np.median(nbhd_vals[nbhd_fin]))
            if band_max < prominence_factor * max(nbhd_baseline, 1e-9):
                return False

    # (3) relative to spectrum max
    sp_max = spectrum_max if spectrum_max is not None else _spectrum_max(spectrum)
    if sp_max > 0 and band_max < relative_to_spectrum_min * sp_max:
        return False

    return True


def band_intensity_gated(
    fam: BandFamily, spectrum: np.ndarray, master_x: np.ndarray,
    spectrum_max: float | None = None,
    absolute_floor: float | None = None,
    prominence_factor: float | None = None,
    relative_to_spectrum_min: float | None = None,
) -> float:
    """Return band intensity if the band passes the gate; else 0.

    Used inside `compute_motif_activation_gated` so that motif activation
    cannot be inflated by noise-level intensities even after the REQUIRED
    co-band check passes.
    """
    if not band_fires_gated(fam, spectrum, master_x,
                              spectrum_max=spectrum_max,
                              absolute_floor=absolute_floor,
                              prominence_factor=prominence_factor,
                              relative_to_spectrum_min=relative_to_spectrum_min):
        return 0.0
    mask = (master_x >= fam.cm1_low) & (master_x <= fam.cm1_high)
    if not mask.any():
        return 0.0
    vals = spectrum[mask]
    fin = np.isfinite(vals)
    if not fin.any():
        return 0.0
    return float(np.max(vals[fin]))


# ─────────────────────────────────────────────────────────────────────
# Gated motif activation — matches the original API of
# `motif_engine.compute_motif_activation`. The driver monkey-patches
# this in place of the original so rescue/discriminative/rankfix/packet
# pick it up automatically.
# ─────────────────────────────────────────────────────────────────────

def compute_motif_activation_gated(
    motif: MotifSpec, spectrum: np.ndarray, master_x: np.ndarray,
    floor: float | None = None,
) -> float:
    """Mean-normalised motif activation with stricter band gating.

    Replaces `gaira.base2.motif_engine.compute_motif_activation` via
    runtime monkey-patch. Same return-value semantics, stricter
    admission.

    Honors `MOTIF_GATE_OVERRIDES` for per-motif gate relaxation:
    {motif_id: {"absolute_floor": ..., "prominence_factor": ...,
                "relative_to_spectrum_min": ...}}
    """
    if not motif.primary_bands:
        return 0.0

    # Pre-compute spectrum max once (reused for all band tests)
    sp_max = _spectrum_max(spectrum)

    # Apply per-motif gate overrides if any
    overrides = MOTIF_GATE_OVERRIDES.get(motif.motif_id, {})
    gate_kwargs = {"spectrum_max": sp_max}
    if "absolute_floor" in overrides:
        gate_kwargs["absolute_floor"] = overrides["absolute_floor"]
    if "prominence_factor" in overrides:
        gate_kwargs["prominence_factor"] = overrides["prominence_factor"]
    if "relative_to_spectrum_min" in overrides:
        gate_kwargs["relative_to_spectrum_min"] = overrides["relative_to_spectrum_min"]

    # Co-band gating (stricter)
    if motif.co_band_requirement == "REQUIRED":
        if not all(
            band_fires_gated(b, spectrum, master_x, **gate_kwargs)
            for b in motif.primary_bands
        ):
            return 0.0

    p_intensities = [
        band_intensity_gated(b, spectrum, master_x, **gate_kwargs)
        for b in motif.primary_bands
    ]
    primary_mean = float(np.mean(p_intensities))

    supporting_mean = 0.0
    if motif.supporting_bands:
        s_intensities = [
            band_intensity_gated(b, spectrum, master_x, **gate_kwargs)
            for b in motif.supporting_bands
        ]
        supporting_mean = float(np.mean(s_intensities))

    return primary_mean + ALPHA_SUPPORTING * supporting_mean


# ─────────────────────────────────────────────────────────────────────
# Runtime-patch helper
# ─────────────────────────────────────────────────────────────────────

def install_gated_activation():
    """Monkey-patch gaira.base2.motif_engine.compute_motif_activation
    to use the gated version. Idempotent. Returns the original function
    for restoration if needed.

    All downstream callers (rescue, discriminative, rankfix, packet)
    perform `from gaira.base2.motif_engine import compute_motif_activation`
    INSIDE their scoring functions, so the patch takes effect on the
    next call without rewiring imports.
    """
    import gaira.base2.motif_engine as _me
    original = getattr(_me, "_original_compute_motif_activation",
                       _me.compute_motif_activation)
    _me._original_compute_motif_activation = original
    _me.compute_motif_activation = compute_motif_activation_gated
    return original


def restore_original_activation():
    import gaira.base2.motif_engine as _me
    if hasattr(_me, "_original_compute_motif_activation"):
        _me.compute_motif_activation = _me._original_compute_motif_activation


__all__ = [
    "ABSOLUTE_FLOOR_GATED", "PROMINENCE_FACTOR",
    "PROMINENCE_HALFWIDTH_CM1", "RELATIVE_TO_SPECTRUM_MIN",
    "MOTIF_GATE_OVERRIDES",
    "band_fires_gated", "band_intensity_gated",
    "compute_motif_activation_gated",
    "install_gated_activation", "restore_original_activation",
]
