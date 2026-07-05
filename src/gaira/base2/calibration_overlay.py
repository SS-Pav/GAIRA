"""Calibration overlay (regime layer) helpers for gaira_base_2.

In the v1 engine, calibration behaviour is already baked into each
motif's `regime_weight` via `calibration_weight` (resolved from the
M2.2 dual-status table in ``motif_engine.compute_motif_score``).

This module exists as the future home of:
  * multi-substrate regime overlays (`calibration_weight_ag`,
    `calibration_weight_au`, `calibration_weight_paper`, ...)
  * regime switching — select which substrate's calibration weights
    to apply at inference time
  * per-substrate conflict/insufficient flags flowing into the
    ambiguity lane (substrate_physics v1.1.2 conflicting channel)

For v1 only Ag-colloid serum calibration (from M4) exists, so the
module provides a single `REGIME_AG_COLLOID_SERUM` constant and a
noop switcher. Adding a new regime is adding a new constant here plus
a calibration table loader; no core engine changes are required.
"""
from __future__ import annotations

from dataclasses import dataclass


REGIME_AG_COLLOID_SERUM: str = "ag_colloid_serum"
DEFAULT_REGIME: str = REGIME_AG_COLLOID_SERUM
SUPPORTED_REGIMES: tuple[str, ...] = (REGIME_AG_COLLOID_SERUM,)


@dataclass(frozen=True)
class Regime:
    """Descriptor for a calibration regime."""
    name: str
    substrate: str
    matrix: str
    source_phase: str   # e.g. "M4_calibration_validation_v1"


AG_COLLOID_SERUM = Regime(
    name=REGIME_AG_COLLOID_SERUM,
    substrate="Ag colloid (Lee-Meisel, citrate-reduced)",
    matrix="Merck commercial serum + physiological-conc spikes",
    source_phase="M4_calibration_validation_v1 + M4_1_refinement_and_recalibration_v1",
)


def get_active_regime(name: str = DEFAULT_REGIME) -> Regime:
    if name != REGIME_AG_COLLOID_SERUM:
        raise ValueError(
            f"gaira_base_2 v1 only supports regime "
            f"'{REGIME_AG_COLLOID_SERUM}'; got '{name}'. "
            f"Add a new Regime constant to extend."
        )
    return AG_COLLOID_SERUM
