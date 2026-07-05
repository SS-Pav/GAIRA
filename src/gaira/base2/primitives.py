"""Band / primitive layer for gaira_base_2.

Computes per-band intensity on a preprocessed spectrum. Stateless;
no motif or axis logic here.
"""
from __future__ import annotations

import numpy as np

from gaira.base2.schema import BAND_FLOOR, BandFamily


def band_intensity(fam: BandFamily, spectrum: np.ndarray, master_x: np.ndarray) -> float:
    """Peak height within the band window, clipped at 0.

    Uses MAX within the window rather than integral — gives a value in
    roughly [0, 1] for an L2-normalised spectrum and matches the M3
    grounding evaluator's local-max logic.
    """
    mask = (master_x >= fam.cm1_low) & (master_x <= fam.cm1_high)
    if not mask.any():
        return 0.0
    y_win = spectrum[mask]
    fin = np.isfinite(y_win)
    if not fin.any():
        return 0.0
    val = float(np.max(y_win[fin]))
    return max(val, 0.0)


def band_fires(fam: BandFamily, spectrum: np.ndarray, master_x: np.ndarray,
               floor: float = BAND_FLOOR) -> bool:
    """Whether a band fires above the floor."""
    return band_intensity(fam, spectrum, master_x) > floor
