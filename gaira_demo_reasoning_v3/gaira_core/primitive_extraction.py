"""GAIRA Demo v1 — primitive extraction.

Detects peaks and computes simple primitives that feed the MSS / motif
scoring layer. This is a simplified extractor; production GAIRA uses
~50 primitives.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Peak:
    position: float
    intensity: float
    prominence: float
    width: float


def detect_peaks(wavenumber: np.ndarray, intensity: np.ndarray,
                  *, prominence_floor: float = 0.005, min_distance_cm1: float = 6.0) -> list[Peak]:
    """Detect peaks above a relative prominence floor."""
    try:
        from scipy.signal import find_peaks
        # Convert distance to indices
        if len(wavenumber) > 1:
            dx = float(np.median(np.diff(wavenumber)))
            min_dist = max(1, int(round(min_distance_cm1 / max(dx, 1e-6))))
        else:
            min_dist = 1
        peaks, props = find_peaks(intensity, prominence=prominence_floor, distance=min_dist)
        widths = _half_widths(intensity, peaks, props.get("prominences"))
        out = []
        for idx, p_idx in enumerate(peaks):
            out.append(Peak(
                position=float(wavenumber[p_idx]),
                intensity=float(intensity[p_idx]),
                prominence=float(props["prominences"][idx]),
                width=float(widths[idx]),
            ))
        return out
    except Exception:
        return _naive_peaks(wavenumber, intensity, prominence_floor)


def _half_widths(y: np.ndarray, peak_idx: np.ndarray, prominences) -> np.ndarray:
    out = np.full(len(peak_idx), 8.0)
    for i, p in enumerate(peak_idx):
        thr = y[p] - 0.5 * (prominences[i] if prominences is not None else 0.0)
        left = p
        while left > 0 and y[left] > thr:
            left -= 1
        right = p
        while right < len(y) - 1 and y[right] > thr:
            right += 1
        out[i] = max(2.0, right - left)
    return out


def _naive_peaks(wavenumber: np.ndarray, intensity: np.ndarray,
                  prominence_floor: float) -> list[Peak]:
    out = []
    for i in range(1, len(intensity) - 1):
        if intensity[i] > intensity[i - 1] and intensity[i] > intensity[i + 1]:
            prom = intensity[i] - min(intensity[max(0, i - 8): i + 9].min(), 0.0)
            if prom >= prominence_floor:
                out.append(Peak(position=float(wavenumber[i]),
                                  intensity=float(intensity[i]),
                                  prominence=float(prom),
                                  width=8.0))
    return out


def primitives_from(wavenumber: np.ndarray, intensity: np.ndarray) -> dict:
    peaks = detect_peaks(wavenumber, intensity)
    return {
        "peaks": peaks,
        "n_peaks": len(peaks),
        "summary": f"{len(peaks)} peaks detected above prominence floor",
    }
