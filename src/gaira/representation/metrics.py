"""Distance / similarity metrics and spectral band diagnostics (Phase 2 §8)."""
from __future__ import annotations
import numpy as np


def cosine_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity matrix between A (m×d) and B (n×d)."""
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return An @ Bn.T


def cosine_dist(A, B):
    return 1.0 - cosine_sim(A, B)


def peaks(v, grid, prominence_frac=0.05, min_sep_cm=8.0):
    """Simple prominence peak picker on a preprocessed (possibly signed) vector.
    Uses |v| so it works for L2/SNV and derivative reps. Returns wavenumbers."""
    from scipy.signal import find_peaks
    a = np.abs(np.nan_to_num(v))
    if a.max() < 1e-12:
        return np.array([])
    dx = float(np.median(np.diff(grid)))
    dist = max(1, int(round(min_sep_cm / dx)))
    idx, _ = find_peaks(a, prominence=prominence_frac * a.max(), distance=dist)
    return grid[idx]


def peak_overlap(p1, p2, tol_cm=10.0):
    """Fraction of peaks in p1 with a match in p2 within tol (Jaccard-like)."""
    if len(p1) == 0 or len(p2) == 0:
        return 0.0
    matched = sum(any(abs(a - b) <= tol_cm for b in p2) for a in p1)
    matched2 = sum(any(abs(b - a) <= tol_cm for a in p1) for b in p2)
    return (matched + matched2) / (len(p1) + len(p2))
