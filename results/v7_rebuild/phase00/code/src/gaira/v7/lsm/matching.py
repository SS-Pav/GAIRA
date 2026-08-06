"""GAIRA V7 — matching spectra and analytes to Local Spectral Motifs.

This is the INTERPRETATION path. It runs strictly downstream of the frozen atlas
projection and never alters it:

    spectrum ──► canonical preprocessing ──► NNLS onto the FROZEN 24 components   (unchanged)
                                                        │
                                                        ▼
                                      per-component activation w_k
                                                        │
                                                        ▼
                          motif attribution: which substructure of component k
                          is this activation actually carrying, for this spectrum?

The motif layer only ever *redistributes* an activation that the frozen atlas already
produced. Summed over the motifs of component k, the attributed evidence equals w_k
exactly, so no new evidence is created and the atlas remains the sole source of activation.
"""
from __future__ import annotations

import numpy as np

from .motif import LSM


def band_mass(x: np.ndarray, motif: LSM, half_width: int = 4) -> np.ndarray:
    """Observed mass of a spectrum inside each of a motif's bands."""
    x = np.nan_to_num(np.asarray(x, float))
    out = []
    for idx in motif.band_indices:
        lo, hi = max(0, idx - half_width), min(len(x) - 1, idx + half_width)
        out.append(float(x[lo:hi + 1].sum()))
    return np.asarray(out, float)


def motif_affinity(x: np.ndarray, motif: LSM, half_width: int = 4) -> float:
    """Cosine between a spectrum's band profile and the motif's band-weight profile.

    Bounded in [0, 1] for non-negative inputs; 0 when the spectrum carries nothing in the
    motif's bands.
    """
    q = band_mass(x, motif, half_width)
    s = q.sum()
    if s <= 0:
        return 0.0
    q = q / s
    w = np.asarray(motif.band_weights, float)
    w = w / (w.sum() + 1e-12)
    return float(np.dot(q, w) / (np.linalg.norm(q) * np.linalg.norm(w) + 1e-12))


def attribute_component(x: np.ndarray, activation: float, motifs: list[LSM],
                        half_width: int = 4) -> dict[str, float]:
    """Split one component's activation across its motifs, conserving the total.

    Returns {motif_id: attributed activation}. The values sum to `activation` (to floating
    point), so the motif layer adds resolution without adding evidence. With no motifs, or
    with no affinity at all, the activation is returned unattributed under the reserved key
    `"_unattributed"` rather than being silently dropped.
    """
    if not motifs or activation <= 0:
        return {"_unattributed": float(max(activation, 0.0))}
    aff = np.array([motif_affinity(x, m, half_width) for m in motifs], float)
    if aff.sum() <= 0:
        return {"_unattributed": float(activation)}
    w = aff / aff.sum()
    return {m.motif_id: float(activation * wi) for m, wi in zip(motifs, w)}


def attribute_spectrum(x: np.ndarray, activations: np.ndarray, registry,
                       half_width: int = 4) -> dict[str, float]:
    """Motif-level evidence for one spectrum, across every component.

    `activations` is the frozen-atlas NNLS result (length 24). Nothing here re-projects.
    """
    out: dict[str, float] = {}
    for k, wk in enumerate(np.asarray(activations, float)):
        if wk <= 0:
            continue
        got = attribute_component(x, float(wk), registry.by_component(k), half_width)
        for key, v in got.items():
            out[key] = out.get(key, 0.0) + v
    return out


def attribution_matrix(X: np.ndarray, W: np.ndarray, registry,
                       half_width: int = 4) -> tuple[np.ndarray, list[str]]:
    """Motif attribution for a batch. Row i depends only on (X[i], W[i]) — batch-independent."""
    ids = [m.motif_id for m in registry.retained] + ["_unattributed"]
    pos = {mid: j for j, mid in enumerate(ids)}
    A = np.zeros((X.shape[0], len(ids)), float)
    for i in range(X.shape[0]):
        for key, v in attribute_spectrum(X[i], W[i], registry, half_width).items():
            if key in pos:
                A[i, pos[key]] += v
    return A, ids


def conservation_error(A: np.ndarray, W: np.ndarray) -> float:
    """Max absolute difference between total attributed evidence and total activation.

    Must be ~0: the motif layer redistributes, it does not create or destroy.
    """
    return float(np.max(np.abs(A.sum(axis=1) - W.sum(axis=1))))
