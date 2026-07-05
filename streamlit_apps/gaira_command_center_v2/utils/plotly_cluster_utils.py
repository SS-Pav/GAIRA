"""Plotly-side helpers for cluster overlays (hulls, ellipses, colour maps)."""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError


# Dark-theme-friendly distinct palette for ~30 categorical classes.
PALETTE_30 = [
    "#79c0ff", "#ff7b72", "#d2a8ff", "#a5d6ff", "#ffa657",
    "#7ee787", "#f0883e", "#bc8cff", "#56d4dd", "#ffdf5d",
    "#ff9492", "#a371f7", "#39c5cf", "#fab8c4", "#85e89d",
    "#f97583", "#b392f0", "#9ecbff", "#ffea7f", "#f0d4a3",
    "#5dade2", "#e59866", "#cb6ce6", "#7dcea0", "#f5b041",
    "#af7ac5", "#48c9b0", "#ec7063", "#5499c7", "#aab7b8",
]

# Stable colour assignment for the 11 BSV families.
BSV_FAMILY_COLORS = {
    "G01": "#79c0ff",  # purine_nucleotide
    "G02": "#a5d6ff",  # purine_metabolite
    "G03": "#56d4dd",  # pyrimidine_nucleotide
    "G04": "#bc8cff",  # nucleic_acid_phosphate
    "G05": "#ffa657",  # glycan_carbohydrate
    "G06": "#7ee787",  # protein_peptide_backbone
    "G07": "#d2a8ff",  # aromatic_residue
    "G08": "#ffdf5d",  # lipid_acyl_membrane
    "G09": "#f0883e",  # sterol_neutral_lipid
    "G10": "#ff7b72",  # sulfur_thiol_redox
    "G11": "#85e89d",  # metabolic_small_molecule
}


def color_map(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    uniq = sorted({str(v) for v in values if v is not None})
    for i, v in enumerate(uniq):
        out[v] = PALETTE_30[i % len(PALETTE_30)]
    return out


# ─── convex hulls ─────────────────────────────────────────────────────────

def convex_hull_xy(points: np.ndarray) -> np.ndarray | None:
    """Return closed hull polygon vertices (Nx2) or None if degenerate."""
    if len(points) < 3:
        return None
    try:
        hull = ConvexHull(points)
        idx = list(hull.vertices) + [hull.vertices[0]]
        return points[idx]
    except (QhullError, ValueError):
        return None


# ─── covariance ellipses ─────────────────────────────────────────────────

def ellipse_xy(points: np.ndarray, n_sigma: float = 2.0,
               n_samples: int = 100) -> np.ndarray | None:
    """Return polygon approximating the n_sigma covariance ellipse."""
    if len(points) < 3:
        return None
    try:
        mean = points.mean(axis=0)
        cov = np.cov(points.T)
        if not np.isfinite(cov).all():
            return None
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 1e-8, None)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        a = n_sigma * np.sqrt(eigvals[0])
        b = n_sigma * np.sqrt(eigvals[1])
        theta = np.linspace(0, 2 * np.pi, n_samples)
        circle = np.column_stack([a * np.cos(theta), b * np.sin(theta)])
        rotated = circle @ eigvecs.T
        return rotated + mean
    except Exception:
        return None
