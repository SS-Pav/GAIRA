"""
Spectral window panel — 22-window broad-band scheme mapping wavenumber
ranges to BSV components.

Consistent with GAIRA spectral query v1-v3 work.
"""
from __future__ import annotations

import numpy as np


BSV_COMPONENTS = [
    "membrane_lipid", "protein_backbone", "aromatic_amino_acid",
    "purine_nucleotide", "pyrimidine_nucleotide", "glycan_carbohydrate",
    "redox_metabolite", "nucleic_acid_backbone",
]

# (window_id, start_cm, end_cm, bsv_component)
WINDOW_DEFS = [
    ("450-500",   450,  500, "redox_metabolite"),
    ("500-540",   500,  540, "redox_metabolite"),
    ("540-580",   540,  580, "unmapped"),
    ("580-620",   580,  620, "unmapped"),
    ("620-660",   620,  660, "aromatic_amino_acid"),
    ("660-700",   660,  700, "purine_nucleotide"),
    ("700-740",   700,  740, "purine_nucleotide"),
    ("740-780",   740,  780, "pyrimidine_nucleotide"),
    ("780-820",   780,  820, "pyrimidine_nucleotide"),
    ("820-860",   820,  860, "aromatic_amino_acid"),
    ("860-920",   860,  920, "glycan_carbohydrate"),
    ("920-980",   920,  980, "protein_backbone"),
    ("980-1020",  980, 1020, "aromatic_amino_acid"),
    ("1020-1080", 1020, 1080, "nucleic_acid_backbone"),
    ("1080-1140", 1080, 1140, "glycan_carbohydrate"),
    ("1140-1200", 1140, 1200, "membrane_lipid"),
    ("1200-1260", 1200, 1260, "protein_backbone"),
    ("1260-1320", 1260, 1320, "protein_backbone"),
    ("1320-1380", 1320, 1380, "purine_nucleotide"),
    ("1380-1450", 1380, 1450, "membrane_lipid"),
    ("1450-1520", 1450, 1520, "protein_backbone"),
    ("1520-1600", 1520, 1600, "aromatic_amino_acid"),
]


def extract_window_features(
    X_norm: np.ndarray,
    wavenumbers: np.ndarray,
) -> np.ndarray:
    """Extract mean intensity per window for each spectrum.

    Args:
        X_norm: Normalized spectra, shape (n_spectra, n_wavenumbers).
        wavenumbers: Wavenumber axis, shape (n_wavenumbers,).

    Returns:
        Window features, shape (n_spectra, 22).
    """
    n = X_norm.shape[0]
    features = np.zeros((n, len(WINDOW_DEFS)))

    for i, (_, wstart, wend, _) in enumerate(WINDOW_DEFS):
        mask = (wavenumbers >= wstart) & (wavenumbers < wend)
        if mask.sum() > 0:
            features[:, i] = X_norm[:, mask].mean(axis=1)

    return features
