"""GAIRA Demo v1 — class-level motif scoring.

A small set of biochemical-class motifs that fire when the spectrum has
sustained signal in a chemistry-relevant band window. These are the
class-level signals that downgrade molecule-level overclaim into
family-level evidence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Motif:
    motif_id: str
    bands: tuple[tuple[float, float], ...]
    primary_axis: str
    description: str


MOTIFS: tuple[Motif, ...] = (
    Motif("purine_ring_breathing_720_735",
           ((720.0, 740.0),), "G01_purine_nucleotide",
           "Purine ring breathing 720–740 cm⁻¹ — purine nucleotide/metabolite class signal."),
    Motif("uric_acid_doublet_640_891",
           ((630.0, 655.0), (880.0, 905.0)), "G02_purine_metabolite",
           "Uric acid 640+891 cm⁻¹ doublet — strong purine-metabolite anchor."),
    Motif("pyrimidine_ring_780",
           ((770.0, 795.0),), "G03_pyrimidine_nucleotide",
           "Pyrimidine ring breathing near 780 cm⁻¹."),
    Motif("phosphate_po2_1080",
           ((1070.0, 1100.0),), "G04_nucleic_acid_phosphate",
           "Symmetric PO₂⁻ stretch 1080 cm⁻¹ — nucleic-acid phosphate."),
    Motif("glycan_co_co_1050_1150",
           ((1020.0, 1150.0),), "G05_glycan_carbohydrate",
           "Carbohydrate C–O/C–C stretches 1020–1150 cm⁻¹."),
    Motif("protein_amide_iii_1230_1300",
           ((1230.0, 1300.0),), "G06_protein_peptide_backbone",
           "Amide III backbone region 1230–1300 cm⁻¹."),
    Motif("aromatic_phe_1003",
           ((996.0, 1010.0),), "G07_aromatic_residue",
           "Phenylalanine ring breathing 1003 cm⁻¹."),
    Motif("lipid_acyl_1440_1655",
           ((1430.0, 1455.0), (1640.0, 1670.0)), "G08_lipid_acyl_membrane",
           "CH₂ deformation 1440 + C=C 1655 — lipid acyl/unsat. signal."),
    Motif("sterol_ring_548",
           ((540.0, 560.0),), "G09_sterol_neutral_lipid",
           "Sterol ring deformation near 548 cm⁻¹."),
    Motif("thione_c_s_490_500",
           ((480.0, 510.0),), "G10_sulfur_thiol_redox",
           "Thione/C–S stretch near 490–500 cm⁻¹ — ergothioneine/thiol redox."),
    Motif("lactate_c_co_845_925",
           ((835.0, 855.0), (915.0, 935.0)), "G11_metabolic_small_molecule",
           "Lactate-like C–C–O 845 + 925 cm⁻¹ small-molecule signal."),
)


def _band_score(wavenumber: np.ndarray, intensity: np.ndarray,
                 band: tuple[float, float]) -> float:
    lo, hi = band
    mask = (wavenumber >= lo) & (wavenumber <= hi)
    if not mask.any():
        return 0.0
    return float(intensity[mask].max())


def score_motifs(wavenumber: np.ndarray, intensity: np.ndarray) -> dict[str, float]:
    """Return motif_id -> fire score (∈ ~[0,1], not strictly bounded)."""
    out = {}
    for m in MOTIFS:
        scores = [_band_score(wavenumber, intensity, b) for b in m.bands]
        # All bands must fire (geometric mean rewards co-firing).
        if any(s <= 0 for s in scores):
            out[m.motif_id] = 0.0
        else:
            gm = float(np.exp(np.mean(np.log(scores))))
            out[m.motif_id] = gm
    return out


def get_motif(motif_id: str) -> Motif | None:
    for m in MOTIFS:
        if m.motif_id == motif_id:
            return m
    return None
