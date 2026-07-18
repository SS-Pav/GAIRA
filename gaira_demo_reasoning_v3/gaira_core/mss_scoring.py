"""GAIRA Demo v1 — MSS (molecular spectral signature) scoring.

Each curated reference in `data_loader.MOLECULES` defines anchor / support /
anti-evidence bands. MSS fire scores are computed against the input
spectrum and rolled up into per-molecule fire intensities.

This is intentionally simple. Production GAIRA's MSS engine
(`src/gaira/base3/mss_engine.py`) is far richer; the demo's purpose is
to make the scoring step *visible* to the user.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data_loader import MOLECULES, MoleculeRef


@dataclass
class MSSFire:
    molecule_id: str
    anchor_score: float
    support_score: float
    anti_score: float
    fire: float


def _local_max(wavenumber: np.ndarray, intensity: np.ndarray,
                center: float, tol: float = 8.0) -> float:
    mask = (wavenumber >= center - tol) & (wavenumber <= center + tol)
    if not mask.any():
        return 0.0
    return float(intensity[mask].max())


def score_one(mol: MoleculeRef, wavenumber: np.ndarray, intensity: np.ndarray,
               *, anchor_tol: float = 8.0, support_tol: float = 10.0,
               anti_tol: float = 8.0) -> MSSFire:
    anchors = [_local_max(wavenumber, intensity, c, anchor_tol) for c in mol.anchors] or [0.0]
    supports = [_local_max(wavenumber, intensity, c, support_tol) for c in mol.supports] or [0.0]
    antis = [_local_max(wavenumber, intensity, c, anti_tol) for c in mol.anti_evidence] or [0.0]
    a_score = float(np.mean(anchors)) if anchors else 0.0
    s_score = float(np.mean(supports)) if supports else 0.0
    n_score = float(np.mean(antis)) if antis else 0.0
    # Anchor must be present; supports add evidence; anti dampens
    fire = max(0.0, a_score * (1.0 + 0.4 * s_score) - 0.5 * n_score)
    return MSSFire(
        molecule_id=mol.name.lower().replace(" ", "_"),
        anchor_score=a_score, support_score=s_score, anti_score=n_score,
        fire=fire,
    )


def score_all(wavenumber: np.ndarray, intensity: np.ndarray) -> dict[str, MSSFire]:
    return {mol_id: score_one(ref, wavenumber, intensity)
              for mol_id, ref in MOLECULES.items()}


def molecule_axis_contributions(mol_id: str) -> dict[str, float]:
    """Static per-molecule contribution profile across the 11 BSV axes.

    Curated for demo clarity: most weight on primary_axis, smaller leaks
    into chemically-related axes. Returns dict normalized to sum=1.
    """
    from . import config as cfg

    ref = MOLECULES[mol_id]
    # Base profile: 0 for all axes
    contrib = {a: 0.0 for a in cfg.BSV_AXES}
    contrib[ref.primary_axis] = 0.7
    # Chemistry-driven secondary leaks
    chem = {
        "G01_purine_nucleotide": ["G02_purine_metabolite", "G04_nucleic_acid_phosphate"],
        "G02_purine_metabolite": ["G01_purine_nucleotide", "G11_metabolic_small_molecule"],
        "G03_pyrimidine_nucleotide": ["G04_nucleic_acid_phosphate"],
        "G04_nucleic_acid_phosphate": ["G01_purine_nucleotide", "G03_pyrimidine_nucleotide"],
        "G05_glycan_carbohydrate": ["G11_metabolic_small_molecule"],
        "G06_protein_peptide_backbone": ["G07_aromatic_residue"],
        "G07_aromatic_residue": ["G06_protein_peptide_backbone"],
        "G08_lipid_acyl_membrane": ["G09_sterol_neutral_lipid"],
        "G09_sterol_neutral_lipid": ["G08_lipid_acyl_membrane"],
        "G10_sulfur_thiol_redox": ["G11_metabolic_small_molecule", "G07_aromatic_residue"],
        "G11_metabolic_small_molecule": ["G10_sulfur_thiol_redox"],
    }
    for sec in chem.get(ref.primary_axis, []):
        contrib[sec] = 0.10
    # Tiny baseline noise so radar isn't binary-looking
    for a in cfg.BSV_AXES:
        if contrib[a] == 0.0:
            contrib[a] = 0.02
    # Normalize
    total = sum(contrib.values())
    if total > 0:
        contrib = {a: v / total for a, v in contrib.items()}
    return contrib
