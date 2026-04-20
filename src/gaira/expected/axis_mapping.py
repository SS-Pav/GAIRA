"""Single source of truth: DuckDB evidence row → BSV axis.

Rules are explicit so ambiguity is first-class. Any row that cannot be mapped
with high confidence resolves to `None` and is counted as ambiguous, never
silently dropped into a fallback axis.

NucleicAcids is the most important branch because the assigned_group lumps
purine, pyrimidine, and backbone evidence together; the molecule string is
the disambiguator.
"""
from __future__ import annotations

import re

from gaira.spectral.window_panel import BSV_COMPONENTS as BSV_AXES


# Purine-family keywords (detected case-insensitively in assigned_molecule
# and in the neighbouring evidence_text when present).
_PURINE_TOKENS = (
    "adenine", "guanine", "hypoxanthine", "xanthine",
    "uric acid", "urate", "purine",
)

# Pyrimidine-family keywords.
_PYRIMIDINE_TOKENS = (
    "cytosine", "thymine", "uracil", "pyrimidine",
)

# Aromatic-AA keywords (for disambiguating the AminoAcids group).
_AROMATIC_AA_TOKENS = (
    "tyrosine", "phenylalanine", "tryptophan",
    "aromatic", "ring breathing", "phe ", "tyr ", "trp ",
)

# Redox / sulfur-metabolite keywords — these sometimes show up inside the
# Metabolites group, and also inside AminoAcids (C-S / cysteine).
_REDOX_TOKENS = (
    "ergothioneine", "glutathione", "gsh",
    "cysteine", "disulfide", "thiol", "s-s",
    "carotenoid", "carotene",
    "vitamin c", "ascorbate", "nad",
)


def _tok_match(haystack: str, tokens: tuple[str, ...]) -> bool:
    h = (haystack or "").lower()
    return any(t in h for t in tokens)


def _disambiguate_nucleic(molecule: str, evidence_text: str) -> str:
    """Break NucleicAcids into purine / pyrimidine / backbone."""
    combined = f"{molecule or ''} {evidence_text or ''}"
    if _tok_match(combined, _PURINE_TOKENS):
        return "purine_nucleotide"
    if _tok_match(combined, _PYRIMIDINE_TOKENS):
        return "pyrimidine_nucleotide"
    return "nucleic_acid_backbone"


def _disambiguate_aminoacids(molecule: str, evidence_text: str) -> str | None:
    """AminoAcids → aromatic_amino_acid if aromatic tokens are present,
    else redox_metabolite if sulfur/thiol, else None (ambiguous AA)."""
    combined = f"{molecule or ''} {evidence_text or ''}"
    if _tok_match(combined, _AROMATIC_AA_TOKENS):
        return "aromatic_amino_acid"
    if _tok_match(combined, _REDOX_TOKENS):
        return "redox_metabolite"
    return None  # ambiguous — leave unmapped


def _disambiguate_metabolites(molecule: str, evidence_text: str) -> str | None:
    combined = f"{molecule or ''} {evidence_text or ''}"
    if _tok_match(combined, _REDOX_TOKENS):
        return "redox_metabolite"
    # Everything else ambiguous; don't force a fallback.
    return None


def assigned_row_to_axis(
    assigned_group: str | None,
    assigned_molecule: str | None = None,
    evidence_text: str | None = None,
) -> str | None:
    """Map a peak_assignments row to a BSV axis or return None for ambiguous.

    None means the evidence is retained in the audit as "ambiguous / unmapped",
    not dropped. Downstream consumers must count it in the ambiguity tally.
    """
    if not assigned_group:
        return None

    g = assigned_group.strip()

    if g == "Proteins":
        return "protein_backbone"
    if g == "Lipids-FattyAcids" or g == "Lipids-Hormones":
        return "membrane_lipid"
    if g.startswith("Saccharides"):
        return "glycan_carbohydrate"
    if g == "NucleicAcids":
        return _disambiguate_nucleic(assigned_molecule or "", evidence_text or "")
    if g == "AminoAcids":
        return _disambiguate_aminoacids(assigned_molecule or "", evidence_text or "")
    if g == "Metabolites":
        return _disambiguate_metabolites(assigned_molecule or "", evidence_text or "")
    if g == "Mixed-Vibrational":
        return None  # explicitly ambiguous by construction
    return None


# ─────────────────────────────────────────────────────────────────────
# Optional anchor hints — NOT used as a filter; used only to decide
# whether a clustered window matches a "canonical" biochemical anchor
# region for an axis. These are the Raman/SERS reference ranges that
# the motif_theme_mapper already encodes; we carry them here to check
# anchor-cluster quality downstream.
# ─────────────────────────────────────────────────────────────────────

AXIS_ANCHOR_HINTS: dict[str, list[tuple[int, int, str]]] = {
    "purine_nucleotide": [
        (660, 700, "guanine / tyrosine-adjacent ring"),
        (720, 740, "adenine / hypoxanthine ring breathing"),
        (1320, 1380, "purine N7-C8"),
    ],
    "pyrimidine_nucleotide": [
        (740, 780, "thymine / uracil ring"),
        (780, 820, "cytosine ring"),
    ],
    "nucleic_acid_backbone": [
        (1020, 1080, "C-N / ribose"),
        (1080, 1140, "PO2⁻ symmetric"),
    ],
    "aromatic_amino_acid": [
        (620, 660, "phenylalanine / Phe-adjacent ring"),
        (820, 860, "tyrosine Fermi doublet"),
        (1000, 1010, "phenylalanine ring breathing"),
        (1520, 1600, "tryptophan / ring C=C"),
    ],
    "membrane_lipid": [
        (1140, 1200, "lipid CH2 twist"),
        (1380, 1450, "lipid δCH2/CH3"),
    ],
    "protein_backbone": [
        (920, 980, "C-C backbone"),
        (1200, 1260, "Amide III β-sheet"),
        (1260, 1320, "Amide III α-helix"),
        (1450, 1520, "Amide II adjacent / δCH2"),
    ],
    "glycan_carbohydrate": [
        (860, 920, "glucose / polysaccharide C-O"),
        (1080, 1140, "sugar C-O-C (overlaps PO2⁻)"),
    ],
    "redox_metabolite": [
        (450, 540, "S-S disulfide"),
        (1005, 1010, "carotenoid C=C"),
        (1155, 1160, "carotenoid C-C"),
        (1510, 1530, "carotenoid C=C"),
    ],
}
