"""
Motif → Theme → BSV mapper for GAIRA text queries.

Organizes retrieved evidence into three layers:
  motifs (specific molecular features)
  → themes (broader biochemical categories)
  → BSV components (GAIRA's 8-axis representation)

All mapping is rule-based and documented.
"""
from __future__ import annotations

import re
from typing import Sequence

from gaira.retrieval.text_query_retriever import RetrievedItem


# ── Motif definitions ──────────────────────────────────────────────────
# Each motif has keywords and maps to one theme.

MOTIF_DEFS = {
    # Lipid motifs
    "CH2/CH3 deformation":   {"keywords": ["ch2", "ch3", "1440", "1450", "1460", "deformation"], "theme": "membrane_lipid", "peaks": "1380–1450 cm⁻¹"},
    "cholesterol":           {"keywords": ["cholesterol"], "theme": "membrane_lipid", "peaks": "700, 1440 cm⁻¹"},
    "phospholipid":          {"keywords": ["phospholipid", "phosphatidyl"], "theme": "membrane_lipid", "peaks": "1080, 1140–1200 cm⁻¹"},
    "lipid chain":           {"keywords": ["lipid chain", "fatty acid", "acyl"], "theme": "membrane_lipid", "peaks": "1060–1130, 1440–1460 cm⁻¹"},

    # Protein motifs
    "amide I/II/III":        {"keywords": ["amide i", "amide ii", "amide iii", "1230", "1260", "1650"], "theme": "protein_backbone", "peaks": "1230–1320, 1520–1600, 1620–1680 cm⁻¹"},
    "collagen":              {"keywords": ["collagen", "proline", "hydroxyproline"], "theme": "protein_backbone", "peaks": "855, 920–940 cm⁻¹"},
    "protein backbone":      {"keywords": ["protein backbone", "peptide bond"], "theme": "protein_backbone", "peaks": "1200–1320 cm⁻¹"},

    # Aromatic AA motifs
    "phenylalanine":         {"keywords": ["phenylalanine", "phe", "1003", "1000"], "theme": "aromatic_amino_acid", "peaks": "1000–1010 cm⁻¹"},
    "tryptophan":            {"keywords": ["tryptophan", "trp", "757"], "theme": "aromatic_amino_acid", "peaks": "757, 1340–1360 cm⁻¹"},
    "tyrosine":              {"keywords": ["tyrosine", "tyr", "830", "850"], "theme": "aromatic_amino_acid", "peaks": "620–660, 830–860 cm⁻¹"},

    # Purine motifs
    "adenine":               {"keywords": ["adenine", "735"], "theme": "purine_nucleotide", "peaks": "660–700, 735 cm⁻¹"},
    "guanine":               {"keywords": ["guanine", "660", "681"], "theme": "purine_nucleotide", "peaks": "660–700 cm⁻¹"},

    # Pyrimidine motifs
    "cytosine":              {"keywords": ["cytosine"], "theme": "pyrimidine_nucleotide", "peaks": "780–790 cm⁻¹"},
    "thymine":               {"keywords": ["thymine", "uracil"], "theme": "pyrimidine_nucleotide", "peaks": "740–780 cm⁻¹"},

    # Glycan motifs
    "glucose/glycogen":      {"keywords": ["glucose", "glycogen", "galactose"], "theme": "glycan_carbohydrate", "peaks": "860–920, 1080–1140 cm⁻¹"},
    "glycan":                {"keywords": ["glycan", "glycosylation", "mannose", "860", "920"], "theme": "glycan_carbohydrate", "peaks": "860–920 cm⁻¹"},

    # Redox motifs
    "disulfide/thiol":       {"keywords": ["disulfide", "thiol", "s-s", "500", "540"], "theme": "redox_metabolite", "peaks": "450–540 cm⁻¹"},
    "glutathione":           {"keywords": ["glutathione", "gsh"], "theme": "redox_metabolite", "peaks": "500–540 cm⁻¹"},
    "carotenoid":            {"keywords": ["carotenoid", "carotene"], "theme": "redox_metabolite", "peaks": "1005, 1155, 1525 cm⁻¹"},

    # Nucleic acid backbone motifs
    "PO2 stretch":           {"keywords": ["po2", "phosphodiester", "phosphate", "1020", "1080"], "theme": "nucleic_acid_backbone", "peaks": "1020–1080 cm⁻¹"},
    "C-O-C vibration":       {"keywords": ["c-o-c", "ribose", "deoxyribose"], "theme": "nucleic_acid_backbone", "peaks": "1040–1100 cm⁻¹"},
}

# ── Theme → BSV mapping ───────────────────────────────────────────────
# Each theme maps 1:1 to a BSV component (by design).

THEME_TO_BSV = {
    "membrane_lipid":         "membrane_lipid",
    "protein_backbone":       "protein_backbone",
    "aromatic_amino_acid":    "aromatic_amino_acid",
    "purine_nucleotide":      "purine_nucleotide",
    "pyrimidine_nucleotide":  "pyrimidine_nucleotide",
    "glycan_carbohydrate":    "glycan_carbohydrate",
    "redox_metabolite":       "redox_metabolite",
    "nucleic_acid_backbone":  "nucleic_acid_backbone",
}

THEME_DISPLAY = {
    "membrane_lipid":         "Membrane / Lipid",
    "protein_backbone":       "Protein Backbone",
    "aromatic_amino_acid":    "Aromatic Amino Acids",
    "purine_nucleotide":      "Purine Nucleotides",
    "pyrimidine_nucleotide":  "Pyrimidine Nucleotides",
    "glycan_carbohydrate":    "Glycan / Carbohydrate",
    "redox_metabolite":       "Redox / Metabolite",
    "nucleic_acid_backbone":  "Nucleic Acid Backbone",
}


def map_evidence_to_motifs_themes_bsv(
    retrieved_items: Sequence[RetrievedItem],
) -> dict:
    """Map retrieved evidence through motifs → themes → BSV components.

    Returns:
        dict with:
          motifs: list of {name, theme, evidence_indices, hit_count}
          themes: list of {name, display, bsv_component, motifs, evidence_indices}
          bsv_links: {component: [evidence_indices]}
          evidence_links: {evidence_idx: [motif_names]}
    """
    # Scan each evidence item for motif keywords
    motif_hits: dict[str, list[int]] = {m: [] for m in MOTIF_DEFS}
    evidence_links: dict[int, list[str]] = {}

    for i, item in enumerate(retrieved_items):
        text = (item.text + " " + (item.title or "")).lower()
        evidence_links[i] = []
        for motif_name, motif_def in MOTIF_DEFS.items():
            if any(kw in text for kw in motif_def["keywords"]):
                motif_hits[motif_name].append(i)
                evidence_links[i].append(motif_name)

    # Build motif list (only those with hits)
    motifs = []
    for name, ev_indices in motif_hits.items():
        if ev_indices:
            motifs.append({
                "name": name,
                "theme": MOTIF_DEFS[name]["theme"],
                "peaks": MOTIF_DEFS[name].get("peaks", ""),
                "evidence_indices": ev_indices,
                "hit_count": len(ev_indices),
            })
    motifs.sort(key=lambda m: -m["hit_count"])

    # Build theme list
    theme_data: dict[str, dict] = {}
    for m in motifs:
        theme = m["theme"]
        if theme not in theme_data:
            theme_data[theme] = {
                "name": theme,
                "display": THEME_DISPLAY.get(theme, theme),
                "bsv_component": THEME_TO_BSV.get(theme, theme),
                "motifs": [],
                "evidence_indices": set(),
            }
        theme_data[theme]["motifs"].append(m["name"])
        theme_data[theme]["evidence_indices"].update(m["evidence_indices"])

    themes = []
    for td in theme_data.values():
        themes.append({
            "name": td["name"],
            "display": td["display"],
            "bsv_component": td["bsv_component"],
            "motifs": td["motifs"],
            "evidence_indices": sorted(td["evidence_indices"]),
            "evidence_count": len(td["evidence_indices"]),
        })
    themes.sort(key=lambda t: -t["evidence_count"])

    # BSV links
    bsv_links: dict[str, list[int]] = {}
    for t in themes:
        comp = t["bsv_component"]
        if comp not in bsv_links:
            bsv_links[comp] = []
        for idx in t["evidence_indices"]:
            if idx not in bsv_links[comp]:
                bsv_links[comp].append(idx)

    return {
        "motifs": motifs,
        "themes": themes,
        "bsv_links": bsv_links,
        "evidence_links": evidence_links,
    }
