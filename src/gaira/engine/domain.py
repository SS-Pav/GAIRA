"""GAIRA engine — domain-aware interpretation (Part 9).

The SAME BSV is interpreted differently per acquisition domain. Domain context NEVER
changes the BSV numbers; it only adds interpretive caveats and expectation-based
down-weighting notes (e.g. a large protein signal is unremarkable in serum). This
keeps the coordinate system frozen and shared while the reading adapts.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class DomainContext:
    domain: str
    description: str
    expected_dominant_themes: list     # themes that are large by construction, not informative
    reliable_themes: list              # themes that transfer reliably in this domain
    caveats: list

    def interpret(self, bsv):
        notes = []
        for t in self.expected_dominant_themes:
            if bsv.composition.get(t, 0) > 0.2:
                notes.append(f"{t}: large but EXPECTED in {self.domain} (matrix-level) — not evidence "
                             "of a perturbation.")
        for t in self.reliable_themes:
            if bsv.composition.get(t, 0) > 0.1:
                notes.append(f"{t}: elevated and reliably recovered in {self.domain}.")
        return {"domain": self.domain, "description": self.description,
                "reliable_themes": self.reliable_themes,
                "expected_background_themes": self.expected_dominant_themes,
                "domain_notes": notes, "caveats": self.caveats}


DOMAINS = {
    "serum": DomainContext(
        domain="serum",
        description="Pooled/patient serum via Ag-colloid SERS. Background-dominated (Spike "
                    "Validation): only strong Ag adsorbers are reliably recovered.",
        expected_dominant_themes=["protein_peptide", "lipid_acyl", "sterol_membrane",
                                  "saccharide_glycan", "background_matrix"],
        reliable_themes=["nucleic_purine", "sulfur_antioxidant"],
        caveats=["Albumin, cholesterol and glucose are millimolar and dominate; large signals in their "
                 "themes are non-specific.",
                 "Weak Ag adsorbers (amino acids, most lipids) are poorly recovered; a LOW score does not "
                 "mean the analyte is absent.",
                 "All serum SERS is out of domain for a Raman atlas; read with the OOD flag."]),
    "ev": DomainContext(
        domain="ev",
        description="Extracellular-vesicle preparations. Membrane-lipid and protein rich; nucleic-acid "
                    "cargo variable.",
        expected_dominant_themes=["lipid_acyl", "sterol_membrane", "protein_peptide"],
        reliable_themes=["lipid_acyl", "protein_peptide"],
        caveats=["EV spectra were NOT used to build or validate the atlas; interpretation is provisional.",
                 "Membrane themes are structurally expected and should be read as composition, not change."]),
    "buffer": DomainContext(
        domain="buffer",
        description="Pure/spiked analyte in buffer (closest to the reference regime).",
        expected_dominant_themes=["background_matrix"],
        reliable_themes=["nucleic_purine", "nucleic_pyrimidine", "sulfur_antioxidant",
                         "lipid_acyl", "saccharide_glycan", "protein_peptide"],
        caveats=["Closest match to the reference frame; lowest OOD of the SERS domains."]),
    "tissue": DomainContext(
        domain="tissue",
        description="Future domain — solid tissue Raman. Not yet validated.",
        expected_dominant_themes=["protein_peptide", "lipid_acyl"],
        reliable_themes=[],
        caveats=["FUTURE DOMAIN: no tissue data has been projected; interpretation unsupported."]),
    "dart": DomainContext(
        domain="dart",
        description="Future domain — DART electrochemical perturbation trajectories.",
        expected_dominant_themes=["background_matrix", "redox_broad"],
        reliable_themes=["redox_broad"],
        caveats=["FUTURE DOMAIN: a DART series is a TRAJECTORY of BSVs, not a single BSV; see the DART "
                 "interface. Interpretation deferred to the trajectory layer."]),
}


def get_domain(name):
    return DOMAINS.get((name or "buffer").lower(), DOMAINS["buffer"])
