"""GAIRA engine — evidence engine (Part 8).

For every biochemical theme in a BSV, trace the full evidence chain: which
components contributed, which reference analytes support them, which perturbation
experiments back them, which literature, and what caveats apply. The output reads
as scientific reasoning, not classification. Deterministic; nothing hidden.
"""
from __future__ import annotations
import numpy as np

from .ontology import Ontology
from .registry import ComponentRegistry


class EvidenceEngine:
    def __init__(self, ontology: Ontology = None, registry: ComponentRegistry = None):
        self.onto = ontology or Ontology()
        self.reg = registry or ComponentRegistry()

    def trace_theme(self, bsv, theme_id, top_components=4):
        onto, reg = self.onto, self.reg
        contribs = onto.contributors(theme_id, top=top_components)
        components = []
        ref_analytes, perturbation, caveats = {}, [], []
        for j, w in contribs:
            ev = onto.weight_evidence(j, theme_id)
            loadings = reg.value(j, "reference_analyte_loadings")[:4]
            dose = reg.value(j, "dose_response_evidence")
            spike = reg.value(j, "serum_spike_evidence")
            components.append({
                "component": j, "theme_weight": round(w, 3),
                "interpretation": reg.value(j, "current_interpretation"),
                "stability": reg.stability(j),
                "confidence": reg.value(j, "interpretation_confidence"),
                "weight_evidence": ev,
                "top_reference_analytes": [f"{l['analyte']} ({l['contribution_pct']}%)" for l in loadings],
            })
            for l in loadings:
                ref_analytes[l["analyte"]] = ref_analytes.get(l["analyte"], 0) + w * l["contribution_pct"]
            for de in dose:
                perturbation.append(f"c{j}: {de['experiment']} ρ={de['spearman_rho']} ({de['direction']})")
            for cav in reg.value(j, "known_caveats"):
                caveats.append(f"c{j}: {cav}")
        theme = onto.theme(theme_id)
        strength = self._strength(bsv, theme_id)
        return {
            "theme": theme_id, "name": theme["name"],
            "description": theme["description"].strip(),
            "composition": round(bsv.composition[theme_id], 3),
            "display": round(bsv.display[theme_id], 3),
            "elevation_z": round(bsv.elevation[theme_id], 2),
            "confidence": round(bsv.confidence[theme_id], 3),
            "evidence_strength": strength,
            "contributing_components": components,
            "supporting_reference_analytes": [a for a, _ in
                                              sorted(ref_analytes.items(), key=lambda x: -x[1])[:6]],
            "perturbation_support": perturbation[:6],
            "literature": theme.get("literature"),
            "known_ambiguities": theme.get("ambiguities", "").strip(),
            "domain_caveats": theme.get("domain_caveats", "").strip(),
            "component_caveats": caveats[:6],
        }

    def _strength(self, bsv, theme_id):
        c = bsv.confidence[theme_id]; comp = bsv.composition[theme_id]
        s = 0.6 * c + 0.4 * min(1.0, comp / 0.25)
        return "strong" if s >= 0.6 else ("moderate" if s >= 0.35 else "weak")

    def full_report(self, bsv, top_themes=6):
        """Evidence table for the strongest biochemical themes + honesty flags."""
        bio = bsv.biochemical_themes()
        order = sorted(bio, key=lambda t: -bsv.composition[t])[:top_themes]
        traces = [self.trace_theme(bsv, t) for t in order]
        flags = []
        if bsv.ood_score > 0.3:
            flags.append(f"OUT OF DISTRIBUTION (score {bsv.ood_score:.2f}): this spectrum sits outside "
                         "the pure-Raman reference cloud; interpret every theme with caution.")
        matrix = bsv.non_biochemical.get("background_matrix", 0)
        if matrix > 0.25:
            flags.append(f"HIGH MATRIX/BACKGROUND SHARE ({matrix:.2f}): a large fraction of the signal is "
                         "colloid/matrix, not biochemistry — themes are down-weighted accordingly.")
        unk = bsv.non_biochemical.get("unknown_mixed", 0)
        if unk > 0.2:
            flags.append(f"UNEXPLAINED SIGNAL ({unk:.2f}): the atlas does not confidently attribute this "
                         "portion of the spectrum.")
        return {"themes": traces, "overall_confidence": round(bsv.overall_confidence, 3),
                "ood_score": round(bsv.ood_score, 3), "honesty_flags": flags,
                "versions": bsv.versions}
