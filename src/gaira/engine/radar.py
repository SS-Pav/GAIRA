"""GAIRA engine — radar backend (Part 7).

The radar now displays BIOCHEMICAL THEMES, not latent components. This is the
BACKEND that produces the radar data structure (no UI). Each axis carries a score,
confidence, OOD modifier, evidence strength and a click-through provenance handle.
"""
from __future__ import annotations
import numpy as np

from .ontology import Ontology
from .evidence import EvidenceEngine


class RadarBackend:
    def __init__(self, ontology: Ontology = None, evidence: EvidenceEngine = None):
        self.onto = ontology or Ontology()
        self.ev = evidence or EvidenceEngine(ontology=self.onto)

    def build(self, bsv, score="display", include_provenance=True):
        """Radar axes for the biochemical themes.
        score: 'display' (bounded elevation) or 'composition' (evidence share)."""
        axes = []
        for t in self.onto.biochemical_theme_ids:
            base = bsv.display[t] if score == "display" else bsv.composition[t]
            ood_mod = 1.0 - bsv.ood_score                       # OOD attenuates every axis
            axis = {
                "theme": t, "label": self.onto.theme(t)["name"],
                "score": round(float(base), 4),
                "score_ood_adjusted": round(float(base * ood_mod), 4),
                "confidence": round(float(bsv.confidence[t]), 3),
                "ood_modifier": round(float(ood_mod), 3),
                "evidence_strength": self.ev._strength(bsv, t),
                "composition_share": round(float(bsv.composition[t]), 4),
                "elevation_z": round(float(bsv.elevation[t]), 2),
            }
            if include_provenance:
                axis["provenance"] = {
                    "contributing_components": [j for j, _ in self.onto.contributors(t)],
                    "literature": self.onto.literature(t),
                    "trace_handle": f"evidence.trace_theme('{t}')",
                }
            axes.append(axis)
        # order axes by composition for a stable, readable radar
        axes.sort(key=lambda a: -a["composition_share"])
        return {
            "axes": axes,
            "n_axes": len(axes),
            "overall_confidence": round(float(bsv.overall_confidence), 3),
            "ood_score": round(float(bsv.ood_score), 3),
            "non_biochemical_share": {k: round(float(v), 3) for k, v in bsv.non_biochemical.items()},
            "scoring": score,
            "note": "Axes are biochemical themes. score_ood_adjusted attenuates by OOD; a large "
                    "non_biochemical_share means the radar should be read cautiously.",
            "versions": bsv.versions,
        }
