"""GAIRA engine — Biochemical State Vector v2 (Part 6).

Deterministic, documented transform from latent component coordinates to a
biochemical theme vector.

EQUATIONS
---------
Given a query spectrum projected into the frozen atlas as non-negative component
activations a in R^24:

  coord_j   = a_j / sum_k a_k                         # L1 evidence share per component
  z_j       = (coord_j - center_j) / spread_j         # robust z vs reference frame
  W_{j,t}   = component->theme weight (rows sum to 1)  # from the ontology mapping

  composition_t = sum_j W_{j,t} * coord_j             # theme's share of the evidence  (>=0, ~sums to 1)
  elevation_t   = sum_j W_{j,t} * z_j                 # how elevated the theme is vs pure references
  display_t     = 0.5 + 0.5 * tanh(elevation_t / 3)   # bounded 0..1 radar value

CONFIDENCE (per theme, in [0,1]) multiplies three factors:
  stability_t  = sum_j W_{j,t} * component_stability_j  (weighted bootstrap stability)
  evidence_t   = concentration of the theme's evidence (1 - normalized entropy of W_{.,t}*coord)
  ood_factor   = 1 - OOD_score                         (down-weight out-of-distribution inputs)
  confidence_t = stability_t * evidence_t * ood_factor

The non-biochemical themes (background_matrix, unknown_mixed) are computed and
REPORTED (a high value lowers overall confidence) but are excluded from radar axes.

Nothing here is learned; the only inputs are the frozen atlas projection and the
versioned ontology weights.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .ontology import Ontology, NON_BIOCHEMICAL_THEMES
from .registry import ComponentRegistry
from .normalization import ReferenceFrame
from .versioning import VERSIONS


@dataclass
class BSV:
    theme_ids: list
    composition: dict            # theme -> share of evidence [0..1]
    elevation: dict              # theme -> signed z vs reference
    display: dict                # theme -> bounded 0..1
    confidence: dict             # theme -> [0..1]
    component_coord: np.ndarray  # 24-vector L1 share
    component_z: np.ndarray      # 24-vector robust z
    ood_score: float
    overall_confidence: float
    non_biochemical: dict        # background/unknown shares
    versions: dict = field(default_factory=lambda: VERSIONS.as_dict())

    def biochemical_themes(self):
        return {t: self.composition[t] for t in self.theme_ids if t not in NON_BIOCHEMICAL_THEMES}

    def as_dict(self):
        return {"composition": self.composition, "elevation": self.elevation,
                "display": self.display, "confidence": self.confidence,
                "ood_score": self.ood_score, "overall_confidence": self.overall_confidence,
                "non_biochemical": self.non_biochemical, "versions": self.versions}


class BSVBuilder:
    def __init__(self, ontology: Ontology = None, registry: ComponentRegistry = None,
                 frame: ReferenceFrame = None):
        self.onto = ontology or Ontology()
        self.reg = registry or ComponentRegistry()
        self.frame = frame or ReferenceFrame()
        self.W = self.onto.W
        self.theme_ids = self.onto.theme_ids
        self.stab = np.array([self.reg.stability(j) for j in range(self.onto.k)])

    def from_activation(self, a):
        """Build a BSV from a single 24-component non-negative activation vector."""
        a = np.clip(np.nan_to_num(np.asarray(a, float)), 0, None)
        s = a.sum()
        coord = a / s if s > 1e-12 else a
        z = self.frame.zscore(coord)

        composition = self.W.T @ coord                     # (n_themes,)
        elevation = self.W.T @ z
        ood = self.frame.ood_score(coord)

        # per-theme confidence
        weighted = self.W * coord[:, None]                 # (24, n_themes) evidence mass
        conf = {}
        for i, t in enumerate(self.theme_ids):
            wt = self.W[:, i]
            stab_t = float(np.sum(wt * self.stab) / (wt.sum() + 1e-12))
            m = weighted[:, i]; ms = m.sum()
            if ms > 1e-12:
                p = m / ms
                ev = 1.0 - float(-np.sum(p[p > 0] * np.log(p[p > 0])) / np.log(len(p)))
            else:
                ev = 0.0
            conf[t] = float(np.clip(stab_t * ev * (1 - ood), 0, 1))

        comp = {t: float(composition[i]) for i, t in enumerate(self.theme_ids)}
        elev = {t: float(elevation[i]) for i, t in enumerate(self.theme_ids)}
        disp = {t: self.frame.display_scale(elevation[i]) for i, t in enumerate(self.theme_ids)}
        nonbio = {t: comp[t] for t in NON_BIOCHEMICAL_THEMES if t in comp}
        # overall confidence: mean biochemical-theme confidence, penalised by matrix/unknown share
        bio = [t for t in self.theme_ids if t not in NON_BIOCHEMICAL_THEMES]
        penalty = 1.0 - min(0.8, sum(nonbio.values()))
        overall = float(np.clip(np.mean([conf[t] for t in bio]) * penalty, 0, 1))
        return BSV(self.theme_ids, comp, elev, disp, conf, coord, z, ood, overall, nonbio)

    def from_spectrum_projection(self, coordinates):
        """Alias when the caller already has L1 coordinates (e.g. cached projection)."""
        return self.from_activation(coordinates)
