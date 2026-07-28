"""GAIRA engine — canonical inference pipeline (Parts 2 & 12).

    input spectrum (wavenumber, intensity, domain)
        -> modality-aware preprocessing (atlas-native; frozen)
        -> projection into the frozen Raman Reference Atlas  (NMF k=24, held fixed)
        -> 24 latent Raman motif coordinates
        -> ontology mapping (component -> theme weights)
        -> Biochemical State Vector (BSV v2)
        -> domain-aware interpretation
        -> evidence report
        -> radar (backend data)

Deterministic. Provenance-preserving. No opaque ML: the only model is the frozen,
non-negative NMF basis, applied by non-negative least squares with the dictionary
held fixed. Every output carries versions and an OOD score.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..foundation import serialization as SER, dataset as DS
from ..preprocessing import pipeline as pp
from .bsv import BSVBuilder
from .evidence import EvidenceEngine
from .radar import RadarBackend
from .domain import get_domain
from .versioning import VERSIONS

FROZEN = None  # resolved lazily


@dataclass
class GAIRAInference:
    bsv: object
    radar: dict
    evidence: dict
    domain: dict
    component_coordinates: list
    ood_score: float
    versions: dict

    def as_dict(self):
        """The full, nothing-hidden output structure (Part 12)."""
        return {
            "biochemical_state_vector": self.bsv.as_dict(),
            "radar": self.radar,
            "evidence": self.evidence,
            "domain_interpretation": self.domain,
            "component_coordinates": self.component_coordinates,
            "ood_score": round(float(self.ood_score), 3),
            "overall_confidence": round(float(self.bsv.overall_confidence), 3),
            "ontology_version": self.versions["biochemical_ontology"],
            "atlas_fingerprint": self.versions["atlas_fingerprint"],
            "versions": self.versions,
        }


class GAIRAEngine:
    """Loads the frozen atlas and the versioned interpretation stack once."""

    def __init__(self):
        global FROZEN
        from . import paths
        FROZEN = paths.foundation_dir()          # assets/foundation first, legacy fallback
        self.atlas = SER.load_frozen_manifold(FROZEN)
        self.builder = BSVBuilder()
        self.evidence = EvidenceEngine(ontology=self.builder.onto, registry=self.builder.reg)
        self.radar = RadarBackend(ontology=self.builder.onto, evidence=self.evidence)
        assert self.atlas.meta["fingerprint"] == VERSIONS.atlas_fingerprint, \
            "loaded atlas fingerprint does not match the engine's pinned version"

    # ── stage 1: preprocessing (atlas-native, frozen) ──
    def preprocess(self, wavenumber, intensity):
        return pp.preprocess(np.asarray(wavenumber, float), np.asarray(intensity, float),
                             DS.PREPROC, DS.GRID, DS.WINDOW)

    # ── stage 2: projection into the frozen atlas ──
    def project(self, atlas_native_spectrum):
        return self.atlas.coordinates(np.atleast_2d(atlas_native_spectrum), normalise=True)[0]

    # ── full pipeline ──
    def infer(self, wavenumber=None, intensity=None, coordinates=None, domain="buffer",
              radar_score="composition"):
        """Run the canonical pipeline. Provide either (wavenumber, intensity) or a
        precomputed 24-component `coordinates` vector (e.g. a cached projection)."""
        if coordinates is None:
            v = self.preprocess(wavenumber, intensity)
            coordinates = self.project(v)
        bsv = self.builder.from_activation(coordinates)
        report = self.evidence.full_report(bsv)
        radar = self.radar.build(bsv, score=radar_score)
        dctx = get_domain(domain).interpret(bsv)
        return GAIRAInference(bsv=bsv, radar=radar, evidence=report, domain=dctx,
                              component_coordinates=[round(float(x), 5) for x in bsv.component_coord],
                              ood_score=bsv.ood_score, versions=VERSIONS.as_dict())
