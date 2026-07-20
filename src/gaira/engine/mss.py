"""GAIRA engine — Molecular Spectral Signatures (MSS) layer.

The interpretable MIDDLE of the GAIRA hierarchy, between the 24 latent Raman
components and the 13 biochemical themes:

    Radar (themes)  ->  "which biochemical systems changed?"
    MSS (this file) ->  "which biochemical spectral MOTIFS explain those changes?"
    Components      ->  "what latent evidence supports those motifs?"
    Reference       ->  "which reference chemistries contributed?"

An MSS motif is a validated spectral pattern (NOT a molecule). Motif *definitions*
(name, characteristic bands, exemplar chemistries, parent theme) are curated in
``data/mss_motifs_v1.yaml``. Everything quantitative — which components express a
motif, the component->motif weights, confidence, and perturbation evidence — is
DERIVED here from the frozen Component Registry and the frozen ontology weight
matrix. Nothing is asserted; nothing frozen is modified.

MSS is ADDITIVE and PARALLEL to the BSV: it does not feed the theme composition
(the BSV still maps components -> themes directly). It is a second, finer-grained
interpretive projection of the SAME frozen component coordinates, used by the demo
and the evidence view to explain a state in spectroscopic-motif terms.

Determinism: the derivation is a pure function of the frozen artifacts, so the MSS
registry is reproducible byte-for-byte. Region-based band matching (never exact
peak) honours GAIRA's "peak != molecule" principle.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import re
import unicodedata
from pathlib import Path
import numpy as np
import yaml

from .ontology import Ontology, NON_BIOCHEMICAL_THEMES
from .registry import ComponentRegistry
from .versioning import VERSIONS

DATA = Path(__file__).resolve().parent / "data"


def _norm_name(s: str) -> str:
    """Normalise an analyte name for matching (ligatures, case, punctuation)."""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()


def _name_matches(exemplar: str, analyte: str) -> bool:
    """Region-tolerant name match: exemplar is a substring token-set of analyte
    (handles '(+)-glucose' vs 'glucose', 'ribo?avin' ligatures, plurals)."""
    e, a = _norm_name(exemplar), _norm_name(analyte)
    if not e or not a:
        return False
    if e in a or a in e:
        return True
    et = set(e.split())
    at = set(a.split())
    # singular/plural tolerance on the head token
    return bool(et & at) and (et <= at or at <= et)


@dataclass
class MSSMotif:
    """A validated spectral motif with its derived component provenance."""
    id: str
    name: str
    description: str
    bands_cm: list
    exemplars: list
    parent_theme: str
    non_biochemical: bool
    # derived
    contributors: list = field(default_factory=list)   # [{component, weight, band, exemplar, theme}]
    confidence: float = 0.0
    stability: float = 0.0
    evidence_breadth: float = 0.0
    reference_analytes: list = field(default_factory=list)
    perturbation: dict = field(default_factory=dict)

    def as_dict(self):
        return {
            "id": self.id, "name": self.name, "description": self.description.strip(),
            "bands_cm": self.bands_cm, "exemplars": self.exemplars,
            "parent_theme": self.parent_theme, "non_biochemical": self.non_biochemical,
            "contributors": self.contributors, "confidence": round(self.confidence, 3),
            "stability": round(self.stability, 3), "evidence_breadth": round(self.evidence_breadth, 3),
            "reference_analytes": self.reference_analytes, "perturbation": self.perturbation,
        }


@dataclass
class MSSActivation:
    """One motif's activation for a specific query spectrum."""
    id: str
    name: str
    parent_theme: str
    non_biochemical: bool
    composition: float     # share-like: sum_j M[j,m] * coord_j        (>=0)
    elevation: float       # signed:     sum_j M[j,m] * z_j            (vs reference)
    display: float         # bounded 0..1: 0.5 + 0.5 tanh(elevation/3)
    confidence: float
    top_components: list   # [(component, weight, coord_share)]

    def as_dict(self):
        return {"id": self.id, "name": self.name, "parent_theme": self.parent_theme,
                "non_biochemical": self.non_biochemical,
                "composition": round(self.composition, 4), "elevation": round(self.elevation, 4),
                "display": round(self.display, 4), "confidence": round(self.confidence, 3),
                "top_components": self.top_components}


class MSSLayer:
    """Derives and applies the Molecular Spectral Signatures.

    Construct standalone (loads its own Ontology + ComponentRegistry) or from an
    existing engine via :meth:`from_engine` to share the loaded frozen stack.
    """

    def __init__(self, ontology: Ontology = None, registry: ComponentRegistry = None,
                 motif_path=None):
        self.onto = ontology or Ontology()
        self.reg = registry or ComponentRegistry()
        assert self.reg.fingerprint == self.onto.atlas_fingerprint == VERSIONS.atlas_fingerprint, \
            "MSS layer: frozen atlas fingerprints disagree"
        self.k = self.onto.k
        spec = yaml.safe_load((Path(motif_path) if motif_path else DATA / "mss_motifs_v1.yaml").read_text())
        self.version = spec["version"]
        self.tol = float(spec["band_match_tolerance_cm"])
        d = spec["derivation"]
        self.wb, self.we, self.wt = d["weights"]["band"], d["weights"]["exemplar"], d["weights"]["theme"]
        self.keep = float(d["keep_threshold"])
        self.max_contrib = int(d["max_contributors"])
        self.sat = float(d["exemplar_saturation_pct"])
        self._raw_motifs = spec["motifs"]
        self.stab = np.array([self.reg.stability(j) for j in range(self.k)])
        self.motifs = [self._derive(m) for m in self._raw_motifs]
        self.M = self._weight_matrix()          # (k, n_motifs), columns sum to 1 over contributors

    # ── derivation (pure function of the frozen artifacts) ──
    def _band_score(self, j, bands):
        comp_bands = np.asarray(self.reg.value(j, "dominant_raman_peaks_cm"), float)
        if comp_bands.size == 0 or not bands:
            return 0.0
        hit = sum(np.any(np.abs(comp_bands - b) <= self.tol) for b in bands)
        return hit / len(bands)

    def _exemplar_score(self, j, exemplars):
        loads = self.reg.value(j, "reference_analyte_loadings")
        pct = 0.0
        matched = []
        for l in loads:
            if any(_name_matches(e, l["analyte"]) for e in exemplars):
                pct += float(l["contribution_pct"])
                matched.append(_norm_name(l["analyte"]))
        return min(1.0, pct / self.sat), sorted(set(matched))

    def _perturbation(self, contributors, exemplars, parent_theme):
        """Aggregate perturbation evidence (dose / serum-spike / depletion) from the
        contributing components — links the motif back to the calibration studies.

        Dose responsiveness and serum-spike activation are matched to the motif's
        exemplar chemistries. The uricase-depletion study is purine-specific, so its
        deltas are attached ONLY to purine-parented motifs (elsewhere they are
        off-target and would be misleading provenance)."""
        dose_resp, spike_hits, depletion = [], [], []
        purine_motif = parent_theme == "nucleic_purine"
        for c in contributors:
            j = c["component"]
            n = self.reg.value(j, "n_dose_experiments_responsive")
            if n:
                dose_resp.append({"component": j, "n_dose_experiments": int(n)})
            spike = self.reg.value(j, "serum_spike_evidence") or {}
            for a in (spike.get("top_activators", []) if isinstance(spike, dict) else []):
                name = a.get("analyte") if isinstance(a, dict) else str(a)
                if any(_name_matches(e, name) for e in exemplars):
                    spike_hits.append({"component": j, "analyte": _norm_name(name),
                                       "delta": a.get("delta") if isinstance(a, dict) else None})
            dep = self.reg.value(j, "depletion_evidence") or {}
            if purine_motif and isinstance(dep, dict) and dep.get("uricase_delta"):
                depletion.append({"component": j, "uricase_delta": dep["uricase_delta"],
                                  "note": dep.get("note", "")})
        return {"dose_responsive_components": dose_resp, "serum_spike_matches": spike_hits,
                "depletion_matches": depletion}

    def _derive(self, m):
        parent = m["parent_theme"]
        ti = self.onto.theme_index(parent)
        rows = []
        for j in range(self.k):
            band = self._band_score(j, m["bands_cm"])
            exemplar, matched = self._exemplar_score(j, m["exemplars"])
            theme = float(self.onto.W[j, ti])
            raw = self.wb * band + self.we * exemplar + self.wt * theme
            if raw >= self.keep:
                rows.append({"component": j, "raw": raw, "band": round(band, 3),
                             "exemplar": round(exemplar, 3), "theme": round(theme, 3),
                             "matched_analytes": matched})
        rows.sort(key=lambda r: -r["raw"])
        rows = rows[: self.max_contrib]
        total = sum(r["raw"] for r in rows) or 1.0
        contributors = [{"component": r["component"], "weight": round(r["raw"] / total, 4),
                         "band": r["band"], "exemplar": r["exemplar"], "theme": r["theme"],
                         "matched_analytes": r["matched_analytes"]} for r in rows]
        # confidence = weighted component stability x evidence breadth
        stability = float(sum(c["weight"] * self.stab[c["component"]] for c in contributors))
        breadth = float(np.mean([((c["band"] > 0) + (c["exemplar"] > 0) + (c["theme"] > 0)) / 3
                                 for c in contributors])) if contributors else 0.0
        ref_analytes = sorted({a for c in contributors for a in c["matched_analytes"]})
        mot = MSSMotif(id=m["id"], name=m["name"], description=m["description"],
                       bands_cm=m["bands_cm"], exemplars=m["exemplars"], parent_theme=parent,
                       non_biochemical=bool(m.get("non_biochemical", False)),
                       contributors=contributors, confidence=round(stability * breadth, 4),
                       stability=stability, evidence_breadth=breadth,
                       reference_analytes=ref_analytes)
        mot.perturbation = self._perturbation(contributors, m["exemplars"], parent)
        return mot

    def _weight_matrix(self):
        M = np.zeros((self.k, len(self.motifs)))
        for mi, mot in enumerate(self.motifs):
            for c in mot.contributors:
                M[c["component"], mi] = c["weight"]
        return M

    # ── application to a query (read-only on the BSV) ──
    def activate(self, bsv):
        """Project a BSV's frozen component coordinates onto the MSS motifs.

        Returns motif activations sorted by elevation (most-elevated first) — the
        answer to "which spectral motifs explain this state / change?".
        """
        coord = np.asarray(bsv.component_coord, float)
        z = np.asarray(bsv.component_z, float)
        comp = self.M.T @ coord
        elev = self.M.T @ z
        out = []
        for mi, mot in enumerate(self.motifs):
            top = sorted(mot.contributors, key=lambda c: -c["weight"] * coord[c["component"]])[:3]
            top_components = [(c["component"], c["weight"], round(float(coord[c["component"]]), 4))
                              for c in top]
            out.append(MSSActivation(
                id=mot.id, name=mot.name, parent_theme=mot.parent_theme,
                non_biochemical=mot.non_biochemical,
                composition=float(comp[mi]), elevation=float(elev[mi]),
                display=float(0.5 + 0.5 * np.tanh(elev[mi] / 3)),
                confidence=mot.confidence, top_components=top_components))
        out.sort(key=lambda a: -a.elevation)
        return out

    def biochemical(self):
        return [m for m in self.motifs if not m.non_biochemical]

    def registry(self):
        """Serialisable, provenance-carrying MSS registry (Part-3-style artifact)."""
        return {
            "version": self.version,
            "atlas_fingerprint": VERSIONS.atlas_fingerprint,
            "ontology_version": self.onto.version,
            "band_match_tolerance_cm": self.tol,
            "derivation": {"weights": {"band": self.wb, "exemplar": self.we, "theme": self.wt},
                           "keep_threshold": self.keep, "max_contributors": self.max_contrib,
                           "exemplar_saturation_pct": self.sat},
            "n_motifs": len(self.motifs),
            "motifs": [m.as_dict() for m in self.motifs],
        }

    @classmethod
    def from_engine(cls, engine):
        """Share an already-loaded engine's frozen ontology + registry."""
        return cls(ontology=engine.builder.onto, registry=engine.builder.reg)
