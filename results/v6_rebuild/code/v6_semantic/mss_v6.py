"""GAIRA V6 — leakage-free MSS derivation.

WHY THIS FILE EXISTS
--------------------
The V1 MSS layer (src/gaira/engine/mss.py) derives each component->motif weight as

    raw = 0.40*band + 0.35*exemplar + 0.25*theme        # mss.py:196
    theme = ontology.W[component, parent_theme]         # mss.py:195

i.e. 25 % of every MSS weight is copied from the component->THEME matrix. MSS is
therefore NOT an independent spectroscopic layer, and any hierarchy that derives
themes FROM MSS would be circular. V6 removes that term.

V6 EVIDENCE LINES — all purely spectroscopic or loading-based, no theme labels:

  1  band_hit        fraction of the motif's characteristic bands matched by the
                     component's dominant peaks within +/- tol      (region-based)
  2  basis_cosine    cosine between the component's FROZEN BASIS SPECTRUM and a
                     synthetic Gaussian band profile built from the motif's bands
                     (NEW in V6 — the most direct spectroscopic evidence available)
  3  exemplar_load   share of the component's reference-analyte loading mass that
                     belongs to the motif's exemplar chemistries
  4  perturbation    dose-responsiveness / serum-spike activation matched to the
                     motif's exemplars  (LOW weight — the perturbation corpus is
                     purine-heavy, so an ablation without it is always reported)

Nothing frozen is read-write: the atlas, the registry and the ontology are inputs.
The ontology is loaded ONLY to measure leakage in the audit, never to score V6 MSS.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import re
import unicodedata

import numpy as np
import yaml

# documented, pre-stated mixing weights (not tuned against any outcome)
V6_WEIGHTS = {"band": 0.30, "basis_cosine": 0.30, "exemplar": 0.30, "perturbation": 0.10}
V6_WEIGHTS_NOPERT = {"band": 1 / 3, "basis_cosine": 1 / 3, "exemplar": 1 / 3, "perturbation": 0.0}
BAND_TOL_CM = 16.0          # same region-based tolerance as V1
BAND_SIGMA_CM = 9.0         # Gaussian sigma for the synthetic motif profile (~21 cm-1 FWHM)
KEEP_THRESHOLD = 0.12
MAX_CONTRIBUTORS = 6
EXEMPLAR_SATURATION_PCT = 12.0


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def name_matches(exemplar: str, analyte: str) -> bool:
    e, a = norm_name(exemplar), norm_name(analyte)
    if not e or not a:
        return False
    if e in a or a in e:
        return True
    et, at = set(e.split()), set(a.split())
    return bool(et & at) and (et <= at or at <= et)


def motif_profile(bands, grid, sigma=BAND_SIGMA_CM):
    """Synthetic non-negative Raman profile implied by a motif's band list."""
    g = np.asarray(grid, float)
    p = np.zeros_like(g)
    for b in bands:
        p += np.exp(-0.5 * ((g - float(b)) / sigma) ** 2)
    n = np.linalg.norm(p)
    return p / n if n > 1e-12 else p


@dataclass
class MotifV6:
    id: str
    name: str
    description: str
    bands_cm: list
    exemplars: list
    non_biochemical: bool = False
    chemical_class: str = ""          # V6: a CHEMISTRY tag, not a theme label
    contributors: list = field(default_factory=list)
    confidence: float = 0.0
    stability: float = 0.0
    evidence_breadth: float = 0.0
    spectral_purity: float = 0.0
    coverage_analytes: list = field(default_factory=list)
    reference_analytes: list = field(default_factory=list)
    perturbation: dict = field(default_factory=dict)

    def as_dict(self):
        d = dict(self.__dict__)
        for k in ("confidence", "stability", "evidence_breadth", "spectral_purity"):
            d[k] = round(float(d[k]), 4)
        return d


class MSSLayerV6:
    """Derives component->motif weights from spectroscopy and loadings only."""

    def __init__(self, spec_path, registry, basis, grid, weights=None,
                 keep=KEEP_THRESHOLD, max_contrib=MAX_CONTRIBUTORS, tol=BAND_TOL_CM):
        spec = yaml.safe_load(open(spec_path).read())
        self.version = spec["version"]
        self.reg = registry
        self.H = np.asarray(basis, float)                 # (24, n_bins) frozen basis
        self.grid = np.asarray(grid, float)
        self.k = self.H.shape[0]
        self.w = dict(weights or V6_WEIGHTS)
        self.keep, self.max_contrib, self.tol = keep, max_contrib, tol
        self.sat = EXEMPLAR_SATURATION_PCT
        self.stab = np.array([registry.stability(j) for j in range(self.k)])
        self._Hn = self.H / (np.linalg.norm(self.H, axis=1, keepdims=True) + 1e-12)
        self.raw_spec = spec["motifs"]
        self.motifs = [self._derive(m) for m in self.raw_spec]
        self.M = self._weight_matrix()
        self.motif_ids = [m.id for m in self.motifs]
        self.bio_ids = [m.id for m in self.motifs if not m.non_biochemical]

    # ── evidence lines ──
    def _band(self, j, bands):
        cb = np.asarray(self.reg.value(j, "dominant_raman_peaks_cm"), float)
        if cb.size == 0 or not bands:
            return 0.0
        return float(sum(np.any(np.abs(cb - b) <= self.tol) for b in bands)) / len(bands)

    def _basis_cosine(self, j, bands):
        """Cosine of the frozen basis spectrum against the motif's synthetic profile."""
        if not bands:
            return 0.0
        p = motif_profile(bands, self.grid)
        return float(max(0.0, np.dot(self._Hn[j], p)))

    def _exemplar(self, j, exemplars):
        loads = self.reg.value(j, "reference_analyte_loadings")
        pct, matched = 0.0, []
        for l in loads:
            if any(name_matches(e, l["analyte"]) for e in exemplars):
                pct += float(l["contribution_pct"])
                matched.append(norm_name(l["analyte"]))
        return min(1.0, pct / self.sat), sorted(set(matched))

    def _perturbation_score(self, j, exemplars):
        """Functional evidence: dose-responsiveness + spike activation matched to exemplars."""
        s = 0.0
        n = self.reg.value(j, "n_dose_experiments_responsive") or 0
        dose = self.reg.value(j, "dose_response_evidence") or []
        for de in dose:
            if any(name_matches(e, str(de.get("experiment", "")).split("::")[0]) for e in exemplars):
                s += min(1.0, abs(float(de.get("spearman_rho", 0.0))))
        spike = self.reg.value(j, "serum_spike_evidence") or {}
        for a in (spike.get("top_activators", []) if isinstance(spike, dict) else []):
            nm = a.get("analyte") if isinstance(a, dict) else str(a)
            if any(name_matches(e, nm) for e in exemplars):
                s += 0.5
        return float(min(1.0, s / 3.0)), int(n)

    # ── derivation ──
    def _derive(self, m):
        rows = []
        for j in range(self.k):
            band = self._band(j, m["bands_cm"])
            bcos = self._basis_cosine(j, m["bands_cm"])
            exe, matched = self._exemplar(j, m["exemplars"])
            pert, ndose = self._perturbation_score(j, m["exemplars"])
            raw = (self.w["band"] * band + self.w["basis_cosine"] * bcos
                   + self.w["exemplar"] * exe + self.w["perturbation"] * pert)
            if raw >= self.keep:
                rows.append({"component": j, "raw": float(raw), "band": round(band, 4),
                             "basis_cosine": round(bcos, 4), "exemplar": round(exe, 4),
                             "perturbation": round(pert, 4), "n_dose": ndose,
                             "matched_analytes": matched})
        rows.sort(key=lambda r: -r["raw"])
        rows = rows[: self.max_contrib]
        tot = sum(r["raw"] for r in rows) or 1.0
        contributors = [{**r, "weight": round(r["raw"] / tot, 4)} for r in rows]

        stability = float(sum(c["weight"] * self.stab[c["component"]] for c in contributors))
        # CORRECT breadth: fraction of the evidence lines that actually fire (int(), not np.bool_)
        breadth = float(np.mean([
            (int(c["band"] > 0) + int(c["basis_cosine"] > 0.05)
             + int(c["exemplar"] > 0) + int(c["perturbation"] > 0)) / 4
            for c in contributors])) if contributors else 0.0
        # spectral purity: how concentrated the motif's weight is on few components (1 - normalised entropy)
        wv = np.array([c["weight"] for c in contributors], float)
        if wv.size > 1:
            p = wv / wv.sum()
            purity = float(1 - (-np.sum(p * np.log(p + 1e-12)) / np.log(len(p))))
        else:
            purity = 1.0 if wv.size else 0.0
        refs = sorted({a for c in contributors for a in c["matched_analytes"]})
        return MotifV6(id=m["id"], name=m["name"], description=m.get("description", "").strip(),
                       bands_cm=m["bands_cm"], exemplars=m["exemplars"],
                       non_biochemical=bool(m.get("non_biochemical", False)),
                       chemical_class=m.get("chemical_class", ""),
                       contributors=contributors, confidence=round(stability * breadth, 4),
                       stability=stability, evidence_breadth=breadth, spectral_purity=purity,
                       reference_analytes=refs)

    def _weight_matrix(self):
        M = np.zeros((self.k, len(self.motifs)))
        for mi, mot in enumerate(self.motifs):
            for c in mot.contributors:
                M[c["component"], mi] = c["weight"]
        return M

    # ── application ──
    def activate(self, coord):
        """Motif composition for one 24-vector of component coordinates."""
        return self.M.T @ np.clip(np.asarray(coord, float), 0, None)

    def activate_bio(self, coord):
        idx = [i for i, m in enumerate(self.motifs) if not m.non_biochemical]
        return self.activate(coord)[idx]

    def motif_spectrum(self, mi):
        """The Raman spectrum this motif implies: its component weights times the frozen basis."""
        return self.M[:, mi] @ self.H

    def registry(self, fingerprint):
        return {"version": self.version, "atlas_fingerprint": fingerprint,
                "derivation": {"weights": self.w, "band_tolerance_cm": self.tol,
                               "band_sigma_cm": BAND_SIGMA_CM, "keep_threshold": self.keep,
                               "max_contributors": self.max_contrib,
                               "theme_evidence_used": False,
                               "note": "V6: no theme label, parent theme or ontology weight enters "
                                       "any MSS quantity. MSS is an independent spectroscopy layer."},
                "n_motifs": len(self.motifs),
                "motifs": [m.as_dict() for m in self.motifs]}
