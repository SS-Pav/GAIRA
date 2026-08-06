"""GAIRA V7 — Phase 04, Part F: `SpectrumState`, GAIRA's canonical internal representation.

Everything the engine knows about one spectrum, in one object, with the provenance chain
resolvable at every level. The design rule: **no number appears without the uncertainty that
qualifies it.** A theme activation and its confidence are not separate results to be joined
later; they are one result.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

SCHEMA = "gaira_v7_inference_v1"
CONFIDENCE_TIERS = ("high", "moderate", "low", "out_of_domain")


@dataclass
class SpectrumState:
    # identity and provenance
    spectrum_id: str
    atlas_fingerprint: str
    lsm_fingerprint: str
    csm_fingerprint: str
    theme_fingerprint: str
    preprocessing_config_hash: str
    engine_version: str

    # quality control, computed before anything is interpreted
    qc: dict

    # the projection ladder
    lsm_activations: np.ndarray
    csm_activations: np.ndarray
    csm_uncertainty: np.ndarray
    theme_activations_all: np.ndarray        # includes rejected themes
    theme_activations: np.ndarray            # accepted themes only
    bsv: np.ndarray                          # ABSOLUTE, non-negative
    bsv_axis_names: list[str]
    bsv_elevation: np.ndarray                # signed, derived — never called `bsv`

    # geometry
    geometry_coords: np.ndarray
    nearest_csms: list[dict]
    nearest_molecules: list[dict]
    nearest_reference_spectra: list[dict]
    bridge_memberships: dict
    ood: dict

    # residual and uncertainty
    residual: dict
    uncertainty: dict
    confidence: float
    confidence_tier: str

    # explanation
    evidence: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)

    def __post_init__(self):
        for name in ("lsm_activations", "csm_activations", "theme_activations", "bsv"):
            v = np.asarray(getattr(self, name), float)
            if (v < -1e-9).any():
                raise ValueError(f"{self.spectrum_id}: {name} must be non-negative")
            setattr(self, name, v)
        if self.confidence_tier not in CONFIDENCE_TIERS:
            raise ValueError(f"unknown confidence tier {self.confidence_tier}")
        if len(self.bsv) != len(self.bsv_axis_names):
            raise ValueError("BSV and its axis names disagree in length")

    # ── the explanation chain ────────────────────────────────────────────────
    def explain(self, theme_index: int, registry: dict, csm_registry: dict,
                top: int = 3) -> dict:
        """Theme → CSMs → LSMs → canonical molecules → source spectra, for one theme.

        `theme_index` indexes the ACCEPTED themes, because that is what `theme_activations`
        holds. The membership matrix has a column for every theme including the rejected one,
        so the index is translated before use — indexing `S` directly with an accepted-theme
        index silently explained the wrong theme, and produced empty support for three of four.

        The chain is resolved from the frozen registries at call time rather than stored, so an
        explanation can never disagree with the atlas it claims to come from.
        """
        S = np.asarray(registry["S"], float)
        csm_ids = registry["csm_ids"]
        accepted = np.asarray(registry.get("accepted", np.ones(S.shape[1], bool)), bool)
        col = int(np.where(accepted)[0][theme_index])
        contrib = self.csm_activations * S[:, col]
        order = np.argsort(-contrib)[:top]
        out = []
        for i in order:
            if contrib[i] <= 0:
                continue
            cid = csm_ids[i]
            rec = csm_registry.get(cid, {})
            out.append({
                "csm_id": cid,
                "contribution": float(contrib[i]),
                "csm_activation": float(self.csm_activations[i]),
                "membership_in_theme": float(S[i, theme_index]),
                "lsms": [l["lsm_id"] for l in rec.get("contributing_lsms", [])],
                "canonical_molecules": rec.get("supporting_analytes", [])[:8],
                "n_canonical_molecules": len(rec.get("supporting_analytes", [])),
            })
        return {"theme_index": theme_index, "membership_column": col,
                "theme_id": registry["theme_ids"][col],
                "theme_name": registry["theme_names"][col],
                "activation": float(self.theme_activations[theme_index]),
                "supporting_csms": out,
                "chain": "theme → CSM → LSM → canonical molecule → source spectrum"}

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = [round(float(x), 6) for x in v.ravel()]
        d["schema"] = SCHEMA
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


def assign_confidence(qc: dict, ood: dict, residual: dict, uncertainty: dict) -> tuple[float, str]:
    """One confidence in [0, 1] and a tier, from the four things that can go wrong.

    Ordered by how badly each invalidates the answer: out-of-domain first (the atlas does not
    cover this chemistry), then reconstruction residual (the atlas covers it but does not
    explain it), then theme ambiguity, then measurement quality.
    """
    ood_s = float(ood.get("ood_score", 0.0))
    rec = float(residual.get("explained_variance", 0.0))
    amb = float(uncertainty.get("theme_entropy_normalised", 0.0))
    q = float(qc.get("quality", 1.0))
    conf = float(np.clip(np.exp(-np.clip(ood_s - 1.0, 0, None)) * rec
                         * (1.0 - 0.5 * amb) * q, 0.0, 1.0))
    if ood_s > 2.0:
        tier = "out_of_domain"
    elif conf >= 0.60:
        tier = "high"
    elif conf >= 0.35:
        tier = "moderate"
    else:
        tier = "low"
    return conf, tier
