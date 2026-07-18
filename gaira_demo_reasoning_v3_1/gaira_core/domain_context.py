"""GAIRA Demo v3 — domain-context layer (post-projection, rule-based).

ONE shared global coordinate system is used for every domain. Domain context is
applied AFTER projection and may only affect:
  * evidence ranking
  * explanation wording
  * confidence caveats

It must NEVER alter raw_bsv or the frozen global coordinates. This module
returns advisory text / reweighting hints only; it does not touch coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config as cfg


@dataclass(frozen=True)
class DomainContext:
    domain: str
    caveats: tuple[str, ...]
    # multiplicative *display-ranking* weights per axis (for evidence ordering
    # only — NOT applied to coordinates). Default 1.0.
    evidence_weight: dict


# Cross-domain evidence downweighting: when interpreting one matrix, evidence
# whose axis is known to be confounded in that matrix is ranked lower.
_EV_CAVEATS = (
    "EV pellets are membrane-rich: lipid/sterol axes (G08/G09) are matrix-"
    "expected, not necessarily disease signal.",
    "EV SERS enhancement is heterogeneous across vesicles; interpret at cohort level.",
)
_SERUM_CAVEATS = (
    "Serum Ag-SERS is dominated by adsorbed small metabolites; purine-metabolite "
    "(G02) is substrate-amplified and carotenoid-confounded near 1517 cm-1.",
    "Serum is a mixture matrix — class-level interpretation only.",
)
_SUBSTRATE_CAVEATS = {
    "Ag colloid SERS": (
        "Ag-SERS amplifies purine ring-breathing (G01) x3-10 and boosts thiol/thione "
        "(G10); the engine dampens/boosts these but molecule-level calls stay class-level.",
    ),
    "Raman": (
        "Spontaneous Raman: amide-III (G06) is reliable; no SERS adsorption bias.",
    ),
}


def get_domain_context(domain: str, substrate: str) -> DomainContext:
    d = (domain or "").lower()
    caveats: list[str] = []
    weight = {a: 1.0 for a in cfg.BSV_AXES}

    if "ev" in d or "vesicle" in d:
        caveats.extend(_EV_CAVEATS)
        # Downweight lipid/sterol as disease evidence in EV (matrix-expected).
        weight["G08_lipid_acyl_membrane"] = 0.7
        weight["G09_sterol_neutral_lipid"] = 0.7
    if "serum" in d:
        caveats.extend(_SERUM_CAVEATS)
        # Downweight substrate-amplified purine-metabolite as disease evidence.
        weight["G02_purine_metabolite"] = 0.7

    caveats.extend(_SUBSTRATE_CAVEATS.get(substrate, ()))
    return DomainContext(domain=domain, caveats=tuple(caveats), evidence_weight=weight)


def rank_axes_for_domain(coord_dict: dict[str, float], ctx: DomainContext,
                         k: int = 4) -> list[tuple[str, float, float]]:
    """Rank axes by |coordinate| * domain evidence-weight (ranking only).

    Returns [(axis, coordinate_value, weighted_rank_score)]. The returned
    coordinate_value is the ORIGINAL coordinate (unchanged); only the ordering
    uses the weight.
    """
    scored = []
    for a in cfg.BSV_AXES:
        v = float(coord_dict.get(a, 0.0))
        w = ctx.evidence_weight.get(a, 1.0)
        scored.append((a, v, abs(v) * w))
    scored.sort(key=lambda t: t[2], reverse=True)
    return scored[:k]
