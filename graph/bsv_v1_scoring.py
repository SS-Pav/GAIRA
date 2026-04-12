"""GAIRA BSV v1 — Biochemical State Vector scoring from motif differentials.

Compresses motif-level evidence into ~8 interpretable biochemical components.
Deterministic. No LLM. Graph-traceable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from graph.phaseC1_scoring import ScoredResult, MotifDifferential


# ── BSV Component Definitions ───────────────────────────────
# Each component maps from motif subfamilies/families/themes

_COMPONENT_MAP: dict[str, dict] = {
    "membrane_lipid": {
        "subfamilies": {"lipid", "cholesterol", "phosphatidylcholine", "phosphatidylserine",
                        "phosphatidylinositol", "sphingomyelin"},
        "families": {"lipid_support"},
        "themes": {"lipid", "lipid (phospholipid)", "lipid (fatty acid)", "lipid (triglyceride)"},
        "weight": 1.0,
    },
    "protein_backbone": {
        "subfamilies": {"collagen", "proline", "glycine", "alanine"},
        "families": {"protein_support", "backbone_support", "peptide_support"},
        "themes": {"protein", "peptide", "backbone"},
        "weight": 1.0,
    },
    "aromatic_amino_acid": {
        "subfamilies": {"tryptophan", "phenylalanine", "tyrosine", "histidine"},
        "families": {"amino_acid_support"},
        "themes": {"amino acid"},
        "weight": 1.0,
    },
    "purine_nucleotide": {
        "subfamilies": {"adenine", "guanine", "amp_(nucleotide)", "dtmp_(nucleotide)",
                        "deoxyadenosine", "datp"},
        "families": {"nucleic_acid_support", "purine_metabolite_support"},
        "themes": {"nucleic acid", "nucleic acid (DNA)", "purine metabolite"},
        "weight": 1.0,
    },
    "pyrimidine_nucleotide": {
        "subfamilies": {"cytosine", "thymine", "uracil"},
        "families": {"pyrimidine_support"},
        "themes": {"nucleic acid (RNA)", "pyrimidine"},
        "weight": 1.0,
    },
    "glycan_carbohydrate": {
        "subfamilies": {"glucose", "galactose", "mannose", "ribose", "glycogen", "fucose"},
        "families": {"carbohydrate_support"},
        "themes": {"carbohydrate"},
        "weight": 1.0,
    },
    "redox_metabolite": {
        "subfamilies": {"glutathione", "ergothioneine", "uric_acid", "bilirubin",
                        "carotenoid", "carotenoid_(carotene)", "dopamine", "serotonin"},
        "families": {"metabolite_support", "lipid-soluble_pigment_support"},
        "themes": {"metabolite", "lipid-soluble pigment (carotenoid)"},
        "weight": 1.0,
    },
    "nucleic_acid_backbone": {
        "subfamilies": set(),
        "families": set(),
        "themes": {"nucleic acid (nucleotide)", "membrane"},
        "weight": 0.8,
        "functional_groups": {"phosphodiester", "phosphate"},
    },
}

_STABILITY_WEIGHTS = {"STABLE": 1.0, "MIXED": 0.6, "UNSTABLE": 0.3, "INSUFFICIENT": 0.15}
_BROAD_PENALTY = 0.5  # applied to motifs with broadly-shared enrichment label
_COMPARATOR_ABSENT_PENALTY = 0.2


@dataclass
class BSVComponent:
    name: str
    raw_score: float = 0.0
    weighted_score: float = 0.0
    normalized_score: float = 0.0
    contributing_motifs: list[str] = field(default_factory=list)
    motif_count: int = 0
    dominant_stability: str = ""
    coverage_note: str = ""   # adequate | sparse | absent
    confidence: str = ""      # strong | moderate | weak | exploratory


def _assign_confidence(comp: BSVComponent) -> str:
    """Assign confidence label based on coverage, motif count, and stability."""
    if comp.coverage_note == "absent":
        return "exploratory"
    if comp.motif_count >= 3 and comp.dominant_stability == "STABLE":
        return "strong"
    if comp.motif_count >= 2 and comp.dominant_stability in ("STABLE", "MIXED"):
        return "moderate"
    if comp.motif_count >= 1:
        return "weak"
    return "exploratory"


@dataclass
class BSVVector:
    query_condition: str
    comparator_condition: str
    components: list[BSVComponent]
    total_motifs_used: int = 0


@dataclass
class BSVComparison:
    query_bsv: BSVVector
    comparator_bsv: BSVVector | None
    delta_components: list[dict] = field(default_factory=list)


def _motif_matches_component(md: MotifDifferential, comp_def: dict) -> bool:
    """Check if a motif differential belongs to a BSV component."""
    sf = md.subfamily.lower().replace(" ", "_")
    fam = md.family.lower().replace(" ", "_")
    if sf in comp_def.get("subfamilies", set()): return True
    if fam in comp_def.get("families", set()): return True
    return False


def _score_motif_contribution(md: MotifDifferential, is_query: bool = True) -> float:
    """Score a single motif's contribution to a BSV component."""
    count = md.q_direct if is_query else md.c_direct
    if count == 0: return 0.0

    base = math.log1p(count)  # log-scaled to prevent single-source dominance
    sources = md.q_sources if is_query else md.c_sources
    source_bonus = min(sources * 0.3, 2.0)

    # Stability weight
    stab_w = _STABILITY_WEIGHTS.get(md.stability_label, 0.15)

    # Broad penalty: apply when motif is shared (not differential)
    broad_p = _BROAD_PENALTY if md.interpretation == "shared" else 1.0

    # Comparator-absent penalty
    comp_p = _COMPARATOR_ABSENT_PENALTY if md.coverage_flag == "comparator_absent" else 1.0

    return (base + source_bonus) * stab_w * broad_p * comp_p


def compute_bsv(scored: ScoredResult, is_query: bool = True) -> BSVVector:
    """Compute BSV from motif differentials for one side of a comparison."""
    components = []

    for comp_name, comp_def in _COMPONENT_MAP.items():
        raw = 0.0
        contributing = []
        stabilities = []

        for md in scored.motif_differentials:
            if _motif_matches_component(md, comp_def):
                contribution = _score_motif_contribution(md, is_query)
                raw += contribution * comp_def.get("weight", 1.0)
                if contribution > 0:
                    contributing.append(md.subfamily or md.motif_id)
                    stabilities.append(md.stability_label)

        # Also check chemistry_support motifs via functional groups
        fg_set = comp_def.get("functional_groups", set())
        if fg_set:
            for md in scored.motif_differentials:
                if md.family.lower().replace(" ", "_") == "chemistry_support":
                    sf = md.subfamily.lower().replace(" ", "_")
                    if sf in fg_set:
                        contribution = _score_motif_contribution(md, is_query) * 0.5  # lower weight for chemistry-only
                        raw += contribution
                        if contribution > 0:
                            contributing.append(md.subfamily or md.motif_id)

        # Dominant stability
        if stabilities:
            from collections import Counter
            dom_stab = Counter(stabilities).most_common(1)[0][0]
        else:
            dom_stab = "INSUFFICIENT"

        coverage = "adequate" if len(contributing) >= 2 else ("sparse" if contributing else "absent")

        comp = BSVComponent(
            name=comp_name, raw_score=round(raw, 3), weighted_score=round(raw, 3),
            contributing_motifs=list(dict.fromkeys(contributing))[:5],
            motif_count=len(contributing), dominant_stability=dom_stab,
            coverage_note=coverage,
        )
        comp.confidence = _assign_confidence(comp)
        components.append(comp)

    # Normalize: scale so max component = 1.0
    max_score = max((c.raw_score for c in components), default=1.0) or 1.0
    for c in components:
        c.normalized_score = round(c.raw_score / max_score, 3)

    total_motifs = sum(c.motif_count for c in components)

    return BSVVector(
        query_condition=scored.matched_entity if is_query else scored.comparator,
        comparator_condition=scored.comparator if is_query else scored.matched_entity,
        components=components, total_motifs_used=total_motifs,
    )


def compute_bsv_comparison(scored: ScoredResult) -> BSVComparison:
    """Compute BSV for both query and comparator, plus delta."""
    query_bsv = compute_bsv(scored, is_query=True)

    comp_bsv = None
    if scored.query_mode in ("pairwise", "one_vs_rest") and scored.motif_differentials:
        comp_bsv = compute_bsv(scored, is_query=False)

    # Delta
    delta = []
    if comp_bsv:
        for qc, cc in zip(query_bsv.components, comp_bsv.components):
            d = round(qc.normalized_score - cc.normalized_score, 3)
            delta.append({
                "component": qc.name,
                "query_score": qc.normalized_score,
                "comparator_score": cc.normalized_score,
                "delta": d,
                "direction": "up" if d > 0.05 else ("down" if d < -0.05 else "flat"),
                "query_stability": qc.dominant_stability,
            })

    return BSVComparison(query_bsv=query_bsv, comparator_bsv=comp_bsv, delta_components=delta)
