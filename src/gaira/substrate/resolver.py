"""GAIRA Substrate Engine v1.1.1 — resolution + composition (conflict-aware).

The resolver answers queries of the form
"given declared substrate X, what substrate-physics context applies to
target T?" and returns a bounded, inspectable composite overlay.

First-class semantics distinguished here per v1.1 requirement:
  - **observed_signal_visibility**  (what the spectrum will *look* like)
  - **biological_abundance_interpretation** (what it *may reflect* biologically)

These are NOT the same. Low visibility does not imply low abundance.
Strong visibility does not imply high abundance.

v1.1.1 adds a third semantic split: **evidence channel**.
  - weighted     → CONVERGED + admitted EMERGING; drives multiplier + visibility
  - conflicting  → CONFLICTING; surfaces caution / unresolved-assignment notes only
  - insufficient → INSUFFICIENT; surfaces "no reliable evidence" caveat only

Composed multipliers are ALWAYS computed from `weighted` only. Conflict
and insufficient evidence are never silently absorbed into the multiplier.
"""
from __future__ import annotations

from math import prod

from gaira.substrate.conflicts import (
    classify_channel, conflict_caveat_lines, insufficient_caveat_lines,
    make_conflict_report, split_by_channel,
)
from gaira.substrate.effects import MULTIPLIER_MAX, MULTIPLIER_MIN
from gaira.substrate.schema import (
    AbundanceTag, ComposedOverlay, ConflictReport, EffectTarget, ResolvedEffect,
    SubstrateBandEffect, SubstrateEffectType, SubstrateFamily, VisibilityTag,
)


# ──────────────────────────────────────────────────────────────────────
# Effect-type → visibility / abundance mapping
# (deterministic, auditable, no learned weights)
# ──────────────────────────────────────────────────────────────────────

_EFFECT_TYPE_TO_VISIBILITY: dict[str, VisibilityTag] = {
    "enhanced":                "enhanced",
    "suppressed":              "suppressed",
    "adsorption_biased":       "biased",
    "orientation_sensitive":   "biased",
    "hotspot_sensitive":       "variable",
    "reproducibility_limited": "variable",
    "substrate_artifact":      "non_biological",
    "uncertain":               "uncertain",
}

_EFFECT_TYPE_TO_ABUNDANCE: dict[str, AbundanceTag] = {
    # When substrate preferentially enhances or adsorbs a class, the observed
    # signal is "inflated" relative to solution concentration → the signal may
    # **overestimate** true abundance.
    "enhanced":                "may_overestimate_abundance",
    "adsorption_biased":       "may_overestimate_abundance",
    # When substrate suppresses a class, the signal is "deflated" → the
    # signal may **underestimate** true abundance.
    "suppressed":              "may_underestimate_abundance",
    # Orientation / hotspot / reproducibility variability cannot be read as
    # abundance signal either way.
    "orientation_sensitive":   "abundance_not_directly_inferable",
    "hotspot_sensitive":       "abundance_not_directly_inferable",
    "reproducibility_limited": "abundance_not_directly_inferable",
    # Substrate artifact is not a biological signal at all.
    "substrate_artifact":      "abundance_not_directly_inferable",
    # Uncertain: do not commit either way.
    "uncertain":               "abundance_not_directly_inferable",
}


# Priority order for collapsing multiple effect visibilities into one.
# Non-biological > biased > suppressed > enhanced > variable > uncertain > neutral.
_VISIBILITY_PRIORITY: list[VisibilityTag] = [
    "non_biological", "biased", "suppressed", "enhanced", "variable",
    "uncertain", "neutral",
]


# ──────────────────────────────────────────────────────────────────────
# Target-matching
# ──────────────────────────────────────────────────────────────────────

def _target_matches(eff: SubstrateBandEffect, query: EffectTarget) -> bool:
    """Return True if the effect applies to the query target under v1 rules."""
    et = eff.target

    # Axis-wide global → always matches the query (once-per-report caveats)
    if et.level == "global" or (et.level == "axis" and et.axis == "all"):
        return True

    # Band level query — inherits from band, band_family, motif, axis
    if query.level == "band":
        if et.level == "band":
            if et.window_id and query.window_id and et.window_id == query.window_id:
                return True
            if et.cm1_range and query.cm1_range:
                return _ranges_overlap(et.cm1_range, query.cm1_range)
            return False
        if et.level in ("band_family", "motif", "axis"):
            if et.cm1_range and query.cm1_range and _ranges_overlap(et.cm1_range, query.cm1_range):
                return True
            if et.axis and query.axis and et.axis == query.axis:
                return True
        return False

    # Band-family level query — inherits from band, band_family, motif, axis
    if query.level == "band_family":
        if et.level in ("band", "band_family", "motif"):
            if et.cm1_range and query.cm1_range and _ranges_overlap(et.cm1_range, query.cm1_range):
                return True
            if et.axis and query.axis and et.axis == query.axis:
                return True
        if et.level == "axis" and et.axis and query.axis and et.axis == query.axis:
            return True
        return False

    # Motif-level query: if motif effects are tagged by axis / cm1_range, match analogously
    if query.level == "motif":
        if et.axis and query.axis and et.axis == query.axis:
            return True
        if et.cm1_range and query.cm1_range and _ranges_overlap(et.cm1_range, query.cm1_range):
            return True
        return False

    # Axis-level query
    if query.level == "axis":
        return bool(et.axis and query.axis and et.axis == query.axis)

    return False


def _ranges_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    lo_a, hi_a = a if a[0] <= a[1] else (a[1], a[0])
    lo_b, hi_b = b if b[0] <= b[1] else (b[1], b[0])
    return not (hi_a < lo_b or hi_b < lo_a)


# ──────────────────────────────────────────────────────────────────────
# Resolution
# ──────────────────────────────────────────────────────────────────────

_UNKNOWN_FAMILIES = {"unknown"}
_UNKNOWN_SERS = {"unknown_SERS"}


def resolve(
    family: str,
    target: EffectTarget,
    *,
    registry: dict[str, SubstrateBandEffect],
    families: dict[str, SubstrateFamily],
    effect_types: dict[str, SubstrateEffectType],
) -> list[ResolvedEffect]:
    """Return all effects that apply to the (family, target) query.

    Each ResolvedEffect carries its `effect_channel` (weighted / conflicting /
    insufficient) per v1.1.1. `weighting_applied` is set to True iff the
    effect would contribute to a composed multiplier (i.e. channel == weighted).
    """
    if family not in families:
        raise KeyError(f"Unknown substrate family: {family!r} (not in vocabulary)")

    resolved: list[ResolvedEffect] = []
    for eff in registry.values():
        if eff.substrate_family != family:
            continue
        if not _target_matches(eff, target):
            continue
        et = effect_types[eff.effect_type]
        channel = classify_channel(eff.convergence_status)
        resolved.append(
            ResolvedEffect(
                effect_id=eff.id,
                effect_type=eff.effect_type,
                bias_direction=et.bias_direction,
                convergence_status=eff.convergence_status,
                evidence_confidence=eff.evidence_confidence,
                confidence_multiplier=eff.confidence_multiplier or et.confidence_multiplier,
                caution_flag=et.caution_flag,
                target_level=eff.target.level,
                interpretation_note=et.interpretation_note,
                effect_summary=eff.summary_of_effect or eff.notes,
                provenance_sources=eff.provenance_sources,
                effect_channel=channel,
                weighting_applied=(channel == "weighted"),
            )
        )
    # Deterministic ordering
    resolved.sort(key=lambda r: (r.target_level, r.effect_type, r.effect_id))
    return resolved


# ──────────────────────────────────────────────────────────────────────
# Composition (weighted-only multiplier; conflict / insufficient surfacing)
# ──────────────────────────────────────────────────────────────────────

def _compose_visibility(weighted_effects: list[ResolvedEffect]) -> VisibilityTag:
    if not weighted_effects:
        return "neutral"
    tags = {_EFFECT_TYPE_TO_VISIBILITY[e.effect_type] for e in weighted_effects}
    for t in _VISIBILITY_PRIORITY:
        if t in tags:
            return t
    return "uncertain"


def _compose_abundance(
    weighted_effects: list[ResolvedEffect],
    *,
    has_conflict: bool,
) -> AbundanceTag:
    """Compose abundance interpretation.

    v1.1.1: any conflict in the resolved set forces
    `abundance_not_directly_inferable` regardless of the weighted-channel
    direction. Conflict cannot be silently overridden by a directional
    weighted effect.
    """
    if has_conflict:
        return "abundance_not_directly_inferable"
    if not weighted_effects:
        return "relatively_neutral_visibility"
    tags = {_EFFECT_TYPE_TO_ABUNDANCE[e.effect_type] for e in weighted_effects}
    has_over = "may_overestimate_abundance" in tags
    has_under = "may_underestimate_abundance" in tags
    has_ndi = "abundance_not_directly_inferable" in tags

    if has_over and has_under:
        return "abundance_not_directly_inferable"
    if has_over:
        return "may_overestimate_abundance"
    if has_under:
        return "may_underestimate_abundance"
    if has_ndi:
        return "abundance_not_directly_inferable"
    return "relatively_neutral_visibility"


def _compose_multiplier(weighted_effects: list[ResolvedEffect]) -> float:
    """Compose multiplier from WEIGHTED effects only (v1.1.1 invariant)."""
    if not weighted_effects:
        return 1.0
    m = prod(e.confidence_multiplier for e in weighted_effects)
    return max(MULTIPLIER_MIN, min(MULTIPLIER_MAX, m))


_ABUNDANCE_CAVEATS: dict[AbundanceTag, str] = {
    "may_underestimate_abundance": (
        "Low or absent observed signal should NOT be interpreted as low abundance: "
        "the declared substrate is known to suppress this channel."
    ),
    "may_overestimate_abundance": (
        "Strong observed signal should NOT be read as proportionally high abundance: "
        "the declared substrate preferentially adsorbs / enhances this channel."
    ),
    "abundance_not_directly_inferable": (
        "Observed signal magnitude cannot be mapped to abundance from this substrate "
        "alone; orientation / hotspot / ionic-condition / artefact / unresolved-conflict "
        "contributions are present."
    ),
    "relatively_neutral_visibility": (
        "No declared substrate-physics distortion for this target; signal may be interpreted "
        "without a substrate-specific abundance caveat."
    ),
}

_VISIBILITY_HEADLINES: dict[VisibilityTag, str] = {
    "enhanced":       "visibility: enhanced by the declared substrate",
    "suppressed":     "visibility: suppressed by the declared substrate",
    "biased":         "visibility: adsorption/orientation-biased on the declared substrate",
    "variable":       "visibility: hotspot/reproducibility-variable on the declared substrate",
    "non_biological": "visibility: partly or wholly a substrate/capping artefact",
    "uncertain":      "visibility: uncertain — literature does not converge",
    "neutral":        "visibility: no declared substrate-physics weighted evidence for this target",
}


# ──────────────────────────────────────────────────────────────────────
# Public composed API
# ──────────────────────────────────────────────────────────────────────

def compose(
    family: str,
    target: EffectTarget,
    *,
    registry: dict[str, SubstrateBandEffect],
    families: dict[str, SubstrateFamily],
    effect_types: dict[str, SubstrateEffectType],
) -> ComposedOverlay:
    """Resolve + compose into a single overlay object (conflict-aware)."""
    # Graceful degradation for unknown families
    if family in _UNKNOWN_FAMILIES:
        return _substrate_blind_overlay(family, target)

    effects = resolve(
        family, target,
        registry=registry, families=families, effect_types=effect_types,
    )

    # unknown_SERS: we still resolve whatever generic caveats exist, but
    # enforce multiplier=1.0 (no downweighting without specific evidence).
    if family in _UNKNOWN_SERS:
        return _unknown_sers_overlay(family, target, effects, registry=registry)

    # Channel split (v1.1.1)
    weighted, conflicting, insufficient = split_by_channel(effects)

    # Build conflict report (with side-channel maps from the registry so
    # the report can name competing biochem classes / region descriptions)
    candidate_classes = {
        e.effect_id: registry[e.effect_id].biochemical_target_class or ""
        for e in conflicting
        if e.effect_id in registry and registry[e.effect_id].biochemical_target_class
    }
    spectral_regions = {
        e.effect_id: registry[e.effect_id].spectral_region_description or ""
        for e in conflicting
        if e.effect_id in registry and registry[e.effect_id].spectral_region_description
    }
    explicit_notes = {
        e.effect_id: registry[e.effect_id].notes or registry[e.effect_id].summary_of_effect
        for e in conflicting if e.effect_id in registry
    }
    report = make_conflict_report(
        conflicting,
        candidate_classes_by_id=candidate_classes,
        spectral_regions_by_id=spectral_regions,
        explicit_notes_by_id=explicit_notes,
    )

    has_conflict = report.has_conflict
    vis = _compose_visibility(weighted)
    abd = _compose_abundance(weighted, has_conflict=has_conflict)
    mult = _compose_multiplier(weighted)
    # Caution: any caution_flag from any channel, OR any conflict, OR any
    # insufficient evidence raises caution.
    caution = (
        any(e.caution_flag for e in effects)
        or has_conflict
        or bool(insufficient)
    )

    # Caveats: weighted-derived headlines first, then conflict, then insufficient.
    caveats: list[str] = [_VISIBILITY_HEADLINES[vis], _ABUNDANCE_CAVEATS[abd]]
    if weighted:
        caveats.insert(
            0,
            "This target has declared substrate-biased weighted evidence that "
            "affects signal visibility.",
        )
    seen: set[str] = set(caveats)
    for e in weighted:
        if e.interpretation_note and e.interpretation_note not in seen:
            caveats.append(e.interpretation_note)
            seen.add(e.interpretation_note)

    for line in conflict_caveat_lines(report):
        if line not in seen:
            caveats.append(line); seen.add(line)
    for line in insufficient_caveat_lines(insufficient):
        if line not in seen:
            caveats.append(line); seen.add(line)

    convergence_labels = tuple(
        sorted({e.convergence_status for e in effects if e.convergence_status})
    )

    provenance: list[str] = []
    for e in effects:
        for s in e.provenance_sources:
            if s and s not in provenance:
                provenance.append(s)

    return ComposedOverlay(
        substrate_family=family,
        target=target,
        resolved_effects=tuple(effects),
        composed_confidence_multiplier=mult,
        caution=caution,
        observed_signal_visibility=vis,
        biological_abundance_interpretation=abd,
        user_facing_caveat_lines=tuple(caveats),
        convergence_labels=convergence_labels,
        provenance_sources=tuple(provenance),
        substrate_blind=False,
        # v1.1.1 channels + flags
        weighted_effects=tuple(weighted),
        conflicting_effects=tuple(conflicting),
        insufficient_effects=tuple(insufficient),
        conflict_flag=has_conflict,
        unresolved_assignment_flag=has_conflict,
        conflict_report=report,
        weighted_multiplier_input_ids=tuple(e.effect_id for e in weighted),
    )


def _substrate_blind_overlay(family: str, target: EffectTarget) -> ComposedOverlay:
    blind_caveat = (
        "substrate-blind interpretation: no substrate metadata was declared; "
        "all substrate-physics overlays are suppressed and no inference about "
        "observed visibility vs true abundance is made."
    )
    return ComposedOverlay(
        substrate_family=family,
        target=target,
        resolved_effects=(),
        composed_confidence_multiplier=1.0,
        caution=True,
        observed_signal_visibility="uncertain",
        biological_abundance_interpretation="abundance_not_directly_inferable",
        user_facing_caveat_lines=(blind_caveat,),
        convergence_labels=(),
        provenance_sources=(),
        substrate_blind=True,
        weighted_effects=(),
        conflicting_effects=(),
        insufficient_effects=(),
        conflict_flag=False,
        unresolved_assignment_flag=False,
        conflict_report=ConflictReport(False, (), (), (), ()),
        weighted_multiplier_input_ids=(),
    )


def _unknown_sers_overlay(
    family: str,
    target: EffectTarget,
    effects: list[ResolvedEffect],
    *,
    registry: dict[str, SubstrateBandEffect],
) -> ComposedOverlay:
    """unknown_SERS: maximum caution, multiplier pinned to 1.00, but channels
    are still surfaced honestly so any global INSUFFICIENT entries register."""
    weighted, conflicting, insufficient = split_by_channel(effects)
    # Force multiplier = 1.0 by ignoring any weighted multipliers
    pinned_mult = 1.0

    candidate_classes = {
        e.effect_id: registry[e.effect_id].biochemical_target_class or ""
        for e in conflicting
        if e.effect_id in registry and registry[e.effect_id].biochemical_target_class
    }
    spectral_regions = {
        e.effect_id: registry[e.effect_id].spectral_region_description or ""
        for e in conflicting
        if e.effect_id in registry and registry[e.effect_id].spectral_region_description
    }
    report = make_conflict_report(
        conflicting,
        candidate_classes_by_id=candidate_classes,
        spectral_regions_by_id=spectral_regions,
    )
    has_conflict = report.has_conflict

    caveats: list[str] = [
        "substrate declared as SERS but family unspecified: every axis carries "
        "maximum-caution; composite multiplier pinned to 1.00 (no downweighting "
        "without specific substrate evidence).",
        _ABUNDANCE_CAVEATS["abundance_not_directly_inferable"],
    ]
    seen = set(caveats)
    for line in conflict_caveat_lines(report):
        if line not in seen:
            caveats.append(line); seen.add(line)
    for line in insufficient_caveat_lines(insufficient):
        if line not in seen:
            caveats.append(line); seen.add(line)

    return ComposedOverlay(
        substrate_family=family,
        target=target,
        resolved_effects=tuple(effects),
        composed_confidence_multiplier=pinned_mult,
        caution=True,
        observed_signal_visibility="uncertain",
        biological_abundance_interpretation="abundance_not_directly_inferable",
        user_facing_caveat_lines=tuple(caveats),
        convergence_labels=tuple(
            sorted({e.convergence_status for e in effects if e.convergence_status})
        ),
        provenance_sources=tuple(
            s for e in effects for s in e.provenance_sources
        ),
        substrate_blind=False,
        weighted_effects=tuple(weighted),
        conflicting_effects=tuple(conflicting),
        insufficient_effects=tuple(insufficient),
        conflict_flag=has_conflict,
        unresolved_assignment_flag=has_conflict,
        conflict_report=report,
        # Pinned to 1.0; record audit trail explicitly (no inputs).
        weighted_multiplier_input_ids=(),
    )
