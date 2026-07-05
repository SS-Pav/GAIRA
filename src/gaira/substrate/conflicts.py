"""GAIRA Substrate Engine v1.1.1 — conflict & evidence-channel routing.

Centralises the rule that maps `convergence_status` → `effect_channel`.

The three channels carry different semantic load and must be kept separated:

  - **weighted**     — CONVERGED + admitted EMERGING; eligible for the
                       composed confidence multiplier.
  - **conflicting**  — CONFLICTING; surfaces caution / unresolved-assignment
                       notes only. Never enters the multiplier composition.
  - **insufficient** — INSUFFICIENT; surfaces a "no reliable substrate evidence"
                       caveat only. Never enters the multiplier composition.

This module is deliberately tiny: classification is a pure function of
`convergence_status`. Anything more clever lives in resolver.py.
"""
from __future__ import annotations

from gaira.substrate.schema import (
    ConflictReport, ConvergenceStatus, EffectChannel, ResolvedEffect,
    SubstrateBandEffect,
)


# ──────────────────────────────────────────────────────────────────────
# convergence_status → effect_channel
# ──────────────────────────────────────────────────────────────────────

# Anything not listed here is treated as "weighted" by default
# (matching pre-v1.1.1 behavior for legacy entries lacking a status).
_STATUS_TO_CHANNEL: dict[str, EffectChannel] = {
    "CONVERGED":    "weighted",
    "EMERGING":     "weighted",
    "CONFLICTING":  "conflicting",
    "INSUFFICIENT": "insufficient",
}


def classify_channel(status: ConvergenceStatus | str | None) -> EffectChannel:
    """Pure-function classifier: convergence_status → effect_channel."""
    if status is None:
        return "weighted"
    return _STATUS_TO_CHANNEL.get(str(status), "weighted")


def classify_effect(eff: SubstrateBandEffect) -> EffectChannel:
    """Convenience wrapper for SubstrateBandEffect."""
    return classify_channel(eff.convergence_status)


# ──────────────────────────────────────────────────────────────────────
# Channel splitting + ConflictReport assembly
# ──────────────────────────────────────────────────────────────────────

def split_by_channel(
    effects: list[ResolvedEffect],
) -> tuple[list[ResolvedEffect], list[ResolvedEffect], list[ResolvedEffect]]:
    """Partition a list of ResolvedEffect into (weighted, conflicting, insufficient)."""
    weighted:    list[ResolvedEffect] = []
    conflicting: list[ResolvedEffect] = []
    insufficient: list[ResolvedEffect] = []
    for e in effects:
        ch = e.effect_channel
        if ch == "conflicting":
            conflicting.append(e)
        elif ch == "insufficient":
            insufficient.append(e)
        else:
            weighted.append(e)
    return weighted, conflicting, insufficient


def make_conflict_report(
    conflicting_effects: list[ResolvedEffect],
    *,
    candidate_classes_by_id: dict[str, str] | None = None,
    spectral_regions_by_id: dict[str, str] | None = None,
    explicit_notes_by_id: dict[str, str] | None = None,
) -> ConflictReport:
    """Build a structured ConflictReport from the conflicting-channel effects.

    `candidate_classes_by_id` and `spectral_regions_by_id` are optional
    side-channel maps from the SubstrateBandEffect that the resolver passes
    through (so the report can name competing biochemical classes and the
    region descriptions of the conflict, without re-reading the registry).
    """
    if not conflicting_effects:
        return ConflictReport(
            has_conflict=False,
            conflicting_effect_ids=(),
            conflict_notes=(),
            candidate_assignment_classes=(),
            spectral_regions=(),
        )
    ids = tuple(sorted({e.effect_id for e in conflicting_effects}))
    notes: list[str] = []
    classes: list[str] = []
    regions: list[str] = []
    seen_class: set[str] = set()
    seen_region: set[str] = set()
    seen_note: set[str] = set()
    for e in conflicting_effects:
        # explicit per-id note (richer than effect_summary if provided)
        n = (explicit_notes_by_id or {}).get(e.effect_id) or e.effect_summary
        if n and n not in seen_note:
            notes.append(n); seen_note.add(n)
        c = (candidate_classes_by_id or {}).get(e.effect_id)
        if c and c not in seen_class:
            classes.append(c); seen_class.add(c)
        r = (spectral_regions_by_id or {}).get(e.effect_id)
        if r and r not in seen_region:
            regions.append(r); seen_region.add(r)
    return ConflictReport(
        has_conflict=True,
        conflicting_effect_ids=ids,
        conflict_notes=tuple(notes),
        candidate_assignment_classes=tuple(classes),
        spectral_regions=tuple(regions),
    )


# ──────────────────────────────────────────────────────────────────────
# Caveat-line generators (deterministic, used by report_overlay)
# ──────────────────────────────────────────────────────────────────────

def conflict_caveat_lines(
    report: ConflictReport,
) -> list[str]:
    """Deterministic conflict-caveat lines for report markdown / overlay."""
    if not (report and report.has_conflict):
        return []
    lines: list[str] = [
        "This target also has conflicting literature assignments under the "
        "declared substrate family.",
    ]
    if report.candidate_assignment_classes:
        lines.append(
            "Signal in this region may reflect overlapping contributions across "
            "competing biochemical classes: "
            + ", ".join(f"`{c}`" for c in report.candidate_assignment_classes)
            + "."
        )
    if report.spectral_regions:
        lines.append(
            "Affected spectral region(s): "
            + ", ".join(f"`{r}`" for r in report.spectral_regions)
            + "."
        )
    lines.append(
        "Observed intensity in this region should not be interpreted as a "
        "direct abundance readout."
    )
    lines.append(
        "No directional substrate-weighting claim is being made from the "
        "conflict itself: conflicting evidence is surfaced for caution only "
        "and is excluded from the composed multiplier."
    )
    return lines


def insufficient_caveat_lines(
    insufficient_effects: list[ResolvedEffect],
) -> list[str]:
    """Deterministic insufficient-caveat lines for report markdown / overlay."""
    if not insufficient_effects:
        return []
    ids = sorted({e.effect_id for e in insufficient_effects})
    return [
        "The current evidence base does not support a unique biochemical "
        "assignment for this region under the declared substrate family.",
        "Insufficient-evidence entries are surfaced as informational only; "
        "they do not affect the composed confidence multiplier "
        f"(entries: {', '.join(f'`{i}`' for i in ids)}).",
    ]
