"""Markdown overlay generators for the Substrate Engine v1.1.1 (conflict-aware).

These functions produce report-ready markdown snippets. They consume only
`ComposedOverlay` objects from `resolver.compose()` — nothing else. No
coupling to any pilot runner.

v1.1.1 patch: explicit per-channel rendering — weighted vs conflicting vs
insufficient evidence are surfaced as distinct sections, never collapsed
into a single ambiguous paragraph.
"""
from __future__ import annotations

from gaira.substrate.schema import (
    ComposedOverlay, EffectTarget, ResolvedEffect, SubstrateFamily,
)


# ──────────────────────────────────────────────────────────────────────
# Helper: target label for human reports
# ──────────────────────────────────────────────────────────────────────

def _target_label(target: EffectTarget) -> str:
    if target.level == "global":
        return "dataset-wide"
    if target.level == "axis":
        return f"axis={target.axis or '?'}"
    if target.level == "band":
        if target.window_id:
            return f"window={target.window_id}"
        if target.cm1_range:
            lo, hi = target.cm1_range
            return f"band ~{int(lo)}–{int(hi)} cm⁻¹"
        return "band (unspecified)"
    if target.level == "band_family":
        if target.axis:
            return f"band-family (axis={target.axis})"
        if target.cm1_range:
            lo, hi = target.cm1_range
            return f"band-family ~{int(lo)}–{int(hi)} cm⁻¹"
        return "band-family (unspecified)"
    if target.level == "motif":
        return f"motif {target.band_family or target.axis or ''}".strip()
    return target.level


def _render_effects_list(
    label: str, effects: tuple[ResolvedEffect, ...] | list[ResolvedEffect],
) -> list[str]:
    if not effects:
        return [f"- {label}: _none_"]
    out = [f"- {label}:"]
    for e in effects:
        srcs = (" · " + ", ".join(f"`{s}`" for s in e.provenance_sources)) \
            if e.provenance_sources else ""
        conv = f" [{e.convergence_status}]" if e.convergence_status else ""
        weighting = " · weighting_applied=`True`" if e.weighting_applied else " · weighting_applied=`False`"
        out.append(
            f"  - `{e.effect_id}` — `{e.effect_type}`{conv} "
            f"(multiplier `{e.confidence_multiplier:.2f}`){weighting}{srcs}"
        )
    return out


# ──────────────────────────────────────────────────────────────────────
# Per-target markdown block
# ──────────────────────────────────────────────────────────────────────

def render_target_block(overlay: ComposedOverlay) -> str:
    lines: list[str] = []
    tgt = _target_label(overlay.target)
    lines.append(f"### {tgt} — `{overlay.substrate_family}`")
    lines.append("")
    if overlay.substrate_blind:
        lines.append("- **substrate-blind**: no metadata declared; substrate-physics overlay suppressed.")
        for c in overlay.user_facing_caveat_lines:
            lines.append(f"  - {c}")
        return "\n".join(lines)

    # Headline (visibility, abundance, multiplier, caution, conflict flag)
    lines.append(
        f"- **visibility**: `{overlay.observed_signal_visibility}` · "
        f"**abundance interpretation**: `{overlay.biological_abundance_interpretation}`"
    )
    lines.append(
        f"- composite confidence multiplier: `{overlay.composed_confidence_multiplier:.3f}` · "
        f"caution: `{overlay.caution}` · "
        f"conflict_flag: `{overlay.conflict_flag}` · "
        f"unresolved_assignment_flag: `{overlay.unresolved_assignment_flag}`"
    )
    if overlay.weighted_multiplier_input_ids:
        lines.append(
            "- multiplier input ids (weighted-only audit trail): "
            + ", ".join(f"`{i}`" for i in overlay.weighted_multiplier_input_ids)
        )
    if overlay.convergence_labels:
        lines.append(
            "- convergence labels across resolved effects: "
            + ", ".join(f"`{s}`" for s in overlay.convergence_labels)
        )

    # Per-channel resolved-effect listings
    lines.append("")
    lines.append("**Evidence by channel**")
    lines.extend(_render_effects_list("weighted", overlay.weighted_effects))
    lines.extend(_render_effects_list("conflicting", overlay.conflicting_effects))
    lines.extend(_render_effects_list("insufficient", overlay.insufficient_effects))

    # Conflict-report breakout
    rep = overlay.conflict_report
    if rep is not None and rep.has_conflict:
        lines.append("")
        lines.append("**Conflict report**")
        lines.append(
            "- conflict_flag: `True` — this region carries CONFLICTING "
            "literature assignments under the declared substrate family."
        )
        if rep.candidate_assignment_classes:
            lines.append(
                "- competing biochemical classes: "
                + ", ".join(f"`{c}`" for c in rep.candidate_assignment_classes)
            )
        if rep.spectral_regions:
            lines.append(
                "- affected region descriptions: "
                + ", ".join(f"`{r}`" for r in rep.spectral_regions)
            )
        lines.append(
            "- conflicting evidence is excluded from the composed multiplier "
            "(weighted-only multiplier policy, v1.1.1)."
        )

    lines.append("")
    lines.append("**Interpretation caveats**")
    for c in overlay.user_facing_caveat_lines:
        lines.append(f"- {c}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Declared substrate header
# ──────────────────────────────────────────────────────────────────────

def render_declared_substrate_header(
    family_id: str,
    families: dict[str, SubstrateFamily],
    *,
    matrix_class: str | None = None,
    excitation_nm: int | float | None = None,
    capping_chemistry: str | None = None,
    confidence_in_metadata: str | None = None,
) -> str:
    lines = ["## Substrate-aware caveats (v1.1.1 overlay)", ""]
    fam = families.get(family_id)
    if fam is None:
        lines.append(f"**Declared substrate:** `{family_id}` (not in v1 vocabulary; "
                      "overlay suppressed).")
        return "\n".join(lines)
    lines.append(
        f"**Declared substrate:** `{fam.id}` · metal `{fam.metal}` · "
        f"geometry `{fam.geometry_class}` · fabrication `{fam.fabrication_class}`"
    )
    extras: list[str] = []
    if matrix_class:
        extras.append(f"matrix `{matrix_class}`")
    if excitation_nm is not None:
        extras.append(f"excitation {excitation_nm} nm")
    if capping_chemistry:
        extras.append(f"capping `{capping_chemistry}`")
    if confidence_in_metadata:
        extras.append(f"metadata confidence `{confidence_in_metadata}`")
    if extras:
        lines.append(" · ".join(extras))
    lines.append("")
    if fam.known_strengths:
        lines.append("**Known strengths of this substrate family**")
        for s in fam.known_strengths:
            lines.append(f"- {s}")
        lines.append("")
    if fam.known_weaknesses:
        lines.append("**Known weaknesses**")
        for s in fam.known_weaknesses:
            lines.append(f"- {s}")
        lines.append("")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Dataset-wide caveats convenience
# ──────────────────────────────────────────────────────────────────────

def render_dataset_wide_caveats(
    overlay_global: ComposedOverlay,
) -> str:
    lines = ["### Dataset-wide caveats", ""]
    if overlay_global.substrate_blind:
        for c in overlay_global.user_facing_caveat_lines:
            lines.append(f"- {c}")
        return "\n".join(lines)
    for c in overlay_global.user_facing_caveat_lines:
        lines.append(f"- {c}")
    if overlay_global.conflict_flag:
        lines.append("")
        lines.append(
            "_⚠ dataset-wide conflict surface: see per-target conflict reports._"
        )
    if overlay_global.insufficient_effects:
        lines.append(
            "_ℹ dataset-wide insufficient-evidence entries present "
            "(informational only, not weighted)._"
        )
    if overlay_global.provenance_sources:
        lines.append("")
        lines.append(
            "_sources (dataset-wide): "
            + ", ".join(f"`{s}`" for s in overlay_global.provenance_sources)
            + "_"
        )
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Provenance appendix
# ──────────────────────────────────────────────────────────────────────

def render_provenance_appendix(
    overlays: list[ComposedOverlay],
) -> str:
    all_sources: list[str] = []
    seen: set[str] = set()
    for ov in overlays:
        for s in ov.provenance_sources:
            if s and s not in seen:
                all_sources.append(s); seen.add(s)
    if not all_sources:
        return ""
    lines = ["### Substrate-physics sources", ""]
    for s in all_sources:
        lines.append(f"- `{s}`")
    return "\n".join(lines)
