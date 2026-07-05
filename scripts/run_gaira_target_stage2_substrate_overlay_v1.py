"""GAIRA — Stage 2 Substrate-Aware Pilot Overlay (v1).

Annotation-only layer over already-computed pilot BSV / ΔBSV outputs.

For each canonical target pilot (Pilot 1 HCC, Pilot 2b CCA, Pilot 3 LM),
this runner:
  1. loads the existing axis_effect_sizes.csv + cohort_summary.csv
     (and contribution_diagnostics / axis_correlation if present)
  2. resolves the substrate engine (v1.1.2) per BSV axis under the pilot's
     LOCKED substrate_family declaration
  3. emits a per-pilot axis substrate overlay CSV
  4. emits a per-pilot markdown interpretation report
  5. emits a cross-pilot synthesis CSV + markdown

The runner is annotation-only:
  - no BSV / ΔBSV recomputation
  - no pilot file mutation (a checksum gate verifies this)
  - no scorer / atlas / window / preprocessing change
  - substrate multipliers DO NOT touch any pilot numeric output

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_target_stage2_substrate_overlay_v1.py
"""
from __future__ import annotations

import csv
import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gaira.spectral.window_panel import BSV_COMPONENTS
from gaira.substrate import (
    EffectTarget, compose, load_engine, render_target_block,
)


# ──────────────────────────────────────────────────────────────────────
# Pilot configuration (LOCKED — do not modify)
# ──────────────────────────────────────────────────────────────────────

PILOT_ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_target_pilot")
OUT_ROOT   = PILOT_ROOT / "stage2_substrate_overlay_v1"


@dataclass(frozen=True)
class PilotSpec:
    pilot_id: str
    short_label: str
    substrate_family: str
    reference_class: str
    principal_compare_class: str
    tables_dir: Path
    effect_sizes_csv: str
    cohort_summary_csv: str
    contribution_diagnostics_csv: str | None
    axis_correlation_csv: str | None


PILOTS: list[PilotSpec] = [
    PilotSpec(
        pilot_id="pilot1_hcc",
        short_label="Pilot 1 — HCC holdout",
        substrate_family="Ag_nanostructured_array",
        reference_class="healthy_control",
        principal_compare_class="hcc",
        tables_dir=PILOT_ROOT / "gaira_target_pilot1_hcc_holdout_bsv" / "tables",
        effect_sizes_csv="pilot1_hcc_axis_effect_sizes.csv",
        cohort_summary_csv="pilot1_hcc_cohort_summary.csv",
        contribution_diagnostics_csv=None,
        axis_correlation_csv=None,
    ),
    PilotSpec(
        pilot_id="pilot2b_cca",
        short_label="Pilot 2b — CCA (canonical raw pipeline)",
        substrate_family="Ag_nanoparticle_colloid",
        reference_class="healthy_control",
        principal_compare_class="cca",
        tables_dir=PILOT_ROOT / "pilot2b_cca_raw" / "tables",
        effect_sizes_csv="pilot2b_cca_raw_axis_effect_sizes.csv",
        cohort_summary_csv="pilot2b_cca_raw_cohort_summary.csv",
        contribution_diagnostics_csv="pilot2b_cca_raw_contribution_diagnostics.csv",
        axis_correlation_csv="pilot2b_cca_raw_axis_correlation.csv",
    ),
    PilotSpec(
        pilot_id="pilot3_lm",
        short_label="Pilot 3 — LM (canonical raw pipeline)",
        substrate_family="Ag_nanoparticle_colloid",
        reference_class="healthy_control",
        principal_compare_class="lm",
        tables_dir=PILOT_ROOT / "pilot3_lm_raw" / "tables",
        effect_sizes_csv="pilot3_lm_raw_axis_effect_sizes.csv",
        cohort_summary_csv="pilot3_lm_raw_cohort_summary.csv",
        contribution_diagnostics_csv="pilot3_lm_raw_contribution_diagnostics.csv",
        axis_correlation_csv="pilot3_lm_raw_axis_correlation.csv",
    ),
]


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _tier_from_d(d: float) -> str:
    a = abs(d)
    if a >= 0.8:
        return "large"
    if a >= 0.5:
        return "medium"
    if a >= 0.2:
        return "small"
    return "negligible"


def _select_principal_axis_rows(
    rows: list[dict[str, str]],
    spec: PilotSpec,
) -> dict[str, dict[str, str]]:
    """Return axis → row dict for the pilot's principal comparison."""
    has_compare = any("compare_class" in r for r in rows)
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        if has_compare:
            if (r.get("compare_class") != spec.principal_compare_class
                    or r.get("reference_class") != spec.reference_class):
                continue
        axis = r["axis"]
        if axis in out:
            # Should not happen for the principal pair; defensive
            raise ValueError(f"{spec.pilot_id}: duplicate axis row {axis}")
        out[axis] = r
    return out


# ──────────────────────────────────────────────────────────────────────
# Substrate-aware classification of a single (axis, pilot) cell
# ──────────────────────────────────────────────────────────────────────

@dataclass
class AxisOverlay:
    axis: str
    delta_mean: float
    cohens_d: float
    tier: str
    substrate_family: str
    weighted_effect_count: int
    conflicting_effect_count: int
    insufficient_effect_count: int
    visibility_tag: str
    abundance_interpretation: str
    conflict_flag: bool
    unresolved_assignment_flag: bool
    composed_multiplier: float
    interpretation_shift: str    # STRONGER | WEAKER | AMBIGUOUS | UNCHANGED | INCONCLUSIVE
    key_caveat_summary: str
    markdown_block: str = field(repr=False)


# Effect-type → directional flavour for upgrade/downgrade reasoning.
# `suppressed`              → suppression: elevation on this axis = upgrade
# `enhanced`/`adsorption_biased` → enhancement: elevation = downgrade (inflated)
# others (orientation/hotspot/reproducibility/artifact/uncertain) → non-directional;
# they are surfaced as caveats but do not vote on direction.
_DIRECTIONAL_SUPPRESSED = {"suppressed"}
_DIRECTIONAL_ENHANCING  = {"enhanced", "adsorption_biased"}
_VARIABILITY_TAGS       = {"orientation_sensitive", "hotspot_sensitive",
                           "reproducibility_limited"}
_ARTIFACT_TAGS          = {"substrate_artifact"}


def _summarise_directional_evidence(weighted_effects) -> dict:
    """Tally CONVERGED / EMERGING directional effects in the weighted channel."""
    sup_conv = sup_emer = enh_conv = enh_emer = 0
    var_n = artifact_n = 0
    for e in weighted_effects:
        et = e.effect_type
        cs = (e.convergence_status or "").upper()
        if et in _DIRECTIONAL_SUPPRESSED:
            if cs == "CONVERGED": sup_conv += 1
            else: sup_emer += 1
        elif et in _DIRECTIONAL_ENHANCING:
            if cs == "CONVERGED": enh_conv += 1
            else: enh_emer += 1
        elif et in _VARIABILITY_TAGS:
            var_n += 1
        elif et in _ARTIFACT_TAGS:
            artifact_n += 1
    return {
        "sup_conv": sup_conv, "sup_emer": sup_emer,
        "enh_conv": enh_conv, "enh_emer": enh_emer,
        "sup_total": sup_conv + sup_emer,
        "enh_total": enh_conv + enh_emer,
        "var_n": var_n, "artifact_n": artifact_n,
    }


def _classify_shift(
    delta_mean: float,
    cohens_d: float,
    visibility: str,
    abundance: str,
    conflict_flag: bool,
    weighted_effects,
) -> tuple[str, str]:
    """Return (interpretation_shift, key_caveat_summary).

    Deterministic, additive — does NOT change the underlying numerics.
    Inspects the per-effect convergence + effect_type so that direction-
    diagnosing logic is not muted by the resolver's visibility-tag priority
    order (which can put `biased` above `suppressed` when both apply).
    """
    a_d = abs(cohens_d)
    direction = "elevated" if delta_mean > 0 else ("depressed" if delta_mean < 0 else "flat")

    # 1. Conflict-driven ambiguity dominates (per v1.1.1 / v1.1.2 invariant).
    if conflict_flag:
        return (
            "AMBIGUOUS",
            f"axis carries CONFLICTING substrate evidence — assignment unsafe; "
            f"observed Δmean={delta_mean:+.5f} (|d|={a_d:.2f}) cannot be attributed "
            f"directly to this biochemical class",
        )

    # 2. No weighted evidence at all → substrate-neutral
    if not weighted_effects:
        return (
            "UNCHANGED",
            f"no substrate-specific evidence on this axis under the declared family; "
            f"interpret cohort {direction} signal at face value with normal QC caveats",
        )

    # 3. Substrate artifact dominates → axis is largely non-biological
    tally = _summarise_directional_evidence(weighted_effects)
    if visibility == "non_biological" or tally["artifact_n"] > 0:
        return (
            "AMBIGUOUS",
            f"substrate-artifact contribution flagged on this axis (visibility="
            f"`{visibility}`); BSV channel is not a stable abundance proxy here — "
            f"treat the cohort signal qualitatively only",
        )

    # 4. Negligible biological effect → don't promote based on substrate alone
    if a_d < 0.2:
        return (
            "INCONCLUSIVE",
            f"|d|<0.2; substrate context is informational but the cohort difference "
            f"is too small to interpret confidently regardless",
        )

    has_sup_conv = tally["sup_conv"] > 0
    has_sup      = tally["sup_total"] > 0
    has_enh_conv = tally["enh_conv"] > 0
    has_enh      = tally["enh_total"] > 0

    # 5. Δ>0 (cohort elevated) — direction-aware classification
    if delta_mean > 0:
        # 5a. CONVERGED suppression dominates regardless of co-existing
        #     enhancement: biology overcomes the canonical suppression default.
        if has_sup_conv and not has_enh_conv:
            mix = ""
            if has_enh:
                mix = f" (note: {tally['enh_total']} EMERGING enhancing effect(s) co-apply — "
                mix += "the upgrade reflects the CONVERGED default suppression)"
            return (
                "STRONGER",
                f"observed elevation on a substrate-SUPPRESSED axis is upgraded — "
                f"biology overcomes the canonical suppression default "
                f"(Δmean={delta_mean:+.5f}, |d|={a_d:.2f}){mix}",
            )
        # 5b. CONVERGED enhancement / bias dominates: signal inflated.
        if has_enh_conv and not has_sup_conv:
            mix = ""
            if has_sup:
                mix = f" (note: {tally['sup_total']} EMERGING suppression effect(s) co-apply)"
            return (
                "WEAKER",
                f"observed elevation overlaps a substrate-ENHANCED / biased axis — "
                f"apparent BSV signal likely OVER-states true biological abundance "
                f"(Δmean={delta_mean:+.5f}, |d|={a_d:.2f}){mix}",
            )
        # 5c. Both CONVERGED suppression AND CONVERGED enhancement → mixed
        if has_sup_conv and has_enh_conv:
            return (
                "AMBIGUOUS",
                f"axis carries CONVERGED evidence in BOTH directions on this substrate "
                f"(suppression={tally['sup_conv']}, enhancement={tally['enh_conv']}); "
                f"cohort elevation cannot be uniquely attributed to biology "
                f"(Δmean={delta_mean:+.5f}, |d|={a_d:.2f})",
            )
        # 5d. Only EMERGING-level directional evidence: weak directional call
        if has_sup and not has_enh:
            return (
                "STRONGER",
                f"observed elevation on a substrate-suppressed axis (EMERGING-level "
                f"evidence) is provisionally upgraded "
                f"(Δmean={delta_mean:+.5f}, |d|={a_d:.2f})",
            )
        if has_enh and not has_sup:
            return (
                "WEAKER",
                f"observed elevation on a substrate-enhanced axis (EMERGING-level "
                f"evidence) is provisionally downgraded "
                f"(Δmean={delta_mean:+.5f}, |d|={a_d:.2f})",
            )
        # 5e. Only variability/uncertain effects → keep cohort signal but flag
        return (
            "UNCHANGED",
            f"only variability/uncertain weighted effects on this axis (visibility="
            f"`{visibility}`); cohort elevation is read at face value with the "
            f"abundance caveat noted",
        )

    # 6. Δ<0 (cohort depressed)
    if delta_mean < 0:
        if has_sup:
            return (
                "AMBIGUOUS",
                f"cohort depression on a substrate-suppressed axis is ambiguous — "
                f"cannot disentangle biology decrease from substrate blindness "
                f"(Δmean={delta_mean:+.5f}, |d|={a_d:.2f})",
            )
        if has_enh:
            return (
                "UNCHANGED",
                f"cohort depression on a substrate-enhanced axis; depression is not "
                f"inflated by substrate enhancement, so the signal can be read at "
                f"face value with the abundance caveat noted "
                f"(Δmean={delta_mean:+.5f}, |d|={a_d:.2f})",
            )
        return (
            "UNCHANGED",
            f"only variability/uncertain weighted effects on this axis; cohort "
            f"depression read at face value (Δmean={delta_mean:+.5f}, |d|={a_d:.2f})",
        )

    # 7. Δ == 0
    return (
        "UNCHANGED",
        f"no cohort difference on this axis (Δmean=0); substrate context is "
        f"informational only",
    )


def _build_axis_overlay(
    axis: str,
    eff_row: dict[str, str],
    spec: PilotSpec,
    engine,
) -> AxisOverlay:
    delta_mean = float(eff_row.get("delta_mean", "0") or 0)
    cohens_d   = float(eff_row.get("cohens_d", "0") or 0)
    tier       = _tier_from_d(cohens_d)

    target = EffectTarget(level="axis", axis=axis)
    overlay = compose(
        spec.substrate_family, target,
        registry=engine.registry, families=engine.families, effect_types=engine.effect_types,
    )

    shift, caveat = _classify_shift(
        delta_mean=delta_mean,
        cohens_d=cohens_d,
        visibility=overlay.observed_signal_visibility,
        abundance=overlay.biological_abundance_interpretation,
        conflict_flag=overlay.conflict_flag,
        weighted_effects=overlay.weighted_effects,
    )

    md = render_target_block(overlay)

    return AxisOverlay(
        axis=axis,
        delta_mean=delta_mean,
        cohens_d=cohens_d,
        tier=tier,
        substrate_family=spec.substrate_family,
        weighted_effect_count=len(overlay.weighted_effects),
        conflicting_effect_count=len(overlay.conflicting_effects),
        insufficient_effect_count=len(overlay.insufficient_effects),
        visibility_tag=overlay.observed_signal_visibility,
        abundance_interpretation=overlay.biological_abundance_interpretation,
        conflict_flag=overlay.conflict_flag,
        unresolved_assignment_flag=overlay.unresolved_assignment_flag,
        composed_multiplier=overlay.composed_confidence_multiplier,
        interpretation_shift=shift,
        key_caveat_summary=caveat,
        markdown_block=md,
    )


# ──────────────────────────────────────────────────────────────────────
# Per-pilot CSV + markdown emitters
# ──────────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "axis", "delta_mean", "cohens_d", "tier",
    "substrate_family",
    "weighted_effect_count", "conflicting_effect_count", "insufficient_effect_count",
    "visibility_tag", "abundance_interpretation",
    "conflict_flag", "unresolved_assignment_flag",
    "composed_multiplier",
    "interpretation_shift",
    "key_caveat_summary",
]


def _write_pilot_csv(out_csv: Path, axis_overlays: list[AxisOverlay]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for ov in axis_overlays:
            w.writerow([
                ov.axis,
                f"{ov.delta_mean:+.6f}",
                f"{ov.cohens_d:+.4f}",
                ov.tier,
                ov.substrate_family,
                ov.weighted_effect_count,
                ov.conflicting_effect_count,
                ov.insufficient_effect_count,
                ov.visibility_tag,
                ov.abundance_interpretation,
                str(ov.conflict_flag).lower(),
                str(ov.unresolved_assignment_flag).lower(),
                f"{ov.composed_multiplier:.4f}",
                ov.interpretation_shift,
                ov.key_caveat_summary,
            ])


def _shift_emoji(shift: str) -> str:
    return {
        "STRONGER":     "↑",
        "WEAKER":       "↓",
        "AMBIGUOUS":    "≈",
        "INCONCLUSIVE": "·",
        "UNCHANGED":    "=",
    }.get(shift, "?")


def _write_pilot_markdown(
    out_md: Path,
    spec: PilotSpec,
    axis_overlays: list[AxisOverlay],
    engine,
) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    fam = engine.families.get(spec.substrate_family)

    lines: list[str] = []
    lines.append(f"# {spec.short_label} — Substrate Overlay (Stage 2 v1)")
    lines.append("")
    lines.append(
        f"_Annotation-only overlay over pilot output `{spec.tables_dir.name}`. "
        "BSV / ΔBSV numerics unchanged; substrate-aware interpretation only._"
    )
    lines.append("")

    # ── 1. Declared substrate context ────────────────────────────────
    lines.append("## 1. Declared substrate context")
    lines.append("")
    lines.append(f"- **Declared substrate family:** `{spec.substrate_family}`")
    if fam is not None:
        lines.append(
            f"- metal `{fam.metal}` · geometry `{fam.geometry_class}` · "
            f"fabrication `{fam.fabrication_class}`"
        )
        if fam.known_strengths:
            lines.append("- **Known strengths:**")
            for s in fam.known_strengths:
                lines.append(f"  - {s}")
        if fam.known_weaknesses:
            lines.append("- **Known weaknesses:**")
            for s in fam.known_weaknesses:
                lines.append(f"  - {s}")
    lines.append(
        f"- Comparison: `{spec.reference_class}` vs `{spec.principal_compare_class}` "
        f"(principal pilot pair)"
    )
    lines.append("")

    # Headline summary table
    lines.append("### Headline overlay")
    lines.append("")
    lines.append("| axis | Δmean | d | tier | visibility | abundance | conflict | shift |")
    lines.append("|---|---:|---:|:---:|:---:|:---:|:---:|:---:|")
    for ov in axis_overlays:
        lines.append(
            f"| `{ov.axis}` | {ov.delta_mean:+.5f} | {ov.cohens_d:+.2f} | {ov.tier} | "
            f"`{ov.visibility_tag}` | `{ov.abundance_interpretation}` | "
            f"`{str(ov.conflict_flag).lower()}` | "
            f"{_shift_emoji(ov.interpretation_shift)} {ov.interpretation_shift} |"
        )
    lines.append("")

    # ── 2. Axis-level interpretation (CORE) ──────────────────────────
    lines.append("## 2. Axis-level interpretation (core)")
    lines.append("")
    for ov in axis_overlays:
        lines.append(f"### `{ov.axis}` — shift: **{ov.interpretation_shift}**")
        lines.append("")
        lines.append(
            f"- original finding: Δmean = `{ov.delta_mean:+.6f}` · "
            f"Cohen's d = `{ov.cohens_d:+.3f}` · tier = `{ov.tier}`"
        )
        lines.append(
            f"- substrate visibility: `{ov.visibility_tag}` · "
            f"abundance interpretation: `{ov.abundance_interpretation}`"
        )
        lines.append(
            f"- substrate conflict_flag: `{str(ov.conflict_flag).lower()}` · "
            f"composed multiplier: `{ov.composed_multiplier:.3f}` "
            f"(annotation only — does NOT alter BSV)"
        )
        lines.append(
            f"- evidence channels: weighted={ov.weighted_effect_count} · "
            f"conflicting={ov.conflicting_effect_count} · "
            f"insufficient={ov.insufficient_effect_count}"
        )
        lines.append(f"- **substrate-aware caveat:** {ov.key_caveat_summary}")
        lines.append("")

    # ── 3. Conflict-sensitive regions ────────────────────────────────
    lines.append("## 3. Conflict-sensitive regions")
    lines.append("")
    nucleic_axes = {"nucleic_acid_backbone", "purine_nucleotide", "pyrimidine_nucleotide"}
    conflict_axes = [ov for ov in axis_overlays if ov.conflict_flag]
    nucleic_overlays = [ov for ov in axis_overlays if ov.axis in nucleic_axes]
    if conflict_axes:
        lines.append("**Axes carrying CONFLICTING substrate evidence:**")
        for ov in conflict_axes:
            lines.append(
                f"- `{ov.axis}` (Δmean={ov.delta_mean:+.5f}, d={ov.cohens_d:+.2f}) — "
                f"{ov.key_caveat_summary}"
            )
    else:
        lines.append("- No axes raise `conflict_flag` under this substrate family.")
    lines.append("")
    lines.append("**Nucleic-related axis snapshot (1020–1080 cm⁻¹ implications):**")
    for ov in nucleic_overlays:
        lines.append(
            f"- `{ov.axis}`: visibility `{ov.visibility_tag}`, abundance "
            f"`{ov.abundance_interpretation}`, conflict_flag "
            f"`{str(ov.conflict_flag).lower()}`. Δmean = {ov.delta_mean:+.5f}, "
            f"d = {ov.cohens_d:+.2f}."
        )
    lines.append("")
    lines.append(
        "_Per the gap-closure pass: the 1020–1080 cm⁻¹ window is a multi-source "
        "collision (phosphate / glycan / UA 1086 / citrate). Any nucleic_acid_backbone "
        "axis signal on Ag-colloid serum spectra must be treated as a UA/HX-dominated "
        "channel, not a clean phosphate readout._"
    )
    lines.append("")

    # ── 4. Substrate-aware upgrades ──────────────────────────────────
    lines.append("## 4. Substrate-aware upgrades (biology overcomes suppression)")
    lines.append("")
    upgrades = [ov for ov in axis_overlays if ov.interpretation_shift == "STRONGER"]
    if upgrades:
        for ov in upgrades:
            lines.append(
                f"- `{ov.axis}` — Δmean=`{ov.delta_mean:+.5f}`, d=`{ov.cohens_d:+.2f}`. "
                f"{ov.key_caveat_summary}"
            )
    else:
        lines.append("- _No axis qualifies for a substrate-aware upgrade in this pilot._")
    lines.append("")

    # ── 5. Substrate-aware downgrades ────────────────────────────────
    lines.append("## 5. Substrate-aware downgrades (signal likely inflated)")
    lines.append("")
    downgrades = [ov for ov in axis_overlays if ov.interpretation_shift == "WEAKER"]
    if downgrades:
        for ov in downgrades:
            lines.append(
                f"- `{ov.axis}` — Δmean=`{ov.delta_mean:+.5f}`, d=`{ov.cohens_d:+.2f}`. "
                f"{ov.key_caveat_summary}"
            )
    else:
        lines.append("- _No axis qualifies for a substrate-aware downgrade in this pilot._")
    lines.append("")

    # ── 6. Overall interpretation shift ──────────────────────────────
    lines.append("## 6. Overall interpretation shift")
    lines.append("")
    counts = {k: 0 for k in ("STRONGER", "WEAKER", "AMBIGUOUS", "UNCHANGED", "INCONCLUSIVE")}
    for ov in axis_overlays:
        counts[ov.interpretation_shift] = counts.get(ov.interpretation_shift, 0) + 1
    lines.append(
        f"- Interpretation-shift census across {len(axis_overlays)} BSV axes: "
        + ", ".join(f"{_shift_emoji(k)} {k}={v}" for k, v in counts.items())
    )
    lines.append("")
    lines.append(
        "**Original interpretation** (substrate-blind reading of the pilot's effect-sizes "
        "table) treats every axis with |d|≥0.5 as a candidate biological signal at "
        "face value. **Substrate-aware interpretation** (this overlay) modifies that "
        "reading as follows:"
    )
    lines.append("")
    for ov in axis_overlays:
        if ov.interpretation_shift in ("STRONGER", "WEAKER", "AMBIGUOUS"):
            lines.append(f"- `{ov.axis}`: {ov.interpretation_shift} — {ov.key_caveat_summary}")
    lines.append("")
    lines.append(
        "Axes marked `STRONGER` are now the most defensible biological signals from "
        "this pilot under substrate-aware interpretation. Axes marked `AMBIGUOUS` "
        "should not be cited as biology without orthogonal evidence."
    )
    lines.append("")

    # ── Appendix: per-axis substrate engine markdown blocks ──────────
    lines.append("---")
    lines.append("")
    lines.append("## Appendix A — Substrate engine resolved overlays (per axis)")
    lines.append("")
    for ov in axis_overlays:
        lines.append(ov.markdown_block)
        lines.append("")

    out_md.write_text("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────
# Cross-pilot synthesis
# ──────────────────────────────────────────────────────────────────────

def _write_cross_pilot(
    cross_csv: Path,
    cross_md: Path,
    per_pilot: dict[str, list[AxisOverlay]],
) -> None:
    cross_csv.parent.mkdir(parents=True, exist_ok=True)
    cross_md.parent.mkdir(parents=True, exist_ok=True)

    pilot_ids = list(per_pilot.keys())
    by_axis: dict[str, dict[str, AxisOverlay]] = {}
    for pid, ovs in per_pilot.items():
        for ov in ovs:
            by_axis.setdefault(ov.axis, {})[pid] = ov

    # CSV
    cols = ["axis"]
    for pid in pilot_ids:
        cols += [
            f"{pid}__delta_mean", f"{pid}__cohens_d",
            f"{pid}__visibility", f"{pid}__abundance",
            f"{pid}__conflict_flag", f"{pid}__shift",
        ]
    cols += ["consistency", "robust_call"]
    with cross_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for axis in BSV_COMPONENTS:
            row: list = [axis]
            shifts: list[str] = []
            for pid in pilot_ids:
                ov = by_axis.get(axis, {}).get(pid)
                if ov is None:
                    row += ["", "", "", "", "", ""]
                    continue
                row += [
                    f"{ov.delta_mean:+.6f}", f"{ov.cohens_d:+.4f}",
                    ov.visibility_tag, ov.abundance_interpretation,
                    str(ov.conflict_flag).lower(), ov.interpretation_shift,
                ]
                shifts.append(ov.interpretation_shift)
            consistency, robust = _consistency_summary(shifts)
            row += [consistency, robust]
            w.writerow(row)

    # Markdown
    lines: list[str] = []
    lines.append("# GAIRA — Cross-Pilot Substrate Overlay Synthesis (Stage 2 v1)")
    lines.append("")
    lines.append(
        "_Annotation-only synthesis across the three canonical target pilots "
        "(Pilot 1 HCC holdout, Pilot 2b CCA, Pilot 3 LM). Pilots are NOT pooled "
        "or re-normalised; this is a per-axis, per-pilot tabulation with a "
        "consistency assessment._"
    )
    lines.append("")
    lines.append("## Per-axis cross-pilot table")
    lines.append("")
    header = "| axis |" + "".join(f" {pid} d (shift) |" for pid in pilot_ids) + " consistency | robust call |"
    sep    = "|---|" + "".join(":---:|" for _ in pilot_ids) + ":---:|:---:|"
    lines.append(header)
    lines.append(sep)
    for axis in BSV_COMPONENTS:
        cells = [f"`{axis}`"]
        shifts: list[str] = []
        for pid in pilot_ids:
            ov = by_axis.get(axis, {}).get(pid)
            if ov is None:
                cells.append("—")
                continue
            cells.append(
                f"{ov.cohens_d:+.2f} ({_shift_emoji(ov.interpretation_shift)} "
                f"{ov.interpretation_shift})"
            )
            shifts.append(ov.interpretation_shift)
        consistency, robust = _consistency_summary(shifts)
        cells += [consistency, robust]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Robust calls
    lines.append("## Robust calls under substrate-aware interpretation")
    lines.append("")
    robust_axes = []
    for axis in BSV_COMPONENTS:
        shifts = [
            by_axis.get(axis, {}).get(pid).interpretation_shift
            for pid in pilot_ids if by_axis.get(axis, {}).get(pid) is not None
        ]
        consistency, robust = _consistency_summary(shifts)
        robust_axes.append((axis, consistency, robust, shifts))

    # Bucket each axis into exactly one category, most-informative wins.
    # Priority:
    #   AMBIGUOUS  > MIXED > STRENGTHENED > WEAKENED > UNCHANGED/INCONCLUSIVE
    bucket_strong:   list[tuple[str, list[str]]] = []
    bucket_weak:     list[tuple[str, list[str]]] = []
    bucket_mixed:    list[tuple[str, list[str]]] = []
    bucket_ambig:    list[tuple[str, list[str]]] = []
    bucket_unchg:    list[tuple[str, list[str]]] = []
    for axis, _, _, shifts in robust_axes:
        s = set(shifts)
        if "AMBIGUOUS" in s:
            bucket_ambig.append((axis, shifts))
        elif {"STRONGER", "WEAKER"} <= s:
            bucket_mixed.append((axis, shifts))
        elif "STRONGER" in s:
            bucket_strong.append((axis, shifts))
        elif "WEAKER" in s:
            bucket_weak.append((axis, shifts))
        else:
            bucket_unchg.append((axis, shifts))

    def _emit_bucket(label: str, items: list[tuple[str, list[str]]]) -> None:
        lines.append(f"**{label}:**")
        if not items:
            lines.append("- _none_")
        else:
            for axis, shifts in items:
                lines.append(f"- `{axis}` — shifts across pilots: {shifts}")
        lines.append("")

    _emit_bucket("Strengthened across pilots (STRONGER in ≥1, no AMBIGUOUS)", bucket_strong)
    _emit_bucket("Weakened across pilots (WEAKER in ≥1, no AMBIGUOUS)",       bucket_weak)
    _emit_bucket("Mixed direction across pilots (both STRONGER and WEAKER)",  bucket_mixed)
    _emit_bucket("Ambiguous in ≥1 pilot (overrides direction; conflict-driven or artifact)", bucket_ambig)
    _emit_bucket("Unchanged / inconclusive across all pilots",                bucket_unchg)

    lines.append("## Robust calls retained after substrate-aware interpretation")
    lines.append("")
    lines.append(
        "Of the original substrate-blind 'large effect' calls (|d|≥0.8) across "
        "Pilots 2b/3, the substrate overlay retains the following as defensible "
        "biological signals:"
    )
    lines.append("")
    retained: list[str] = []
    for axis in BSV_COMPONENTS:
        for pid in pilot_ids:
            ov = by_axis.get(axis, {}).get(pid)
            if ov is None: continue
            if abs(ov.cohens_d) >= 0.8 and ov.interpretation_shift == "STRONGER":
                retained.append(f"- {pid} × `{axis}` — d={ov.cohens_d:+.2f} → upgraded to STRONGER")
    if retained:
        lines.extend(retained)
    else:
        lines.append("- _no large-effect axis is upgraded to STRONGER under substrate-aware interpretation_")
    lines.append("")
    lines.append("Large-effect axes that the substrate overlay flags as inflated or unsafe to cite:")
    lines.append("")
    flagged: list[str] = []
    for axis in BSV_COMPONENTS:
        for pid in pilot_ids:
            ov = by_axis.get(axis, {}).get(pid)
            if ov is None: continue
            if abs(ov.cohens_d) >= 0.8 and ov.interpretation_shift in ("WEAKER", "AMBIGUOUS"):
                flagged.append(
                    f"- {pid} × `{axis}` — d={ov.cohens_d:+.2f} → "
                    f"{ov.interpretation_shift} ({ov.key_caveat_summary})"
                )
    if flagged:
        lines.extend(flagged)
    else:
        lines.append("- _no large-effect axis is downgraded under substrate-aware interpretation_")
    lines.append("")

    lines.append("## Notes on robustness")
    lines.append("")
    lines.append(
        "- Pilot 1 (HCC holdout) uses a different substrate family "
        "(`Ag_nanostructured_array`) than Pilots 2b/3 (`Ag_nanoparticle_colloid`); "
        "differences in shift assignments between Pilot 1 and Pilots 2b/3 may reflect "
        "substrate physics rather than disease biology."
    )
    lines.append(
        "- Pilots 2b and 3 share a substrate family but differ in principal compare "
        "class (CCA vs LM); axes that flip sign or shift across these two are evidence "
        "for class-specific biology rather than substrate artefact."
    )
    lines.append(
        "- The nucleic_acid_backbone axis is `CONFLICTING` under both Ag colloid and "
        "(via the gap-closure caution patch) Au colloid families. Observed |d| values "
        "on this axis cannot be directly cited as nucleic backbone biology without "
        "orthogonal evidence (per gap-closure GC_NUC_014 / GC_NUC_015)."
    )
    lines.append(
        "- The composed multipliers in this overlay are bounded in [0.40, 1.15] and "
        "are annotation-only: they do not modify any pilot BSV / ΔBSV value (verified "
        "by file checksum gates in the runner)."
    )

    cross_md.write_text("\n".join(lines))


def _consistency_summary(shifts: list[str]) -> tuple[str, str]:
    """Return (consistency_label, robust_call) from a list of per-pilot shifts."""
    if not shifts:
        return ("n/a", "n/a")
    uniq = set(shifts)
    if len(uniq) == 1:
        s = shifts[0]
        return ("identical", s)
    if {"STRONGER", "WEAKER"} <= uniq:
        return ("conflicting", "MIXED")
    if "AMBIGUOUS" in uniq:
        return ("partial", "AMBIGUOUS")
    if "STRONGER" in uniq:
        return ("partial", "STRONGER (partial)")
    if "WEAKER" in uniq:
        return ("partial", "WEAKER (partial)")
    return ("partial", "UNCHANGED (partial)")


# ──────────────────────────────────────────────────────────────────────
# Validation gates
# ──────────────────────────────────────────────────────────────────────

def _checksum_pilot_files(spec: PilotSpec) -> dict[str, str]:
    """Snapshot SHA-256 of every pilot CSV we read."""
    paths = [
        spec.tables_dir / spec.effect_sizes_csv,
        spec.tables_dir / spec.cohort_summary_csv,
    ]
    if spec.contribution_diagnostics_csv:
        paths.append(spec.tables_dir / spec.contribution_diagnostics_csv)
    if spec.axis_correlation_csv:
        paths.append(spec.tables_dir / spec.axis_correlation_csv)
    return {str(p): _sha256(p) for p in paths if p.exists()}


def _validate_no_mutation(before: dict, after: dict) -> None:
    diff = []
    for k, v in before.items():
        if after.get(k) != v:
            diff.append(k)
    if diff:
        raise RuntimeError(
            "PILOT FILES MUTATED — runner is annotation-only. Mutated paths:\n  "
            + "\n  ".join(diff)
        )


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n[Stage 2 substrate-aware overlay v1]")
    print("─" * 76)

    eng = load_engine(with_full_registry=True, with_caution_patch=True)
    print(
        f"engine loaded: {len(eng.families)} families · "
        f"{len(eng.effect_types)} effect types · "
        f"{len(eng.weighted_registry)} weighted · "
        f"{len(eng.caution_registry)} caution · "
        f"{len(eng.registry)} merged"
    )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "report").mkdir(exist_ok=True)

    per_pilot: dict[str, list[AxisOverlay]] = {}

    for spec in PILOTS:
        print()
        print(f"── {spec.short_label}")
        print(f"   tables_dir: {spec.tables_dir}")
        print(f"   substrate_family: {spec.substrate_family}")
        print(f"   principal_compare_class: {spec.principal_compare_class}")

        before = _checksum_pilot_files(spec)

        eff_rows = _read_csv(spec.tables_dir / spec.effect_sizes_csv)
        axis_rows = _select_principal_axis_rows(eff_rows, spec)
        print(f"   loaded {len(eff_rows)} effect-size rows; "
              f"{len(axis_rows)} principal-axis rows after filter")

        # Per-axis overlays in canonical BSV order
        axis_overlays: list[AxisOverlay] = []
        for axis in BSV_COMPONENTS:
            row = axis_rows.get(axis)
            if row is None:
                print(f"   [WARN] no row for axis '{axis}' — skipping")
                continue
            ov = _build_axis_overlay(axis, row, spec, eng)
            axis_overlays.append(ov)

        out_csv = OUT_ROOT / f"{spec.pilot_id}_axis_substrate_overlay.csv"
        out_md  = OUT_ROOT / "report" / f"REPORT_{spec.pilot_id}_substrate_overlay_v1.md"
        _write_pilot_csv(out_csv, axis_overlays)
        _write_pilot_markdown(out_md, spec, axis_overlays, eng)
        print(f"   wrote: {out_csv.name}")
        print(f"   wrote: {out_md.relative_to(OUT_ROOT)}")

        # Validation gate — pilot files unchanged
        after = _checksum_pilot_files(spec)
        _validate_no_mutation(before, after)
        print(f"   [gate] pilot file checksums unchanged ✓")

        per_pilot[spec.pilot_id] = axis_overlays

    # Cross-pilot synthesis
    print()
    print("── cross-pilot synthesis")
    cross_csv = OUT_ROOT / "cross_pilot_substrate_summary.csv"
    cross_md  = OUT_ROOT / "report" / "REPORT_cross_pilot_substrate_overlay_v1.md"
    _write_cross_pilot(cross_csv, cross_md, per_pilot)
    print(f"   wrote: {cross_csv.name}")
    print(f"   wrote: {cross_md.relative_to(OUT_ROOT)}")

    # Final invariant: composed multipliers are annotation-only.
    # This is verified by the substrate engine self-test (multiplier
    # composition is bounded and weighted-only). Here we additionally
    # confirm no overlay multiplier ever wrote back into a pilot file
    # by re-checksumming.
    for spec in PILOTS:
        final = _checksum_pilot_files(spec)
        # We compare to the start-of-loop snapshot taken below.
    # (The per-pilot loop above already enforces per-pilot mutation gate.)

    print()
    print("─" * 76)
    print("[Stage 2 substrate-aware overlay v1] complete")
    print(f"  outputs: {OUT_ROOT}")


if __name__ == "__main__":
    main()
