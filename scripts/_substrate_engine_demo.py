"""GAIRA Substrate Engine v1.1.2 — demonstration (Au-conflict aware).

Prints canonical (family, target) queries + their composed overlays +
report-overlay markdown snippets. Intended as a human-readable walkthrough
of the engine's conflict-aware semantic output, including the v1.1.2
Au-side nucleic-related conflict surfacing.

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/_substrate_engine_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gaira.substrate import (
    EffectTarget, compose, load_engine, render_target_block,
)


EXAMPLES = [
    ("Ag_nanoparticle_colloid", EffectTarget(level="axis", axis="purine_nucleotide"),
     "1. Ag colloid × Purine axis — weighted-only path (no conflict)"),
    ("Ag_nanoparticle_colloid", EffectTarget(level="axis", axis="glycan_carbohydrate"),
     "2. Ag colloid × Glycan axis — weighted suppression (no conflict)"),
    ("Ag_nanoparticle_colloid", EffectTarget(level="axis", axis="nucleic_acid_backbone"),
     "3. Ag colloid × Nuc.Backbone axis — weighted + CONFLICTING surfaced"),
    ("Ag_nanoparticle_colloid", EffectTarget(level="band_family", cm1_range=(1020.0, 1080.0)),
     "4. Ag colloid × 1020–1080 cm⁻¹ band-family — pure conflict region"),
    ("Au_nanoparticle_colloid", EffectTarget(level="band_family", cm1_range=(1020.0, 1080.0)),
     "5. [v1.1.2] Au colloid × 1020–1080 cm⁻¹ band-family — Au-side conflict now surfaced"),
    ("Au_nanoparticle_colloid", EffectTarget(level="axis", axis="nucleic_acid_backbone"),
     "6. [v1.1.2] Au colloid × Nuc.Backbone axis — Au-side conflict on the axis target"),
    ("Au_nanoparticle_colloid", EffectTarget(level="axis", axis="purine_nucleotide"),
     "7. [v1.1.2] Au colloid × Purine axis — weighted-only (Au patch is multiplier-neutral here)"),
    ("unknown_SERS", EffectTarget(level="axis", axis="nucleic_acid_backbone"),
     "8. unknown_SERS × Nuc.Backbone axis — degrade-gracefully"),
]


def _hr(c: str = "─", n: int = 78) -> str:
    return c * n


def main():
    eng = load_engine()
    print("\n" + _hr())
    print(" GAIRA Substrate Engine v1.1.1 — conflict-aware demo")
    print(_hr())
    print(f"families loaded:           {len(eng.families)}")
    print(f"effect types:              {len(eng.effect_types)}")
    print(f"weighted (seed) effects:   {len(eng.weighted_registry)}")
    print(f"caution  (CSV)  effects:   {len(eng.caution_registry)}")
    print(f"merged registry total:     {len(eng.registry)}")

    for family, target, label in EXAMPLES:
        overlay = compose(
            family, target,
            registry=eng.registry, families=eng.families, effect_types=eng.effect_types,
        )
        print("\n" + "═" * 78)
        print(f" {label}")
        print("═" * 78)
        print(f"family = {family}")
        print(f"target = level:{target.level}, axis:{target.axis}, "
              f"window_id:{target.window_id}, cm1_range:{target.cm1_range}")
        print()
        print(f"  composed_confidence_multiplier = {overlay.composed_confidence_multiplier:.3f}")
        print(f"  caution                        = {overlay.caution}")
        print(f"  conflict_flag                  = {overlay.conflict_flag}")
        print(f"  unresolved_assignment_flag     = {overlay.unresolved_assignment_flag}")
        print(f"  observed_signal_visibility     = {overlay.observed_signal_visibility}")
        print(f"  biological_abundance_interp.   = {overlay.biological_abundance_interpretation}")
        print(f"  substrate_blind                = {overlay.substrate_blind}")
        print(f"  convergence_labels             = {list(overlay.convergence_labels)}")

        # weighted channel
        print(f"\n  weighted_effects (n={len(overlay.weighted_effects)}):")
        for e in overlay.weighted_effects:
            sources = (" — " + ", ".join(e.provenance_sources)) if e.provenance_sources else ""
            conv = f" [{e.convergence_status}]" if e.convergence_status else ""
            print(f"    · {e.effect_id} :: {e.effect_type}{conv} "
                  f"(×{e.confidence_multiplier:.2f}){sources}")

        # conflicting channel
        print(f"\n  conflicting_effects (n={len(overlay.conflicting_effects)}):")
        for e in overlay.conflicting_effects:
            sources = (" — " + ", ".join(e.provenance_sources)) if e.provenance_sources else ""
            print(f"    · {e.effect_id} :: {e.effect_type} [CONFLICTING] "
                  f"(weighting_applied={e.weighting_applied}){sources}")

        # insufficient channel
        print(f"\n  insufficient_effects (n={len(overlay.insufficient_effects)}):")
        for e in overlay.insufficient_effects:
            sources = (" — " + ", ".join(e.provenance_sources)) if e.provenance_sources else ""
            print(f"    · {e.effect_id} :: {e.effect_type} [INSUFFICIENT] "
                  f"(weighting_applied={e.weighting_applied}){sources}")

        # multiplier audit trail
        print(f"\n  weighted_multiplier_input_ids ({len(overlay.weighted_multiplier_input_ids)}):")
        for eid in overlay.weighted_multiplier_input_ids:
            print(f"    · {eid}")

        # conflict report
        rep = overlay.conflict_report
        if rep is not None and rep.has_conflict:
            print("\n  conflict_report:")
            print(f"    · conflicting_effect_ids: {list(rep.conflicting_effect_ids)}")
            print(f"    · candidate_assignment_classes: {list(rep.candidate_assignment_classes)}")
            print(f"    · spectral_regions: {list(rep.spectral_regions)}")

        # caveat lines
        print("\n  user_facing_caveat_lines:")
        for c in overlay.user_facing_caveat_lines:
            print(f"    » {c}")

        # markdown snippet
        print("\n  markdown_overlay_block:")
        print("  " + _hr("─", 74))
        for line in render_target_block(overlay).splitlines():
            print("  " + line)
        print("  " + _hr("─", 74))

    print("\n" + _hr())
    print(" demo complete")
    print(_hr())


if __name__ == "__main__":
    main()
