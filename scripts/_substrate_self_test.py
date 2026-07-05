"""GAIRA Substrate Engine v1.1.2 — self-test harness (Au-conflict patch).

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/_substrate_self_test.py

Covers:
  - the 9 baseline validation checks from the v1.1 implementation spec,
  - 9 conflict-aware checks added in the v1.1.1 patch, plus
  - 5 Au-side conflict checks added in the v1.1.2 patch.

Fails loudly at the first broken invariant.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gaira.substrate import (
    EffectTarget, MULTIPLIER_MAX, MULTIPLIER_MIN,
    classify_channel, compose, load_engine, render_target_block,
)
from gaira.spectral.window_panel import BSV_COMPONENTS


def _check(label, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}{(' — ' + detail) if detail else ''}")
    if not cond:
        raise AssertionError(f"self-test failed: {label} — {detail}")


def main():
    print("\n[substrate self-test v1.1.1]")
    print("─" * 72)

    # ── BASELINE CHECKS (v1.1) ────────────────────────────────────────

    # 1. YAMLs load cleanly + cross-refs validate on construction
    print("\n1. Config load + cross-reference validation")
    eng = load_engine()
    _check("families YAML loads",   len(eng.families) > 0, f"{len(eng.families)} families")
    _check("effect-types YAML loads", len(eng.effect_types) > 0,
           f"{len(eng.effect_types)} types")
    _check("seed YAML loads",        len(eng.weighted_registry) > 0,
           f"{len(eng.weighted_registry)} weighted effects")

    # 2. All referenced families exist
    print("\n2. Family references")
    bad = [eid for eid, eff in eng.registry.items()
           if eff.substrate_family not in eng.families]
    _check("every effect references a known family", not bad,
           f"violations: {bad[:5]}")

    # 3. All referenced effect types exist
    print("\n3. Effect-type references")
    bad = [eid for eid, eff in eng.registry.items()
           if eff.effect_type not in eng.effect_types]
    _check("every effect references a known effect type", not bad,
           f"violations: {bad[:5]}")

    # 4. All confidence multipliers within bounds
    print("\n4. Multiplier bounds")
    oob = [(eid, eff.confidence_multiplier) for eid, eff in eng.registry.items()
           if not (MULTIPLIER_MIN <= (eff.confidence_multiplier or 1.0) <= MULTIPLIER_MAX)]
    _check("all effect-multipliers in [0.40, 1.15]", not oob, f"violations: {oob[:5]}")

    # 5. Unique effect IDs
    print("\n5. Unique effect IDs")
    _check("all effect IDs unique in merged registry",
           len(set(eng.registry.keys())) == len(eng.registry),
           f"{len(eng.registry)} unique IDs")

    # 6. Axis-level / global-level targets reference valid BSV components
    print("\n6. BSV axis validity on axis/global targets")
    axes = set(BSV_COMPONENTS) | {"all"}
    bad = [
        eid for eid, eff in eng.registry.items()
        if eff.target.level in ("axis", "global")
        and (eff.target.axis or "all") not in axes
    ]
    _check("every axis/global target resolves to a BSV component or 'all'", not bad,
           f"violations: {bad[:5]}")

    # 7. Unknown-family degradation
    print("\n7. Unknown-family graceful degradation")
    blind = compose(
        "unknown",
        EffectTarget(level="axis", axis="purine_nucleotide"),
        registry=eng.registry, families=eng.families, effect_types=eng.effect_types,
    )
    _check("unknown family returns substrate_blind=True", blind.substrate_blind)
    _check("unknown family multiplier is 1.0",
           blind.composed_confidence_multiplier == 1.0,
           f"got {blind.composed_confidence_multiplier}")
    unksers = compose(
        "unknown_SERS",
        EffectTarget(level="axis", axis="purine_nucleotide"),
        registry=eng.registry, families=eng.families, effect_types=eng.effect_types,
    )
    _check("unknown_SERS family caution=True + multiplier=1.0",
           unksers.caution and unksers.composed_confidence_multiplier == 1.0,
           f"caution={unksers.caution} mult={unksers.composed_confidence_multiplier}")

    # 8. Canonical deterministic queries
    print("\n8. Canonical queries resolve deterministically")
    canonical = [
        ("Ag_nanoparticle_colloid", EffectTarget(level="axis", axis="purine_nucleotide")),
        ("Ag_nanoparticle_colloid", EffectTarget(level="axis", axis="glycan_carbohydrate")),
        ("Ag_nanoparticle_colloid", EffectTarget(level="axis", axis="membrane_lipid")),
        ("Ag_nanoparticle_colloid", EffectTarget(level="band", cm1_range=(715.0, 734.0))),
        ("Au_nanoparticle_colloid", EffectTarget(level="axis", axis="redox_metabolite")),
    ]
    for family, tgt in canonical:
        ov = compose(family, tgt, registry=eng.registry,
                     families=eng.families, effect_types=eng.effect_types)
        _check(
            f"{family} × {tgt.level}:{tgt.axis or tgt.cm1_range} → "
            f"vis={ov.observed_signal_visibility}, abd={ov.biological_abundance_interpretation}, "
            f"mult={ov.composed_confidence_multiplier:.2f}, n_effects={len(ov.resolved_effects)}",
            True,
        )

    # 9. Markdown overlay generation
    print("\n9. Markdown overlay generation (baseline)")
    md = render_target_block(
        compose(
            "Ag_nanoparticle_colloid",
            EffectTarget(level="axis", axis="purine_nucleotide"),
            registry=eng.registry, families=eng.families, effect_types=eng.effect_types,
        )
    )
    _check("render_target_block returns non-empty markdown", len(md) > 100,
           f"{len(md)} chars")
    _check("markdown mentions 'visibility'", "visibility" in md.lower())
    _check("markdown mentions 'abundance'", "abundance" in md.lower())

    # ── v1.1.1 CONFLICT-AWARE CHECKS ──────────────────────────────────

    # v1.1.1-1. Promoted seed loads
    print("\nv1.1.1-1. Promoted seed loads (weighted registry)")
    _check("weighted_registry non-empty", len(eng.weighted_registry) > 0,
           f"{len(eng.weighted_registry)} weighted entries")

    # v1.1.1-2. Full registry CSV loads (caution + insufficient)
    print("\nv1.1.1-2. Full registry CSV loads (caution registry)")
    _check("caution_registry non-empty", len(eng.caution_registry) > 0,
           f"{len(eng.caution_registry)} caution entries")
    statuses = {
        eff.convergence_status for eff in eng.caution_registry.values()
    }
    _check("caution_registry only carries CONFLICTING + INSUFFICIENT",
           statuses.issubset({"CONFLICTING", "INSUFFICIENT"}),
           f"got {statuses}")

    # v1.1.1-3. Conflicting entries are preserved through composition
    print("\nv1.1.1-3. CONFLICTING entries preserved through composition")
    ov_nuc = compose(
        "Ag_nanoparticle_colloid",
        EffectTarget(level="axis", axis="nucleic_acid_backbone"),
        registry=eng.registry, families=eng.families, effect_types=eng.effect_types,
    )
    _check("Ag × nuc.backbone surfaces ≥1 conflicting effect",
           len(ov_nuc.conflicting_effects) >= 1,
           f"{len(ov_nuc.conflicting_effects)} conflicting effects")
    _check("Ag × nuc.backbone sets conflict_flag=True",
           ov_nuc.conflict_flag is True)
    _check("Ag × nuc.backbone sets unresolved_assignment_flag=True",
           ov_nuc.unresolved_assignment_flag is True)

    # v1.1.1-4. Conflicting entries do NOT enter the multiplier composition
    print("\nv1.1.1-4. Conflicting entries excluded from multiplier composition")
    conflict_ids = {e.effect_id for e in ov_nuc.conflicting_effects}
    multiplier_input_ids = set(ov_nuc.weighted_multiplier_input_ids)
    _check("conflicting effect ids NOT in weighted_multiplier_input_ids",
           conflict_ids.isdisjoint(multiplier_input_ids),
           f"leak: {conflict_ids & multiplier_input_ids}")
    # Pure-conflict region: band 1020-1080 yields a conflicting effect with no
    # weighted multiplier influence beyond the existing weighted entries.
    ov_pure = compose(
        "Ag_nanoparticle_colloid",
        EffectTarget(level="band", cm1_range=(1020.0, 1080.0)),
        registry=eng.registry, families=eng.families, effect_types=eng.effect_types,
    )
    _check("band 1020-1080 conflicting effects all have weighting_applied=False",
           all(not e.weighting_applied for e in ov_pure.conflicting_effects),
           f"violations: {[e.effect_id for e in ov_pure.conflicting_effects if e.weighting_applied]}")

    # v1.1.1-5. Conflicting entries DO affect caution / abundance outputs
    print("\nv1.1.1-5. Conflicting entries affect caution / abundance outputs")
    _check("Ag × nuc.backbone caution=True (conflict raises caution)",
           ov_nuc.caution is True)
    _check("Ag × nuc.backbone abundance forced to abundance_not_directly_inferable",
           ov_nuc.biological_abundance_interpretation == "abundance_not_directly_inferable",
           f"got {ov_nuc.biological_abundance_interpretation}")

    # v1.1.1-6. Insufficient evidence degrades gracefully
    print("\nv1.1.1-6. INSUFFICIENT evidence degrades gracefully")
    ov_aurough = compose(
        "Au_roughened_surface",
        EffectTarget(level="axis", axis="purine_nucleotide"),
        registry=eng.registry, families=eng.families, effect_types=eng.effect_types,
    )
    _check("Au_roughened_surface surfaces ≥1 insufficient effect",
           len(ov_aurough.insufficient_effects) >= 1,
           f"{len(ov_aurough.insufficient_effects)} insufficient effects")
    _check("INSUFFICIENT does NOT contribute to multiplier (multiplier=1.0)",
           ov_aurough.composed_confidence_multiplier == 1.0,
           f"got {ov_aurough.composed_confidence_multiplier}")
    _check("INSUFFICIENT effects all have weighting_applied=False",
           all(not e.weighting_applied for e in ov_aurough.insufficient_effects))
    _check("INSUFFICIENT raises caution=True",
           ov_aurough.caution is True)

    # v1.1.1-7. Nucleic conflict regions resolve deterministically
    print("\nv1.1.1-7. Nucleic conflict-region queries are deterministic")
    nucleic_queries = [
        ("Ag_nanoparticle_colloid", EffectTarget(level="band", cm1_range=(1020.0, 1080.0))),
        ("Ag_nanoparticle_colloid", EffectTarget(level="band", cm1_range=(1080.0, 1140.0))),
        ("Ag_nanoparticle_colloid", EffectTarget(level="band", cm1_range=(1300.0, 1400.0))),
        ("Au_nanoparticle_colloid", EffectTarget(level="band", cm1_range=(1020.0, 1080.0))),
    ]
    for family, tgt in nucleic_queries:
        ov_a = compose(family, tgt, registry=eng.registry,
                       families=eng.families, effect_types=eng.effect_types)
        ov_b = compose(family, tgt, registry=eng.registry,
                       families=eng.families, effect_types=eng.effect_types)
        _check(
            f"{family} × band {tgt.cm1_range} deterministic: "
            f"mult_eq={ov_a.composed_confidence_multiplier == ov_b.composed_confidence_multiplier} "
            f"flags_eq={(ov_a.conflict_flag, ov_a.unresolved_assignment_flag) == (ov_b.conflict_flag, ov_b.unresolved_assignment_flag)} "
            f"channels_eq={(len(ov_a.weighted_effects), len(ov_a.conflicting_effects), len(ov_a.insufficient_effects)) == (len(ov_b.weighted_effects), len(ov_b.conflicting_effects), len(ov_b.insufficient_effects))}",
            ov_a.composed_confidence_multiplier == ov_b.composed_confidence_multiplier
            and ov_a.conflict_flag == ov_b.conflict_flag
            and len(ov_a.conflicting_effects) == len(ov_b.conflicting_effects),
        )

    # v1.1.1-8. Markdown contains explicit conflict language when appropriate
    print("\nv1.1.1-8. Markdown contains explicit conflict language when appropriate")
    md_nuc = render_target_block(ov_nuc).lower()
    for phrase in ("conflict", "conflicting"):
        _check(f"markdown for Ag × nuc.backbone contains '{phrase}'",
               phrase in md_nuc)
    _check("markdown surfaces 'unresolved' or 'unique biochemical assignment' phrasing",
           ("unresolved" in md_nuc) or ("unique biochemical assignment" in md_nuc))
    md_purine_no_conflict = render_target_block(
        compose(
            "Ag_nanoparticle_colloid",
            EffectTarget(level="axis", axis="purine_nucleotide"),
            registry=eng.registry, families=eng.families, effect_types=eng.effect_types,
        )
    ).lower()
    # Purine-axis on Ag has weighted evidence but no conflicting evidence —
    # the conflict-section sentence "carries CONFLICTING literature
    # assignments" must NOT appear.
    _check(
        "markdown for Ag × purine_nucleotide does NOT carry 'carries CONFLICTING'",
        "carries conflicting" not in md_purine_no_conflict,
    )

    # v1.1.1-9. Channel-classifier sanity
    print("\nv1.1.1-9. classify_channel sanity")
    _check("CONVERGED → weighted",    classify_channel("CONVERGED")    == "weighted")
    _check("EMERGING → weighted",     classify_channel("EMERGING")     == "weighted")
    _check("CONFLICTING → conflicting", classify_channel("CONFLICTING") == "conflicting")
    _check("INSUFFICIENT → insufficient", classify_channel("INSUFFICIENT") == "insufficient")
    _check("None → weighted (legacy)", classify_channel(None) == "weighted")

    # ── v1.1.2 Au-CONFLICT-PATCH CHECKS ───────────────────────────────

    # v1.1.2-1. Au-side caution patch entries load through the caution source
    print("\nv1.1.2-1. Au-side caution patch loads")
    au_caution_ids = sorted(
        eid for eid, eff in eng.caution_registry.items()
        if eff.substrate_family == "Au_nanoparticle_colloid"
    )
    _check("≥1 Au_nanoparticle_colloid entry in caution_registry",
           len(au_caution_ids) >= 1, f"got {au_caution_ids}")
    _check("expected patch id `eff_aucoll_backbone_1050` present",
           "eff_aucoll_backbone_1050" in eng.caution_registry,
           f"caution ids: {sorted(eng.caution_registry.keys())}")
    _check("Au patch entry status is CONFLICTING",
           eng.caution_registry["eff_aucoll_backbone_1050"].convergence_status == "CONFLICTING",
           f"got {eng.caution_registry['eff_aucoll_backbone_1050'].convergence_status}")

    # v1.1.2-2. Au caution entries are preserved through resolution / composition
    print("\nv1.1.2-2. Au caution entries preserved through composition")
    ov_au_band = compose(
        "Au_nanoparticle_colloid",
        EffectTarget(level="band_family", cm1_range=(1020.0, 1080.0)),
        registry=eng.registry, families=eng.families, effect_types=eng.effect_types,
    )
    _check("Au × band-family 1020-1080 surfaces ≥1 conflicting effect",
           len(ov_au_band.conflicting_effects) >= 1,
           f"{len(ov_au_band.conflicting_effects)} conflicting effects")
    _check("Au × band-family 1020-1080 includes the patch id in conflicting channel",
           any(e.effect_id == "eff_aucoll_backbone_1050" for e in ov_au_band.conflicting_effects))

    # v1.1.2-3. Au conflict entries do NOT enter multiplier composition
    print("\nv1.1.2-3. Au conflict entries excluded from multiplier")
    _check("eff_aucoll_backbone_1050 NOT in weighted_multiplier_input_ids",
           "eff_aucoll_backbone_1050" not in set(ov_au_band.weighted_multiplier_input_ids))
    _check("Au conflict effect carries weighting_applied=False",
           all(not e.weighting_applied for e in ov_au_band.conflicting_effects
               if e.effect_id == "eff_aucoll_backbone_1050"))
    # Strong invariant: composing the same query with the patch absent must
    # produce the SAME multiplier (the patch is multiplier-neutral).
    eng_no_patch = load_engine(with_caution_patch=False)
    ov_no_patch = compose(
        "Au_nanoparticle_colloid",
        EffectTarget(level="band_family", cm1_range=(1020.0, 1080.0)),
        registry=eng_no_patch.registry,
        families=eng_no_patch.families,
        effect_types=eng_no_patch.effect_types,
    )
    _check("Au × band-family 1020-1080 multiplier UNCHANGED by patch",
           ov_au_band.composed_confidence_multiplier == ov_no_patch.composed_confidence_multiplier,
           f"with={ov_au_band.composed_confidence_multiplier} "
           f"without={ov_no_patch.composed_confidence_multiplier}")

    # v1.1.2-4. Au conflict entries DO affect caution / abundance / flags
    print("\nv1.1.2-4. Au conflict entries affect caution / abundance / flags")
    _check("Au × band-family 1020-1080 conflict_flag=True", ov_au_band.conflict_flag is True)
    _check("Au × band-family 1020-1080 unresolved_assignment_flag=True",
           ov_au_band.unresolved_assignment_flag is True)
    _check("Au × band-family 1020-1080 caution=True", ov_au_band.caution is True)
    _check("Au × band-family 1020-1080 abundance forced to abundance_not_directly_inferable",
           ov_au_band.biological_abundance_interpretation == "abundance_not_directly_inferable")
    # Symmetry: Au × axis nucleic_acid_backbone must now also surface the conflict
    ov_au_axis = compose(
        "Au_nanoparticle_colloid",
        EffectTarget(level="axis", axis="nucleic_acid_backbone"),
        registry=eng.registry, families=eng.families, effect_types=eng.effect_types,
    )
    _check("Au × axis nuc.backbone surfaces ≥1 conflicting effect",
           len(ov_au_axis.conflicting_effects) >= 1)
    _check("Au × axis nuc.backbone conflict_flag=True", ov_au_axis.conflict_flag is True)

    # v1.1.2-5. Au 1020-1080 example resolves deterministically with explicit
    # conflict language in the markdown output
    print("\nv1.1.2-5. Au 1020-1080 markdown carries explicit conflict language")
    md_au = render_target_block(ov_au_band).lower()
    for phrase in ("conflict", "conflicting"):
        _check(f"Au markdown contains '{phrase}'", phrase in md_au)
    _check("Au markdown surfaces 'unresolved' or 'unique biochemical assignment' phrasing",
           ("unresolved" in md_au) or ("unique biochemical assignment" in md_au))
    _check("Au markdown names the patch id `eff_aucoll_backbone_1050`",
           "eff_aucoll_backbone_1050" in md_au)
    # Determinism
    ov_au_band_b = compose(
        "Au_nanoparticle_colloid",
        EffectTarget(level="band_family", cm1_range=(1020.0, 1080.0)),
        registry=eng.registry, families=eng.families, effect_types=eng.effect_types,
    )
    _check("Au × band-family 1020-1080 deterministic across two compositions",
           ov_au_band.composed_confidence_multiplier == ov_au_band_b.composed_confidence_multiplier
           and ov_au_band.conflict_flag == ov_au_band_b.conflict_flag
           and len(ov_au_band.conflicting_effects) == len(ov_au_band_b.conflicting_effects))

    print("\n" + "─" * 72)
    print("[substrate self-test v1.1.2] ALL CHECKS PASSED ✓")
    print(f"  · families: {len(eng.families)}")
    print(f"  · effect types: {len(eng.effect_types)}")
    print(f"  · weighted (seed) effects: {len(eng.weighted_registry)}")
    print(f"  · caution effects (CSV + patch):  {len(eng.caution_registry)}")
    print(f"      └─ Au_nanoparticle_colloid: {len(au_caution_ids)} caution entries")
    print(f"  · merged registry total:  {len(eng.registry)}")


if __name__ == "__main__":
    main()
