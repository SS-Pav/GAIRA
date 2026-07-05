"""GAIRA Substrate Engine v1.1.1 — public API (conflict-aware).

Typical use (single-source, weighted-only — pre-1.1.1 behaviour):

    from gaira.substrate import load_engine, compose, EffectTarget

    eng = load_engine()
    target = EffectTarget(level="axis", axis="purine_nucleotide")
    overlay = compose("Ag_nanoparticle_colloid", target,
                      registry=eng.registry, families=eng.families,
                      effect_types=eng.effect_types)

Dual-source (recommended for v1.1.1) — loads promoted seed for weighted
evidence AND full registry CSV for caution / conflict surfacing:

    eng = load_engine(with_full_registry=True)
    # eng.registry now contains promoted-seed entries (CONVERGED + EMERGING)
    # PLUS CSV entries with status CONFLICTING or INSUFFICIENT.

The engine is additive and read-only. It never mutates GAIRA BSV / ΔBSV
numerics. Multipliers are bounded in [0.4, 1.15] by policy; conflicting
and insufficient evidence are excluded from multiplier composition.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from gaira.substrate.schema import (
    AbundanceTag, ComposedOverlay, ConflictReport, EffectChannel, EffectTarget,
    ResolvedEffect, SubstrateBandEffect, SubstrateEffectType, SubstrateFamily,
    VisibilityTag,
)
from gaira.substrate.families import load_families
from gaira.substrate.effects import MULTIPLIER_MAX, MULTIPLIER_MIN, load_effect_types
from gaira.substrate.registry import (
    load_full_registry_csv, load_registry, merge_registries,
)
from gaira.substrate.resolver import compose, resolve
from gaira.substrate.report_overlay import (
    render_dataset_wide_caveats, render_declared_substrate_header,
    render_provenance_appendix, render_target_block,
)
from gaira.substrate.conflicts import (
    classify_channel, classify_effect, conflict_caveat_lines,
    insufficient_caveat_lines, make_conflict_report, split_by_channel,
)


# Build-artifact defaults. Override with GAIRA_BUILD env var if needed.
_DEFAULT_BUILD_ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD")


def _build_root() -> Path:
    return Path(os.environ.get("GAIRA_BUILD", _DEFAULT_BUILD_ROOT))


def default_paths() -> dict[str, Path]:
    root = _build_root()
    return {
        "families":     root / "substrate_physics_v1" / "config" / "substrate_families_v1.yaml",
        "effect_types": root / "substrate_physics_v1" / "config" / "substrate_effect_types_v1.yaml",
        "seed":         root / "substrate_physics_v1_expansion_pass2" / "config" / "substrate_band_effects_seed_v1_2.yaml",
        "full_registry_csv": root / "substrate_physics_v1_expansion_pass2" / "tables" / "substrate_physics_evidence_registry_v1_2.csv",
        # v1.1.2: Au-side conflict caution patch (additive, conflict-only)
        "caution_patch_csv": root / "substrate_physics_v1_1_2_au_conflict_patch" / "config" / "substrate_caution_patch_v1_1_2.csv",
        "query_schema": root / "substrate_physics_v1" / "config" / "substrate_query_metadata_schema_v1.yaml",
    }


@dataclass(frozen=True)
class SubstrateEngine:
    families: dict[str, SubstrateFamily]
    effect_types: dict[str, SubstrateEffectType]
    registry: dict[str, SubstrateBandEffect]
    paths: dict[str, Path]
    # v1.1.1: separable handles for inspection
    weighted_registry: dict[str, SubstrateBandEffect]
    caution_registry: dict[str, SubstrateBandEffect]


def load_engine(
    *,
    families_path: str | Path | None = None,
    effect_types_path: str | Path | None = None,
    seed_path: str | Path | None = None,
    full_registry_csv: str | Path | None = None,
    caution_patch_csv: str | Path | None = None,
    with_full_registry: bool = True,
    with_caution_patch: bool = True,
) -> SubstrateEngine:
    """Load + validate the substrate-physics YAMLs and (optionally) the full
    evidence registry CSV plus any caution-patch CSVs. Returns an engine handle.

    With `with_full_registry=True` (the v1.1.1 default), the engine merges:
      - promoted-seed YAML entries (CONVERGED + admitted EMERGING) into the
        weighted registry, AND
      - CONFLICTING / INSUFFICIENT entries from the full registry CSV into
        the caution registry.

    With `with_caution_patch=True` (the v1.1.2 default), the engine then merges
    additional CONFLICTING / INSUFFICIENT rows from the caution-patch CSV
    (e.g. `substrate_caution_patch_v1_1_2.csv` carrying the Au-side
    nucleic-related 1020-1080 conflict). Patch entries are additive — they
    cannot shadow existing entries (`prefer_base`).

    Set `with_full_registry=False` for the legacy weighted-only behaviour.
    """
    paths = default_paths()
    if families_path:
        paths["families"] = Path(families_path)
    if effect_types_path:
        paths["effect_types"] = Path(effect_types_path)
    if seed_path:
        paths["seed"] = Path(seed_path)
    if full_registry_csv:
        paths["full_registry_csv"] = Path(full_registry_csv)
    if caution_patch_csv:
        paths["caution_patch_csv"] = Path(caution_patch_csv)

    families = load_families(paths["families"])
    effect_types = load_effect_types(paths["effect_types"])
    weighted = load_registry(paths["seed"], families=families, effect_types=effect_types)

    caution: dict[str, SubstrateBandEffect] = {}
    if with_full_registry and Path(paths["full_registry_csv"]).exists():
        caution = load_full_registry_csv(
            paths["full_registry_csv"],
            families=families, effect_types=effect_types,
            statuses_to_load=("CONFLICTING", "INSUFFICIENT"),
        )

    # v1.1.2: caution-patch CSV (additive). Same loader; same status filter;
    # patch rows merged into the caution channel, never the weighted channel.
    if with_caution_patch and Path(paths["caution_patch_csv"]).exists():
        patch = load_full_registry_csv(
            paths["caution_patch_csv"],
            families=families, effect_types=effect_types,
            statuses_to_load=("CONFLICTING", "INSUFFICIENT"),
        )
        caution = merge_registries(caution, patch, on_conflict="prefer_base")

    merged = merge_registries(weighted, caution, on_conflict="prefer_base")
    return SubstrateEngine(
        families=families,
        effect_types=effect_types,
        registry=merged,
        paths=paths,
        weighted_registry=weighted,
        caution_registry=caution,
    )


__all__ = [
    # Schema types
    "AbundanceTag",
    "ComposedOverlay",
    "ConflictReport",
    "EffectChannel",
    "EffectTarget",
    "ResolvedEffect",
    "SubstrateBandEffect",
    "SubstrateEffectType",
    "SubstrateFamily",
    "VisibilityTag",
    # Constants
    "MULTIPLIER_MIN",
    "MULTIPLIER_MAX",
    # Loaders
    "SubstrateEngine",
    "default_paths",
    "load_effect_types",
    "load_engine",
    "load_families",
    "load_registry",
    "load_full_registry_csv",
    "merge_registries",
    # Resolver
    "compose",
    "resolve",
    # Conflict-channel helpers
    "classify_channel",
    "classify_effect",
    "conflict_caveat_lines",
    "insufficient_caveat_lines",
    "make_conflict_report",
    "split_by_channel",
    # Overlay
    "render_dataset_wide_caveats",
    "render_declared_substrate_header",
    "render_provenance_appendix",
    "render_target_block",
]
