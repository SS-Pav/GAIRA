"""Loader + validator for the substrate band-effects seed corpus.

Supports both the v1 seed (`substrate_band_effects_seed_v1.yaml`) and the
convergence-annotated v1.2 seed
(`substrate_band_effects_seed_v1_2.yaml`). Validates cross-references
against loaded families and effect-type vocabularies.

v1.1.1 patch:
  - CSV loader for the full evidence registry
    (`substrate_physics_evidence_registry_v1_2.csv`) so that CONFLICTING
    and INSUFFICIENT entries — deliberately excluded from the promoted
    seed — can be surfaced by the engine as caution-only channels.
  - `merge_registries()` to combine the promoted-seed (weighted) registry
    with caution-only registry entries from the CSV.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from gaira.substrate.schema import (
    EffectTarget, SubstrateBandEffect, SubstrateEffectType, SubstrateFamily,
)
from gaira.substrate.effects import MULTIPLIER_MIN, MULTIPLIER_MAX


# BSV axis validation is imported lazily to avoid circular deps at module load.
def _bsv_axes() -> set[str]:
    from gaira.spectral.window_panel import BSV_COMPONENTS
    return set(BSV_COMPONENTS) | {"all"}


_LEVEL_ALIASES = {
    # Upper-case (Pass-2 registry native form)
    "BAND_LEVEL":                "band",
    "BAND_FAMILY_LEVEL":         "band_family",
    "MOTIF_LEVEL":               "motif",
    "AXIS_LEVEL":                "axis",
    "GLOBAL_SUBSTRATE_BEHAVIOR": "global",
    # Lower-case (v1.2 seed YAML form after CSV → YAML normalisation)
    "band_level":                "band",
    "band_family_level":         "band_family",
    "motif_level":               "motif",
    "axis_level":                "axis",
    "global_substrate_behavior": "global",
    # Already normalised forms
    "band": "band", "band_family": "band_family",
    "motif": "motif", "axis": "axis", "global": "global",
}


# Map registry CSV biochemical_target_class → BSV axis name
# (used by the CSV loader; the YAML seed sets axis explicitly)
_TARGET_CLASS_TO_BSV_AXIS = {
    "purine_nucleobase":        "purine_nucleotide",
    "pyrimidine":               "pyrimidine_nucleotide",
    "pyrimidine_nucleobase":    "pyrimidine_nucleotide",
    "nucleic_acid_backbone":    "nucleic_acid_backbone",
    "protein_peptide_backbone": "protein_backbone",
    "aromatic_amino_acid":      "aromatic_amino_acid",
    "glycan_carbohydrate":      "glycan_carbohydrate",
    "lipid_membrane":           "membrane_lipid",
    "membrane_lipid":           "membrane_lipid",
    "sulfur_thiol_redox":       "redox_metabolite",
    "redox_metabolite":         "redox_metabolite",
    "uric_acid_HX_overlap":     "purine_nucleotide",
    "citrate_baseline":         None,
    "global":                   "all",
}


def _parse_target(raw: dict) -> EffectTarget:
    raw = raw or {}
    level_in = str(raw.get("level", "axis"))
    level = _LEVEL_ALIASES.get(level_in, level_in.lower())
    if level not in {"band", "band_family", "motif", "axis", "global"}:
        raise ValueError(f"Unknown target.level: {level_in}")
    cm = raw.get("cm1_range")
    if cm is not None:
        if not (isinstance(cm, (list, tuple)) and len(cm) == 2):
            raise ValueError(f"cm1_range must be a 2-element list: got {cm!r}")
        cm = (float(cm[0]), float(cm[1]))
    return EffectTarget(
        level=level,  # type: ignore[arg-type]
        axis=(str(raw["axis"]) if "axis" in raw and raw["axis"] is not None else None),
        window_id=(str(raw["window_id"]) if raw.get("window_id") else None),
        cm1_range=cm,
        band_family=(str(raw["band_family"]) if raw.get("band_family") else None),
    )


def _pick_multiplier(e: dict, effect_types: dict[str, SubstrateEffectType]) -> float:
    if "confidence_multiplier" in e and e["confidence_multiplier"] is not None:
        return float(e["confidence_multiplier"])
    et = effect_types.get(str(e["effect_type"]))
    if et is None:
        raise ValueError(
            f"Effect {e.get('id')}: effect_type '{e.get('effect_type')}' not in loaded effect-type vocabulary"
        )
    return et.confidence_multiplier


def _coerce_provenance(e: dict) -> tuple[tuple[str, ...], str]:
    prov = e.get("provenance") or {}
    # v1 shape: primary_sources: [...]
    # v1.1 shape: source_identifier: "PMID:..."
    # v1.2 shape: key_sources: [...] OR primary_sources
    sources: list[str] = []
    for key in ("key_sources", "primary_sources"):
        if key in prov and prov[key]:
            sources.extend([str(s) for s in prov[key] if s])
    if "source_identifier" in prov and prov["source_identifier"]:
        sources.append(str(prov["source_identifier"]))
    # evidence_types can also carry provenance hints in v1.2
    notes = str(prov.get("provenance_notes") or prov.get("citations_note") or "")
    return tuple(sources), notes


def load_registry(
    yaml_path: str | Path,
    families: dict[str, SubstrateFamily],
    effect_types: dict[str, SubstrateEffectType],
) -> dict[str, SubstrateBandEffect]:
    """Load + validate a seed-corpus YAML into effect_id → SubstrateBandEffect dict."""
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Seed YAML not found: {path}")
    doc = yaml.safe_load(path.read_text())
    if not doc or "effects" not in doc:
        raise ValueError(f"{path}: malformed YAML; expected top-level 'effects' list")

    axes = _bsv_axes()
    out: dict[str, SubstrateBandEffect] = {}
    for e in doc["effects"]:
        eid = str(e["id"])
        if eid in out:
            raise ValueError(f"Duplicate effect id: {eid}")

        family = str(e["substrate_family"])
        if family not in families:
            raise ValueError(f"Effect {eid}: substrate_family '{family}' not in families vocabulary")

        et_id = str(e["effect_type"])
        if et_id not in effect_types:
            raise ValueError(f"Effect {eid}: effect_type '{et_id}' not in effect-type vocabulary")

        target = _parse_target(e.get("target") or {})

        # Axis validation for axis- and global-level targets.
        if target.level in ("axis", "global"):
            axis = target.axis or "all"
            if axis not in axes:
                raise ValueError(
                    f"Effect {eid}: axis '{axis}' is not a valid BSV component "
                    f"(valid: {sorted(axes)})"
                )

        m = _pick_multiplier(e, effect_types)
        if not (MULTIPLIER_MIN <= m <= MULTIPLIER_MAX):
            raise ValueError(
                f"Effect {eid}: multiplier {m} outside [{MULTIPLIER_MIN}, {MULTIPLIER_MAX}]"
            )

        sources, prov_notes = _coerce_provenance(e)
        evidence_types = e.get("evidence_types") or []
        if isinstance(evidence_types, str):
            evidence_types = [evidence_types]

        out[eid] = SubstrateBandEffect(
            id=eid,
            substrate_family=family,
            target=target,
            effect_type=et_id,
            evidence_confidence=str(e.get("evidence_confidence", "moderate")),  # type: ignore[arg-type]
            convergence_status=(str(e["convergence_status"]) if e.get("convergence_status") else None),  # type: ignore[arg-type]
            biochemical_target_class=(str(e["biochemical_target_class"]) if e.get("biochemical_target_class") else None),
            spectral_region_description=(str(e["spectral_region_description"]) if e.get("spectral_region_description") else None),
            evidence_types=tuple(evidence_types),
            confidence_multiplier=m,
            provenance_sources=sources,
            provenance_notes=prov_notes,
            summary_of_effect=str(e.get("summary_of_effect", "")).strip(),
            notes=str(e.get("notes", "")).strip(),
        )
    return out


# ──────────────────────────────────────────────────────────────────────
# v1.1.1: Full registry CSV loader (caution / conflict surfacing)
# ──────────────────────────────────────────────────────────────────────

def _parse_cm1_range_csv(raw: str) -> tuple[float, float] | None:
    """Parse 'spectral_range_cm1' from registry CSV. Forms seen:
       '[715, 740]'  '[1020,1080]'  ''  None
    """
    if raw is None or str(raw).strip() == "":
        return None
    s = str(raw).strip()
    try:
        v = json.loads(s)
        if isinstance(v, list) and len(v) == 2:
            return (float(v[0]), float(v[1]))
    except Exception:
        pass
    # Fallback: strip brackets and split on comma
    s2 = s.strip("[]() ")
    parts = [p.strip() for p in s2.split(",") if p.strip()]
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    return None


def load_full_registry_csv(
    csv_path: str | Path,
    families: dict[str, SubstrateFamily],
    effect_types: dict[str, SubstrateEffectType],
    *,
    statuses_to_load: tuple[str, ...] = ("CONFLICTING", "INSUFFICIENT"),
) -> dict[str, SubstrateBandEffect]:
    """Load the full evidence registry CSV.

    By default loads only CONFLICTING + INSUFFICIENT entries (the ones the
    promoted seed YAML deliberately excludes). Pass `statuses_to_load=()`
    to load every row regardless of convergence_status.

    Returns an effect_id → SubstrateBandEffect dict. The dict can be merged
    with the seed registry via `merge_registries()`.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Full registry CSV not found: {path}")

    axes = _bsv_axes()
    out: dict[str, SubstrateBandEffect] = {}

    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = (row.get("convergence_status") or "").strip()
            if statuses_to_load and status not in statuses_to_load:
                continue

            eid = (row.get("effect_id") or "").strip()
            if not eid:
                continue
            if eid in out:
                raise ValueError(f"Duplicate effect id in CSV: {eid}")

            family = (row.get("substrate_family") or "").strip()
            if family not in families:
                # Skip rather than crash — registry can carry future families
                continue

            et_id = (row.get("effect_type") or "").strip()
            if et_id not in effect_types:
                # Skip rather than crash — registry can carry future effect types
                continue

            level_raw = (row.get("target_level") or "axis").strip()
            level = _LEVEL_ALIASES.get(level_raw, level_raw.lower())
            if level not in {"band", "band_family", "motif", "axis", "global"}:
                raise ValueError(f"CSV row {eid}: unknown target_level {level_raw!r}")

            cm = _parse_cm1_range_csv(row.get("spectral_range_cm1") or "")

            # Derive axis from biochemical_target_class for axis/global/band-family entries
            btc = (row.get("biochemical_target_class") or "").strip()
            axis: str | None
            if level == "global":
                axis = "all"
            else:
                mapped = _TARGET_CLASS_TO_BSV_AXIS.get(btc)
                axis = mapped if mapped else None

            if level in ("axis", "global"):
                axis_eff = axis or "all"
                if axis_eff not in axes:
                    raise ValueError(
                        f"CSV row {eid}: derived axis '{axis_eff}' (from biochemical_target_class "
                        f"'{btc}') is not a valid BSV component"
                    )

            target = EffectTarget(
                level=level,  # type: ignore[arg-type]
                axis=axis,
                window_id=None,
                cm1_range=cm,
                band_family=(row.get("spectral_region_description") or None),
            )

            # Multiplier: CSV does NOT carry per-effect multipliers; we use the
            # effect-type default. This is harmless for caution channels because
            # CONFLICTING / INSUFFICIENT are excluded from the multiplier
            # composition by the v1.1.1 resolver invariant.
            m = effect_types[et_id].confidence_multiplier
            if not (MULTIPLIER_MIN <= m <= MULTIPLIER_MAX):
                raise ValueError(
                    f"CSV row {eid}: multiplier {m} outside [{MULTIPLIER_MIN}, {MULTIPLIER_MAX}]"
                )

            sources_raw = (row.get("key_sources") or "").strip()
            sources = tuple(s.strip() for s in sources_raw.split(";") if s.strip()) if sources_raw else ()
            evidence_raw = (row.get("evidence_types") or "").strip()
            evidence_types_t = tuple(s.strip() for s in evidence_raw.split(";") if s.strip()) if evidence_raw else ()

            out[eid] = SubstrateBandEffect(
                id=eid,
                substrate_family=family,
                target=target,
                effect_type=et_id,
                evidence_confidence=str(row.get("evidence_confidence") or "moderate"),  # type: ignore[arg-type]
                convergence_status=status or None,  # type: ignore[arg-type]
                biochemical_target_class=btc or None,
                spectral_region_description=(row.get("spectral_region_description") or None),
                evidence_types=evidence_types_t,
                confidence_multiplier=m,
                provenance_sources=sources,
                provenance_notes=str(row.get("provenance_notes") or "").strip(),
                summary_of_effect=str(row.get("summary_of_effect") or "").strip(),
                notes="",
            )
    return out


def merge_registries(
    base: dict[str, SubstrateBandEffect],
    overlay: dict[str, SubstrateBandEffect],
    *,
    on_conflict: str = "prefer_base",
) -> dict[str, SubstrateBandEffect]:
    """Merge two registries.

    `on_conflict`:
      - "prefer_base"   : if same id appears in both, keep the base entry
                          (intended for promoted-seed-wins semantics).
      - "prefer_overlay": if same id appears in both, keep the overlay entry.
      - "raise"         : raise on duplicate.

    Returns a new dict; inputs are not mutated.
    """
    out = dict(base)
    for eid, eff in overlay.items():
        if eid in out:
            if on_conflict == "prefer_base":
                continue
            if on_conflict == "prefer_overlay":
                out[eid] = eff
                continue
            raise ValueError(f"Duplicate effect id during merge: {eid}")
        out[eid] = eff
    return out
