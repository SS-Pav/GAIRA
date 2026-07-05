"""Registry loaders for gaira_base_2.

Loads:
  - motif registry v1.2 (from M4.1 output)
  - motif→axis mapping skeleton v1.1 (from pressure-test output)
  - M2.2 dual-status table (core_status × calibration_status)

Emits immutable dataclasses from schema.py. No scoring logic here.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from gaira.base2.schema import (
    AxisMapping,
    BandFamily,
    MotifDualStatus,
    MotifSpec,
)


# ──────────────────────────────────────────────────────────────────────
# Canonical paths to locked artefacts (absolute, for reproducibility)
# ──────────────────────────────────────────────────────────────────────

MOTIF_REGISTRY_V1_2 = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_build_motifs_v1/"
    "M4_1_refinement_and_recalibration_v1/registry/motif_candidate_registry_v1_2.yaml"
)
MAPPING_SKELETON_V1_1 = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_2_preimplementation_pressure_test_v1/"
    "tables/motif_to_axis_mapping_skeleton_v1_1.csv"
)
DUAL_STATUS_TABLE = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_build_motifs_v1/"
    "M2_2_ontology_untangling_v1/tables/motif_dual_status_v1.csv"
)


def _band_family_from_dict(d: dict) -> BandFamily:
    return BandFamily(
        family_id=d["family_id"],
        cm1_centre=float(d["cm1_centre"]),
        cm1_tolerance=float(d["cm1_tolerance"]),
        role=d.get("role", "primary"),
        vibrational_origin=d.get("vibrational_origin", ""),
    )


def load_motif_registry(path: Path = MOTIF_REGISTRY_V1_2) -> dict[str, MotifSpec]:
    """Load motif registry v1.2 and return a dict keyed by motif_id.

    Raises on any parsing failure (no silent data loss).
    """
    with path.open("r") as f:
        reg = yaml.safe_load(f)
    out: dict[str, MotifSpec] = {}
    for m in reg["motifs"]:
        primary = tuple(
            _band_family_from_dict(fd) for fd in (m.get("primary_band_families") or [])
        )
        supporting = tuple(
            _band_family_from_dict(fd) for fd in (m.get("supporting_band_families") or [])
        )
        exc = tuple(m.get("exclusion_conditions") or ())
        out[m["motif_id"]] = MotifSpec(
            motif_id=m["motif_id"],
            motif_family=m.get("motif_family", ""),
            motif_type=m.get("motif_type", ""),
            primary_bands=primary,
            supporting_bands=supporting,
            co_band_requirement=m.get("co_band_requirement_type", "SUPPORTING"),
            v1_active=bool(m.get("v1_active", True)),
            ambiguity_class=m.get("ambiguity_class", ""),
            exclusion_conditions=exc,
        )
    return out


def load_axis_mapping(path: Path = MAPPING_SKELETON_V1_1) -> dict[str, AxisMapping]:
    """Load mapping skeleton v1.1 and return a dict keyed by motif_id.

    Parses the revised axis column (comma-separated if CROSS_AXIS).
    Marks HELD_V2 / mapping_weight=0 motifs as active=False based on
    prior-pipeline rationale text ("HELD_V2" or "mapping_weight=0"
    anywhere in the rationale cell).
    """
    df = pd.read_csv(path)
    out: dict[str, AxisMapping] = {}
    for _, r in df.iterrows():
        mid = str(r["motif_id"])
        mtype = str(r["mapping_type"])
        raw_axis = str(r["revised_parent_axis_or_axes"])
        # Split by ", " (post-pandas-quote handling) — the CSV uses ", " separator
        axes = tuple(a.strip() for a in raw_axis.split(",") if a.strip())
        if not axes:
            continue
        primary_axis = axes[0]
        secondary_axes = tuple(axes[1:])
        rationale = str(r.get("rationale", ""))
        # HELD motifs: mapping_weight=0 per the rationale note
        held = ("HELD_V2" in rationale) or ("mapping_weight=0" in rationale)
        out[mid] = AxisMapping(
            motif_id=mid,
            primary_axis=primary_axis,
            secondary_axes=secondary_axes,
            mapping_type=mtype,
            active=not held,
        )
    return out


def load_dual_status(path: Path = DUAL_STATUS_TABLE) -> dict[str, MotifDualStatus]:
    """Load the M2.2 dual-status table keyed by motif_id."""
    df = pd.read_csv(path)
    out: dict[str, MotifDualStatus] = {}
    for _, r in df.iterrows():
        mid = str(r["motif_id"])
        out[mid] = MotifDualStatus(
            motif_id=mid,
            core_status=str(r["core_status"]),
            calibration_status=str(r.get("calibration_status", "NOT_RUN")),
            final_v1_role=str(r.get("final_v1_role", "HOLD_OUT")),
        )
    return out


def load_active_registry() -> tuple[
    dict[str, MotifSpec],
    dict[str, AxisMapping],
    dict[str, MotifDualStatus],
]:
    """One-shot loader: motif spec + mapping + dual-status.

    Returns only v1-active motifs (v1_active == True in the registry).
    HELD_V2 motifs remain in the raw registry but are filtered out here
    so that the scoring pipeline never sees them as contributors.
    """
    specs = load_motif_registry()
    mapping = load_axis_mapping()
    dual = load_dual_status()
    active_specs = {mid: s for mid, s in specs.items() if s.v1_active}
    return active_specs, mapping, dual
