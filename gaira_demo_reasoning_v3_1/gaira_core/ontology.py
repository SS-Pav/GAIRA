"""GAIRA Demo v3 — biochemical ontology loader.

Loads the versioned, disease-label-independent ontology spec
(`config/biochemical_ontology_v1.yaml`). The ontology defines axis MEANING
(names, motifs, MSS analytes, collisions, substrate sensitivity, grounding
status). It does NOT define global-coordinate scale — that is a separate
frozen calibration (see global_coordinates.py).

Read-only. Never mutated at runtime.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config as cfg

ONTOLOGY_DIR = cfg.DEMO_ROOT / "config"
ONTOLOGY_YAML = ONTOLOGY_DIR / "biochemical_ontology_v1.yaml"
ONTOLOGY_JSON = ONTOLOGY_DIR / "biochemical_ontology_v1.json"

GROUNDING_STATUS_ORDER = (
    "independently_grounded",
    "partially_grounded",
    "derived_split",
    "insufficiently_grounded",
)


@dataclass(frozen=True)
class Axis:
    id: str
    display_name: str
    scientific_name: str
    interpretation: str
    legacy_source_axis: str
    from_legacy_split: bool
    split_siblings: tuple[str, ...]
    contributing_motifs: tuple[str, ...]
    contributing_mss_analytes: tuple[str, ...]
    known_collisions: tuple[str, ...]
    substrate_sensitivities: tuple[str, ...]
    evidence_confidence: str
    validation_status: str
    grounding_status: str
    limitations: str


@dataclass(frozen=True)
class Ontology:
    version: str
    name: str
    n_axes: int
    disease_label_independent: bool
    legacy_source: str
    notes: str
    legacy8_to_v11: dict
    axes: tuple[Axis, ...]
    _raw: dict = field(default_factory=dict, repr=False)

    def axis(self, axis_id: str) -> Axis | None:
        for a in self.axes:
            if a.id == axis_id:
                return a
        return None

    @property
    def axis_ids(self) -> tuple[str, ...]:
        return tuple(a.id for a in self.axes)

    def status_of(self, axis_id: str) -> str:
        a = self.axis(axis_id)
        return a.grounding_status if a else "unknown"


def _load_spec() -> dict[str, Any]:
    if ONTOLOGY_YAML.exists():
        try:
            import yaml  # noqa
            with open(ONTOLOGY_YAML) as f:
                return yaml.safe_load(f)
        except Exception:
            pass
    if ONTOLOGY_JSON.exists():
        with open(ONTOLOGY_JSON) as f:
            return json.load(f)
    raise FileNotFoundError(
        f"No ontology spec found at {ONTOLOGY_YAML} or {ONTOLOGY_JSON}")


def load_ontology() -> Ontology:
    spec = _load_spec()
    axes = tuple(
        Axis(
            id=a["id"],
            display_name=a.get("display_name", a["id"]),
            scientific_name=a.get("scientific_name", ""),
            interpretation=a.get("interpretation", ""),
            legacy_source_axis=a.get("legacy_source_axis", ""),
            from_legacy_split=bool(a.get("from_legacy_split", False)),
            split_siblings=tuple(a.get("split_siblings", []) or []),
            contributing_motifs=tuple(a.get("contributing_motifs", []) or []),
            contributing_mss_analytes=tuple(a.get("contributing_mss_analytes", []) or []),
            known_collisions=tuple(a.get("known_collisions", []) or []),
            substrate_sensitivities=tuple(a.get("substrate_sensitivities", []) or []),
            evidence_confidence=a.get("evidence_confidence", "unknown"),
            validation_status=a.get("validation_status", ""),
            grounding_status=a.get("grounding_status", "unknown"),
            limitations=a.get("limitations", ""),
        )
        for a in spec.get("axes", [])
    )
    return Ontology(
        version=spec.get("ontology_version", "v1"),
        name=spec.get("ontology_name", "GAIRA Biochemical Ontology"),
        n_axes=int(spec.get("n_axes", len(axes))),
        disease_label_independent=bool(spec.get("disease_label_independent", True)),
        legacy_source=spec.get("legacy_source", ""),
        notes=spec.get("notes", ""),
        legacy8_to_v11=spec.get("legacy8_to_v11", {}),
        axes=axes,
        _raw=spec,
    )


# Convenience singleton (loaded once)
_ONTOLOGY: Ontology | None = None


def ontology() -> Ontology:
    global _ONTOLOGY
    if _ONTOLOGY is None:
        _ONTOLOGY = load_ontology()
    return _ONTOLOGY
