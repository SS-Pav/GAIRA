"""GAIRA engine — versioned artifact registry (Part 13).

Every layer is independently versioned so the ONTOLOGY can evolve without touching
the FROZEN Raman coordinates. A single VERSIONS object records the version and
provenance of every component of the inference stack and is stamped into every
inference output.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Versions:
    reference_atlas: str = "v0.1"
    atlas_fingerprint: str = "09ed804a40836f4a05a91ba10900cded"
    component_registry: str = "v1.0"
    biochemical_ontology: str = "v2.0"
    component_theme_weights: str = "v1.0"
    reference_normalization: str = "v1.0"
    bsv: str = "v2.0"
    radar: str = "v2.0"
    interpretation_engine: str = "v1.0"
    engine_build: str = "v1.0"

    def as_dict(self):
        return asdict(self)


VERSIONS = Versions()

# what each version layer is allowed to change WITHOUT invalidating the layer below
LAYER_INDEPENDENCE = {
    "reference_atlas": "frozen — changing it invalidates everything above",
    "component_registry": "may add evidence fields; must not change component identity",
    "biochemical_ontology": "may re-theme freely; does not touch atlas coordinates",
    "component_theme_weights": "may re-weight; recomputed from registry + ontology",
    "reference_normalization": "recomputed only if the atlas changes",
    "bsv": "may change aggregation/normalisation math; documented in equations",
    "radar": "presentation only",
    "interpretation_engine": "domain context + evidence prose; no numeric effect on BSV",
}
