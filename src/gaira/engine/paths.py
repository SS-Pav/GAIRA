"""Frozen-asset path resolver.

The frozen biochemical reference space lives in a single canonical bundle,
``assets/foundation/``. This resolver prefers that bundle and falls back to the
legacy in-repo locations, so the engine loads identically whether or not the
canonical bundle is present — fully backwards compatible, no behaviour change.
"""
from __future__ import annotations
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ASSETS = REPO / "assets" / "foundation"                       # canonical frozen bundle
LEGACY_FOUNDATION = REPO / "results/v5_rebuild/foundation/artifacts"
LEGACY_ENGINE = REPO / "results/v5_rebuild/engine_v1/artifacts"
PKG_DATA = Path(__file__).resolve().parent / "data"           # source-bundled ontology/MSS yaml

_FALLBACKS = (ASSETS, LEGACY_FOUNDATION, LEGACY_ENGINE, PKG_DATA)


def frozen_file(name: str) -> Path:
    """Resolve one frozen artifact by filename: assets/foundation first, then legacy."""
    for base in _FALLBACKS:
        p = base / name
        if p.exists():
            return p
    return ASSETS / name                                      # default; clear error if missing


def foundation_dir() -> Path:
    """Directory holding manifold.json + manifold_components.npz (the NMF basis)."""
    return ASSETS if (ASSETS / "manifold.json").exists() else LEGACY_FOUNDATION
