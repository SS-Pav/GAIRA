"""GAIRA V7 runtime — orchestration around the frozen canonical engine.

Nothing in this package computes science. Preprocessing, projection, retrieval, chemistry
evidence, calibration and confidence live in `gaira.v7.canonical.engine` and only there.
"""
from .freeze import FrozenAssetError, verify as verify_frozen_assets

__all__ = ["FrozenAssetError", "verify_frozen_assets"]
