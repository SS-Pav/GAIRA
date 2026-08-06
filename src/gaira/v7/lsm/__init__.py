"""GAIRA V7 — Local Spectral Motifs (Phase 01).

Deterministic decomposition of the FROZEN atlas components into reusable spectral
substructures. The atlas, its projection and its fingerprint are unchanged; only the
interpretation layer gains resolution.
"""
from .motif import LSM, Band  # noqa: F401
from .registry import LSMRegistry  # noqa: F401

__all__ = ["LSM", "Band", "LSMRegistry"]
