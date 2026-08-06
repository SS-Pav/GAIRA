"""GAIRA V7 — Atlas Component Substructures (Phase 01).

Deterministic decomposition of the FROZEN atlas components into reusable spectral
substructures. The atlas, its projection and its fingerprint are unchanged; only the
interpretation layer gains resolution.
"""
from .motif import ACS, Band  # noqa: F401
from .registry import ACSRegistry  # noqa: F401

__all__ = ["ACS", "Band", "ACSRegistry"]
