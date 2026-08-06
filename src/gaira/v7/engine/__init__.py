"""GAIRA V7 — Phase 04: the frozen projection engine.

    new Raman spectrum
        ↓  QC, measured before anything is interpreted
    LSM activations          projection onto the frozen 50-motif dictionary
        ↓
    CSM activations          + per-CSM disagreement
        ↓
    theme activations        accepted themes only; the rejected theme feeds uncertainty
        ↓
    Biochemical State Vector absolute, non-negative
        ↓
    latent geometry          out-of-sample extension into the frozen manifold
        ↓
    SpectrumState            coordinates, neighbours, OOD, residual, confidence, provenance

**Nothing is fitted.** Every layer above was frozen by Phases 00–03 and is fingerprint-verified
on load; this package only infers activations against it. A spectrum's output depends on that
spectrum and the atlas, never on its batch-mates.
"""
from .inference import ENGINE_VERSION, FrozenAtlas, project_spectrum  # noqa: F401
from .state import SpectrumState  # noqa: F401

__all__ = ["FrozenAtlas", "project_spectrum", "SpectrumState", "ENGINE_VERSION"]
