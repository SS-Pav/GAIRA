"""GAIRA V7 — Consensus Spectral Motifs (Phase 02).

A CSM is a group of Local Spectral Motifs that several **independent** class-local
decompositions found to describe the same biochemical spectral phenomenon:

    50 pooled LSMs                       from 16 independent class-local fits
            ↓
    seven-feature similarity graph       cosine, diagnostic bands, peak positions,
                                         bootstrap behaviour, activation pattern,
                                         discounted provenance, substitutability
            ↓
    weighted Consensus Spectral Graph    an edge is a CONFIDENCE, not a decision
            ↓
    graph communities                    threshold swept, never a single cut
            ↓
    Consensus Spectral Motifs            consensus spectrum + full provenance + uncertainty

The object exists to answer one question: does the class-local decomposition of Phase 01 buy
fair capacity for rare chemistry at the cost of comparability across classes (risk R-03)? If
LSMs from different classes cannot be reintegrated, Strategy D has traded one problem for a
worse one.

**Every merge is a falsifiable hypothesis and the default is "not merged".** High spectral
cosine alone is never sufficient — biological Raman spectra share broad structure, and the
geometric-mean edge weight in `graph.py` is what enforces multiple independent lines of
evidence. Leaving motifs separate is an acceptable scientific outcome.
"""
from .csm import CSM  # noqa: F401
from .registry import CSMRegistry  # noqa: F401

__all__ = ["CSM", "CSMRegistry"]
