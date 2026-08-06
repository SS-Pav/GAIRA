"""GAIRA V7 — Phase 04.5: hierarchical NMF over frozen CSM activations.

    375 spectra → frozen projection engine → A ∈ ℝ₊^{375×49} CSM activations
            ↓  NMF on A — not on spectra, not on a similarity matrix, not on a graph
    W (375×K) programme activations · H (K×49) which CSMs each programme uses
            ↓  frozen H
    inference: spectrum → 49 CSM activations → NNLS onto H → Meta Component vector

A Meta Component is a pattern of **motif usage**, not of spectral similarity: two spectra can
share one without sharing a band. The Phase 02.5 geometry enters only as a one-sided smoothness
prior that encourages nearby CSMs to co-activate and can never push distant CSMs apart.

**CSMs remain the canonical inference representation** unless Meta Components demonstrate
measurable benefit. This layer is a candidate, and the phase is written to be able to reject it.
"""
from . import evaluation, factorization, perturbations  # noqa: F401

__all__ = ["factorization", "perturbations", "evaluation"]
