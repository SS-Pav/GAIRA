"""GAIRA V7 — Local Spectral Motifs (canonical Phase 01).

An LSM is a row of `H_c` from an **independent class-local non-negative factorisation** over
**balanced canonical references**:

    balanced reference corpus            one canonical molecule = one reference unit
            ↓
    split by chemistry class             X → {X_c}
            ↓
    independent NMF per class            X_c ≈ W_c H_c,  adaptive k_c,  no global competition
            ↓
    Local Spectral Motifs                the stability-selected rows of H_c

This is the object defined by `GAIRA_v7_rebuild/architecture/LEARNING_MODE_ARCHITECTURE.md`
Stage 1 and `TERMINOLOGY_AND_DEFINITIONS.md`.

**The frozen V5 atlas is NOT an input here** (principle P-15). It is used only as a baseline
control and a benchmark comparator. The decomposition that operates on the frozen atlas lives
in `gaira.v7.atlas_decomposition` and its objects are Atlas Component Substructures, not LSMs.
"""
from .lsm import LSM  # noqa: F401
from .registry import LSMRegistry  # noqa: F401

__all__ = ["LSM", "LSMRegistry"]
