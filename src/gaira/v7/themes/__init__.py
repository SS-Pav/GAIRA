"""GAIRA V7 — Phase 03: emergent biochemical theme discovery.

    49 frozen CSMs  +  frozen Phase 02.5 geometry
            ↓
    5 soft-membership models, K swept 2–15      no chemistry label is visible here
            ↓
    label-free criteria + band-based admissibility as a HARD veto
            ↓
    K selected: smallest admissible K on the contiguous Pareto plateau
            ↓
    nine validations — bootstrap · leave-one-out · neighbour consistency · modularity ·
    spectral coherence · reconstruction · value-over-CSM · source/excitation robustness ·
    ontology agreement (post hoc only)
            ↓
    themes, with membership that is soft, sparse, row-normalised and allowed to overlap

**Discovery first, names later.** Themes are Theme-01 … through discovery and validation, and
acquire a chemical name only afterwards — with a confidence, and `Unknown Theme` where the
interpretation is weak.

**Bridges and isolates are first-class.** A CSM whose membership is genuinely split stays a
bridge; a CSM no theme claims stays unassigned. Inventing a theme to absorb an isolate is the
L-03 failure mode — a motif borrowing foreign mass.

Nothing upstream is refitted. The LSM registry, the CSM dictionary, the canonical identities
and the Phase 02.5 geometry are read-only and fingerprint-verified before anything runs.
"""
from .registry import Theme, ThemeRegistry, check_name  # noqa: F401

__all__ = ["Theme", "ThemeRegistry", "check_name"]
