"""GAIRA V7 — Phase 02.5: the latent geometry of spectral motif space.

Phase 02 asked *which motifs are interchangeable* and answered: one pair. Phase 02.5 asks a
different and much larger question — **how are the motifs related, even when they are not
interchangeable** — and the answer becomes the prior structure Phase 03 builds themes on.

    50 frozen LSMs  ·  49 frozen CSMs (sensitivity control)
            ↓
    7 complementary representations       profile · peaks · band families · activations ·
            ↓                             reconstruction · provenance · edge features
    10 distance metrics, benchmarked      amplitude, shift, width, background, stability, null
            ↓
    6 null geometries                     nothing is a finding until it beats one
            ↓
    linear · nonlinear · graph · hierarchy
            ↓
    discrete vs continuous, measured      conductance · local dimension · density valleys
            ↓
    neighbourhoods, with relationship tiers
            ↓
    provisional Phase 03 priors           constrain, never decide

**Nothing here refits anything.** The LSM dictionary, the CSM dictionary, the canonical
identities and the class-local fits are read-only inputs. This phase produces geometry and
priors; it does not produce themes.

**Two firewalls.** Chemistry-class labels and source labels are excluded from every
representation and every distance used to build the geometry. They are revealed only afterwards,
to evaluate what was found. A geometry built on the class partition would rediscover the class
partition and prove nothing (risk R-01).
"""
from . import embedding, fusion, metrics, neighbourhoods, nulls, representations, structure  # noqa: F401

__all__ = ["representations", "metrics", "nulls", "embedding", "structure", "fusion",
           "neighbourhoods"]
