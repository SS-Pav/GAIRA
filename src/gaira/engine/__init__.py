"""GAIRA converged reasoning engine (v6 architecture).

A deterministic, provenance-preserving biochemical reasoning engine built ON TOP OF
the frozen Raman Reference Atlas v0.1. It is NOT a classifier and NOT a latent
embedding: the frozen NMF basis is the canonical coordinate system, the Component
Registry names the latent Raman motifs, the Biochemical Ontology v2 converts motifs
into biochemical themes, the Biochemical State Vector is the canonical biochemical
representation, the Evidence Engine supplies provenance, and the radar is one
visualization of the BSV.

No opaque ML: the only model is the frozen, non-negative NMF basis applied with the
dictionary held fixed. Everything else is documented linear algebra plus curated,
versioned interpretation. Additive — the historical inference stack is untouched.
"""
from .pipeline import GAIRAEngine, GAIRAInference          # noqa: F401
from .bsv import BSVBuilder, BSV                            # noqa: F401
from .mss import MSSLayer, MSSMotif, MSSActivation          # noqa: F401
from .ontology import Ontology                              # noqa: F401
from .registry import ComponentRegistry                    # noqa: F401
from .evidence import EvidenceEngine                        # noqa: F401
from .radar import RadarBackend                             # noqa: F401
from .normalization import ReferenceFrame                   # noqa: F401
from .domain import get_domain, DOMAINS                     # noqa: F401
from .versioning import VERSIONS                            # noqa: F401
