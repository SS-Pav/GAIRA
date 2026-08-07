"""GAIRA V7 extension contracts. Specifications, not implementations.

    from gaira.v7.plugins import modality, context
    modality.get(Modality.RAMAN).admit(x, y, {})     # works
    modality.get(Modality.AG_SERS).admit(x, y, {})   # raises NotImplementedAdapter

Nothing here fabricates a scientific result, and `tests/test_v7_phase10_plugins.py` asserts it.
"""
from . import context, modality
from .protocols import (ContextFraming, InterpretationAdapter, ModalityAdapter, ModalityDecision,
                        NotImplementedAdapter, SampleContextAdapter, TrajectoryAdapter)

__all__ = ["modality", "context", "ModalityAdapter", "SampleContextAdapter",
           "InterpretationAdapter", "TrajectoryAdapter", "ModalityDecision", "ContextFraming",
           "NotImplementedAdapter"]
