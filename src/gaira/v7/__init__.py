"""GAIRA V7 — the frozen Raman biochemical inference engine and its runtime platform.

The scientific architecture is frozen after Phase 09. Phase 10 adds the runtime, the public
contract and the consumer surfaces; it changes no science.

Public entry points::

    from gaira.v7 import GAIRA                 # the Python SDK
    from gaira.v7.canonical import GAIRAEngine # the frozen engine itself
    from gaira.v7.contracts import ...         # the typed public data contract

`GAIRA` is exposed lazily so that importing a leaf module — `gaira.v7.io`, say — does not drag
in pydantic and a 1-second atlas load.
"""
from __future__ import annotations

__version__ = "7.10.0"
ENGINE_VERSION = "gaira-v7-canonical-phase09"

__all__ = ["GAIRA", "__version__", "ENGINE_VERSION"]


def __getattr__(name: str):
    if name == "GAIRA":
        from gaira.v7.sdk import GAIRA as _G
        return _G
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
