"""GAIRA V6 demo — thin presentation layer over the frozen reasoning engine.

Nothing scientific lives here. The engine, atlas, ontology, component registry,
theme weights, BSV equations, confidence engine and MSS layer are all frozen in
``gaira.engine`` and only DRIVEN by this package.
"""
from . import theme, components, figures, data          # noqa: F401
from .engine_bridge import get_bridge, Bridge, REPO, K   # noqa: F401
