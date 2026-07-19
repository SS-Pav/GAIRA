"""Stage B0 — replicate aggregation and train-fitted analysis scaling (re-exported).

Visualization-only scalings are kept separate and never enter metrics.
"""
from __future__ import annotations
from .normalization import aggregate, AGGREGATORS, QuantileScaler, viz_scale  # noqa: F401
