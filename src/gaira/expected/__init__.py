"""Expected-BSV layer v2 — literature-grounded, contrast-aware, ambiguity-aware.

NOT the observed spectral BSV. This module works on the LITERATURE side only:
evidence → expected biochemical shifts for specific contrasts.

Public surface:
    axis_mapping       — assigned_group + molecule → BSV axis (single source of truth)
    axis_audit         — per-axis evidence summary
    anchor_windows     — peak clustering with ambiguity flags
    delta_objects      — contrast-specific expected-delta registry
    comparator_v2      — ambiguity-aware expected comparator builder
"""
from gaira.expected.axis_mapping import (
    BSV_AXES,
    assigned_row_to_axis,
    AXIS_ANCHOR_HINTS,
)
from gaira.expected.axis_audit import build_axis_audit
from gaira.expected.anchor_windows import build_anchor_window_registry
from gaira.expected.delta_objects import (
    ExpectedDelta,
    CONTRAST_REGISTRY,
    build_expected_delta_objects,
)
from gaira.expected.comparator_v2 import (
    ExpectedComparatorV2,
    build_expected_comparator_v2,
)

__all__ = [
    "BSV_AXES", "assigned_row_to_axis", "AXIS_ANCHOR_HINTS",
    "build_axis_audit",
    "build_anchor_window_registry",
    "ExpectedDelta", "CONTRAST_REGISTRY", "build_expected_delta_objects",
    "ExpectedComparatorV2", "build_expected_comparator_v2",
]
