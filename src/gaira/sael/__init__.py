"""Spectral Anchor Evidence Layer (SAEL) v1.

Literature-grounded, contrast-aware, physically-anchored expected-BSV layer.

Pipeline:
    Part A  extractor       — pull contrast / direction / peak / molecule rows
                              from DB evidence sources
    Part B  anchor_builder  — cluster extracted peaks into anchor windows
    Part C  delta_builder   — build anchor-based expected-delta objects per contrast
    Part D  bsv_derivation  — derive expected BSV from anchors, ambiguity-aware

Does NOT touch:
    - direct spectral BSV engine (src/gaira/spectral/)
    - calibration datasets
    - v4 Streamlit demo
"""
# Submodules are imported lazily to allow partial use during development.
# Callers should import from submodules directly:
#   from gaira.sael.extractor import extract_anchor_evidence
#   from gaira.sael.anchor_builder import build_sael_anchor_windows
#   ...
