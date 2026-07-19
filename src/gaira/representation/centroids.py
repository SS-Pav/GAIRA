"""Analyte-condition centroids (Phase 2 §5). Two analysis levels:
spectrum-level and centroid-level. Centroids are built per
(canonical analyte × modality × source) — NEVER across modalities, and never
across materially different acquisition conditions (source kept separate).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Centroid:
    analyte: str
    modality: str
    source: str
    n: int                 # replicate spectra averaged
    vector: np.ndarray     # mean spectrum
    dispersion: float      # mean cosine distance of members to centroid


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def build_centroids(X: np.ndarray, meta: pd.DataFrame, group_cols=("analyte", "modality", "source")):
    """Return (C, cmeta) — centroid matrix and metadata. Grouping never mixes
    modalities (modality is a grouping key)."""
    assert "modality" in group_cols, "centroids must be stratified by modality (no cross-modality averaging)"
    cents = []
    for key, idx in meta.groupby(list(group_cols)).groups.items():
        rows = X[[meta.index.get_loc(i) for i in idx]]
        mean = rows.mean(axis=0)
        disp = float(np.mean([1.0 - _cos(mean, r) for r in rows])) if len(rows) > 1 else 0.0
        kv = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        cents.append(Centroid(kv.get("analyte"), kv.get("modality"), kv.get("source"),
                              len(rows), mean, disp))
    C = np.vstack([c.vector for c in cents])
    cmeta = pd.DataFrame([dict(analyte=c.analyte, modality=c.modality, source=c.source,
                               n=c.n, dispersion=c.dispersion) for c in cents])
    return C, cmeta
