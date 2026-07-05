"""Helpers around the BSV saliency matrix: per-axis top bands + axis overlap network."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from utils.embedding_loader import BSV_AXES_ORDER


# ─── Per-axis top bands (for the axis selector under the heatmap) ─────────

def top_bands_for_axis(mat: np.ndarray, bin_centers: list[int],
                       contributors: dict[str, dict[int, list[str]]],
                       axis_id: str, k: int = 10) -> pd.DataFrame:
    if axis_id not in BSV_AXES_ORDER:
        return pd.DataFrame(columns=["band_cm", "weight", "top_classes"])
    ax_idx = BSV_AXES_ORDER.index(axis_id)
    weights = mat[ax_idx]
    order = np.argsort(weights)[::-1]
    rows = []
    for j in order:
        w = float(weights[j])
        if w <= 0:
            break
        bc = bin_centers[j]
        classes = contributors.get(axis_id, {}).get(bc, [])
        top_classes = ", ".join(c for c, _ in Counter(classes).most_common(3))
        rows.append({"band_cm": bc, "weight": round(w, 3),
                     "top_classes": top_classes})
        if len(rows) >= k:
            break
    return pd.DataFrame(rows)


# ─── Axis overlap network ────────────────────────────────────────────────

def axis_overlap_edges(mat: np.ndarray, bin_centers: list[int],
                       threshold: float = 0.30) -> pd.DataFrame:
    """For every axis pair, count bands where both axes weight ≥ threshold.

    Returns a long-form edges DataFrame with shared band counts + sample bands.
    """
    n_axes = mat.shape[0]
    rows = []
    for i in range(n_axes):
        for j in range(i + 1, n_axes):
            shared_idx = np.where((mat[i] >= threshold) & (mat[j] >= threshold))[0]
            if len(shared_idx) == 0:
                continue
            shared_bands = [bin_centers[k] for k in shared_idx]
            avg_weight = float(np.mean(mat[[i, j]][:, shared_idx]))
            rows.append({
                "axis_a": BSV_AXES_ORDER[i],
                "axis_b": BSV_AXES_ORDER[j],
                "n_shared_bands": int(len(shared_idx)),
                "shared_bands": ", ".join(f"{b}" for b in shared_bands[:8]),
                "mean_weight": round(avg_weight, 3),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("n_shared_bands", ascending=False).reset_index(drop=True)


# ─── Curated interpretation labels for canonical edges ──────────────────

EDGE_INTERPRETATION = {
    ("G01", "G02"): "purine ring breath 720-740 shared between nucleotides + metabolites",
    ("G04", "G05"): "phosphate (PO₂⁻) at 1080 overlaps glycan C-O-C anomeric",
    ("G06", "G07"): "Phe 1003 shared with protein backbone",
    ("G06", "G08"): "amide-I 1655 ↔ lipid C=C cis stretch — needs co-fire to disambiguate",
    ("G08", "G09"): "lipid CH₂ vs sterol skeletal — triglyceride boundary",
    ("G05", "G09"): "1080 region shared with sterol skeletal",
    ("G07", "G02"): "aromatic ring vs UA 1517 carotenoid-overlap zone",
    ("G06", "G10"): "S-S / C-S protein cysteine bridges",
}


def edge_interpretation(a: str, b: str) -> str:
    pair = tuple(sorted([a, b]))
    return EDGE_INTERPRETATION.get(pair, "")


# ─── Traffic-light shared-band classification ────────────────────────────

def traffic_light_overlay(mat: np.ndarray, threshold: float = 0.30) -> np.ndarray:
    """Return per-band axis count; caller maps to colours (1=clean, 2=shared, ≥3=collision)."""
    return (mat >= threshold).sum(axis=0)


def traffic_light_colors(counts: np.ndarray) -> list[str]:
    """Map per-band counts to green / orange / red dark-theme colours."""
    out = []
    for c in counts:
        if c <= 0:
            out.append("#30363d")            # neutral grey for zero
        elif c == 1:
            out.append("#7ee787")            # clean
        elif c == 2:
            out.append("#ffa657")            # shared
        else:
            out.append("#ff7b72")            # collision
    return out


# ─── Per-family node sizing for the network ──────────────────────────────

def axis_node_weights(mat: np.ndarray, threshold: float = 0.30) -> np.ndarray:
    """For each axis, count bands above threshold (→ node size)."""
    return (mat >= threshold).sum(axis=1)


def axis_overlap_matrix(mat: np.ndarray, threshold: float = 0.30) -> np.ndarray:
    """Symmetric (axes × axes) matrix of shared-band counts.

    Diagonal is set to 0 (we don't count an axis against itself).
    """
    binary = (mat >= threshold).astype(int)
    co = binary @ binary.T
    np.fill_diagonal(co, 0)
    return co.astype(int)


def _parse_band_field(field: object) -> list[tuple[int, float, float | None]]:
    """Parse a 'shared_core_anchors'-style field into (band_cm, DR, CV)."""
    import re
    rx = re.compile(r"(\d+)cm-1\(DR=([+\-\d\.]+)(?:,CV=([+\-\d\.]+))?\)")
    if field is None:
        return []
    if isinstance(field, float) and np.isnan(field):
        return []
    out = []
    for m in rx.finditer(str(field)):
        out.append((int(m.group(1)), float(m.group(2)),
                    float(m.group(3)) if m.group(3) is not None else None))
    return out


def family_band_frequencies(sig_df, broad_classes: list[str],
                            top_k: int = 10) -> pd.DataFrame:
    """Aggregate most-frequent anchor+support bands across MSS signatures
    whose `analyte_class` is in `broad_classes` (the broad classes that map
    to a single BSV family).

    Returns columns: band_cm, count, mean_DR.
    """
    if sig_df is None or sig_df.empty:
        return pd.DataFrame(columns=["band_cm", "count", "mean_DR"])
    rows = sig_df[sig_df["analyte_class"].isin(broad_classes)]
    if rows.empty:
        return pd.DataFrame(columns=["band_cm", "count", "mean_DR"])
    bag: dict[int, list[float]] = {}
    for _, r in rows.iterrows():
        for field in ("shared_core_anchors",
                      "raman_support_features",
                      "sers_support_features"):
            for band, dr, _cv in _parse_band_field(r.get(field)):
                bag.setdefault(band, []).append(abs(dr))
    out = pd.DataFrame([
        {"band_cm": band, "count": len(weights),
         "mean_DR": round(float(np.mean(weights)), 3)}
        for band, weights in bag.items()
    ])
    if out.empty:
        return out
    out = out.sort_values(["count", "mean_DR"], ascending=[False, False]).head(top_k)
    return out.reset_index(drop=True)
