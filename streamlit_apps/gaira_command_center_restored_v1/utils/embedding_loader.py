"""Embedding + cluster + BSV-saliency data preparation for Tab 2 visuals.

All loads are precomputed-artifact-driven. Cached at the Streamlit layer.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Embedding loaders ─────────────────────────────────────────────────────


def load_embedding(build_root: Path, mode: str) -> pd.DataFrame | None:
    """Load MSS or Motif analyte-level UMAP embedding table.

    Required columns: analyte_id, broad_class, regime, support_tier, n_spectra,
    umap_1, umap_2, cluster_id, dbscan_cluster.
    Returns None if missing.
    """
    name = "mss_analyte_embedding_v1.csv" if mode.upper() == "MSS" else "motif_analyte_embedding_v1.csv"
    p = build_root / "gaira_representation_cluster_analysis_v1" / "tables" / name
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["broad_class"] = df["broad_class"].fillna("uncategorised")
    df["mode"] = mode.upper()
    return df


def load_cluster_breakdown(build_root: Path, mode: str) -> pd.DataFrame | None:
    name = "mss_cluster_breakdown_v1.csv" if mode.upper() == "MSS" else "motif_cluster_breakdown_v1.csv"
    p = build_root / "gaira_representation_cluster_analysis_v1" / "tables" / name
    if not p.exists():
        return None
    return pd.read_csv(p)


# ─── MSS-signature parsing for hover tooltips + BSV saliency ──────────────

_BAND_RX = re.compile(r"(\d+)cm-1\(DR=([+\-\d\.]+)(?:,CV=([+\-\d\.]+))?\)")


def parse_band_field(field: object) -> list[tuple[int, float, float | None]]:
    """Parse a 'shared_core_anchors'-style field into (band_cm, DR, CV)."""
    if pd.isna(field):
        return []
    out = []
    for m in _BAND_RX.finditer(str(field)):
        band = int(m.group(1))
        dr = float(m.group(2))
        cv = float(m.group(3)) if m.group(3) is not None else None
        out.append((band, dr, cv))
    return out


def load_mss_signatures(build_root: Path) -> pd.DataFrame | None:
    p = build_root / "gaira_base_4_mss_core_build_v1" / "registry" / "grounding_molecular_signatures_v4.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def top_anchors_for_class(sig_df: pd.DataFrame, broad_class: str, k: int = 3) -> str:
    """Return a short string of top-k anchor bands for a broad class."""
    if sig_df is None or sig_df.empty:
        return ""
    rows = sig_df[sig_df["analyte_class"] == broad_class]
    if rows.empty:
        return ""
    bands = parse_band_field(rows.iloc[0].get("shared_core_anchors"))
    bands_sorted = sorted(bands, key=lambda x: -abs(x[1]))[:k]
    return ", ".join(f"{b}cm⁻¹" for b, _, _ in bands_sorted)


# ─── Hybrid BSV registry (for axis labelling) ─────────────────────────────


def load_bsv_registry(build_root: Path) -> pd.DataFrame | None:
    p = build_root / "gaira_base_4_hybrid_bsv_build_v1" / "tables" / "hybrid_bsv_group_registry_v1.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def load_analyte_to_group(build_root: Path) -> pd.DataFrame | None:
    p = build_root / "gaira_base_4_hybrid_bsv_build_v1" / "tables" / "analyte_to_hybrid_group_map_v1.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def broad_class_to_group(amap: pd.DataFrame) -> dict[str, str]:
    """Map broad_class → most-common BSV group (mode of primary_group)."""
    if amap is None or amap.empty:
        return {}
    return (
        amap.groupby("broad_class")["primary_group"]
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )


# ─── BSV saliency map data ────────────────────────────────────────────────


BSV_AXES_ORDER = [f"G{i:02d}" for i in range(1, 12)]


def build_bsv_band_saliency(sig_df: pd.DataFrame,
                            amap: pd.DataFrame,
                            band_min: int = 400,
                            band_max: int = 1800,
                            band_bin: int = 10) -> tuple[np.ndarray, list[int], dict[str, dict[int, list[str]]]]:
    """Build an (axes × bands) saliency matrix.

    For every MSS signature row, parse its anchor + raman_support + sers_support
    bands, weight by |DR|, route to the BSV group of the parent broad_class,
    and accumulate into the binned (band) bucket.

    Returns:
      mat: array of shape (11, n_bins)
      bin_centers: list of band centers (cm⁻¹)
      contributors: dict[axis][bin_center] = list of broad_class strings
    """
    n_bins = (band_max - band_min) // band_bin
    bin_centers = [band_min + band_bin // 2 + i * band_bin for i in range(n_bins)]
    mat = np.zeros((len(BSV_AXES_ORDER), n_bins), dtype=float)
    contributors: dict[str, dict[int, list[str]]] = {ax: {} for ax in BSV_AXES_ORDER}

    if sig_df is None or amap is None:
        return mat, bin_centers, contributors

    cls_to_grp = broad_class_to_group(amap)

    for _, row in sig_df.iterrows():
        cls = row.get("analyte_class")
        grp = cls_to_grp.get(cls)
        if grp not in BSV_AXES_ORDER:
            continue
        ax_idx = BSV_AXES_ORDER.index(grp)
        for field in ("shared_core_anchors", "raman_support_features",
                      "sers_support_features"):
            for band, dr, _cv in parse_band_field(row.get(field)):
                if band < band_min or band >= band_max:
                    continue
                bi = (band - band_min) // band_bin
                weight = abs(dr)
                # anchors weighted higher than supports
                if field == "shared_core_anchors":
                    weight *= 2.0
                mat[ax_idx, bi] += weight
                contributors[grp].setdefault(bin_centers[bi], []).append(str(cls))

    # Normalise per axis (max=1) so shapes are comparable across axes
    for i in range(mat.shape[0]):
        m = mat[i].max()
        if m > 0:
            mat[i] = mat[i] / m
    return mat, bin_centers, contributors


def shared_band_overlay(mat: np.ndarray, threshold: float = 0.25) -> np.ndarray:
    """Per band, return how many axes have >=threshold contribution.

    >=2 indicates a shared / collision-prone band.
    """
    return (mat >= threshold).sum(axis=0)


# ─── Family-first join helpers ────────────────────────────────────────────

def attach_bsv_family(emb_df: pd.DataFrame,
                      amap: pd.DataFrame) -> pd.DataFrame:
    """Add primary_group / secondary_group columns to an embedding by analyte_id."""
    if emb_df is None:
        return None
    if amap is None:
        out = emb_df.copy()
        out["primary_group"] = None
        out["secondary_group"] = None
        return out
    cols = ["analyte_id", "primary_group", "secondary_group"]
    return emb_df.merge(amap[cols], on="analyte_id", how="left")


def family_name_lookup(bsv_reg: pd.DataFrame | None) -> dict[str, str]:
    """Map G01..G11 → 'G01 · purine_nucleotide' style label."""
    if bsv_reg is None:
        return {ax: ax for ax in BSV_AXES_ORDER}
    return {row["group_id"]: f"{row['group_id']} · {row['group_name']}"
            for _, row in bsv_reg.iterrows()}


def family_short_lookup(bsv_reg: pd.DataFrame | None) -> dict[str, str]:
    if bsv_reg is None:
        return {ax: ax for ax in BSV_AXES_ORDER}
    return {row["group_id"]: row["group_name"]
            for _, row in bsv_reg.iterrows()}
