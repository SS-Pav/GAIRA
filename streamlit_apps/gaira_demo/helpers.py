"""Shared helpers for the GAIRA polished demo: data loaders + Plotly styling.

Keep strictly Plotly-based (graph_objects). Uses derived demo tables only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"

# ──────────────────────────────────────────────────────────────────────
# Canonical axis order and styling
# ──────────────────────────────────────────────────────────────────────

BSV_COMPONENTS = [
    "membrane_lipid",
    "protein_backbone",
    "aromatic_amino_acid",
    "purine_nucleotide",
    "pyrimidine_nucleotide",
    "glycan_carbohydrate",
    "redox_metabolite",
    "nucleic_acid_backbone",
]

AXIS_LABELS = {
    "membrane_lipid": "Lipid",
    "protein_backbone": "Protein",
    "aromatic_amino_acid": "Aromatic AA",
    "purine_nucleotide": "Purine",
    "pyrimidine_nucleotide": "Pyrimidine",
    "glycan_carbohydrate": "Glycan",
    "redox_metabolite": "Redox",
    "nucleic_acid_backbone": "Nuc.Backbone",
}

AXIS_COLORS = {
    "membrane_lipid":      "#4C78A8",
    "protein_backbone":    "#F58518",
    "aromatic_amino_acid": "#54A24B",
    "purine_nucleotide":   "#E45756",
    "pyrimidine_nucleotide": "#72B7B2",
    "glycan_carbohydrate": "#B279A2",
    "redox_metabolite":    "#9D755D",
    "nucleic_acid_backbone": "#EECA3B",
}

DEFAULT_LINE_COLORS = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756",
    "#72B7B2", "#B279A2", "#9D755D", "#EECA3B",
    "#2CA02C", "#D62728", "#17BECF", "#BCBD22",
]


# ──────────────────────────────────────────────────────────────────────
# Loaders (cached)
# ──────────────────────────────────────────────────────────────────────

def _load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_parquet_or_csv(base: str) -> pd.DataFrame:
    pq = DATA_DIR / f"{base}.parquet"
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception:
            pass
    csv = DATA_DIR / f"{base}.csv"
    if csv.exists():
        return pd.read_csv(csv)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_atlas_explorer() -> pd.DataFrame:
    return _load_csv("atlas_explorer.csv")


@st.cache_data(show_spinner=False)
def load_corpus_summary() -> pd.DataFrame:
    return _load_csv("grounding_corpus_summary.csv")


@st.cache_data(show_spinner=False)
def load_family_counts() -> pd.DataFrame:
    return _load_csv("grounding_family_counts.csv")


@st.cache_data(show_spinner=False)
def load_axis_coverage() -> pd.DataFrame:
    return _load_csv("atlas_axis_coverage.csv")


@st.cache_data(show_spinner=False)
def load_molecule_index() -> pd.DataFrame:
    return _load_csv("grounding_molecule_index.csv")


@st.cache_data(show_spinner=False)
def load_molecule_bsv() -> pd.DataFrame:
    return _load_csv("grounding_molecule_bsv.csv")


@st.cache_data(show_spinner=False)
def load_molecule_spectra() -> pd.DataFrame:
    return _load_parquet_or_csv("grounding_molecule_spectra")


@st.cache_data(show_spinner=False)
def load_calibration_conditions() -> pd.DataFrame:
    return _load_csv("calibration_conditions.csv")


@st.cache_data(show_spinner=False)
def load_calibration_delta_bsv() -> pd.DataFrame:
    return _load_csv("calibration_delta_bsv.csv")


@st.cache_data(show_spinner=False)
def load_erg_per_conc() -> pd.DataFrame:
    return _load_csv("ergothioneine_bsv_per_concentration.csv")


@st.cache_data(show_spinner=False)
def load_erg_dose_long() -> pd.DataFrame:
    return _load_csv("ergothioneine_dose_response.csv")


@st.cache_data(show_spinner=False)
def load_erg_mean_spectra() -> pd.DataFrame:
    return _load_parquet_or_csv("ergothioneine_spectra_mean")


# ──────────────────────────────────────────────────────────────────────
# Plotly layout helpers
# ──────────────────────────────────────────────────────────────────────

def _base_layout(title: str | None = None, height: int = 420) -> dict:
    return dict(
        template="simple_white",
        font=dict(family="Inter, Helvetica, Arial, sans-serif", size=13),
        title=dict(text=title or "", x=0.02, xanchor="left", font=dict(size=15)),
        margin=dict(l=60, r=30, t=50 if title else 20, b=50),
        height=height,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1.0, font=dict(size=11),
        ),
    )


def radar_figure(
    traces: list[dict],
    title: str | None = None,
    radial_max: float | None = None,
    height: int = 480,
    show_legend: bool = True,
) -> go.Figure:
    """Build a GAIRA canonical-order radar.

    Each trace dict: {
        "name": str,
        "values": list[float] in BSV_COMPONENTS order,
        "color": str (hex),
        "fill_alpha": float in [0, 1] (optional, default 0.25),
    }
    """
    labels = [AXIS_LABELS[c] for c in BSV_COMPONENTS]
    # Close the polygon
    theta_closed = labels + [labels[0]]

    fig = go.Figure()
    max_v = 0.0
    for tr in traces:
        vals = list(tr["values"])
        r_closed = vals + [vals[0]]
        color = tr.get("color", "#4C78A8")
        alpha = tr.get("fill_alpha", 0.25)
        fig.add_trace(go.Scatterpolar(
            r=r_closed, theta=theta_closed,
            name=tr["name"],
            mode="lines+markers",
            line=dict(color=color, width=2.2),
            marker=dict(color=color, size=6),
            fill="toself",
            fillcolor=_rgba(color, alpha),
            hovertemplate=(
                "<b>%{theta}</b><br>"
                "%{r:.3f}<extra>" + tr["name"] + "</extra>"
            ),
        ))
        max_v = max(max_v, max(vals))

    if radial_max is None:
        radial_max = max(0.15, max_v * 1.15)

    fig.update_layout(
        **_base_layout(title, height=height),
        polar=dict(
            bgcolor="white",
            radialaxis=dict(
                visible=True, range=[0, radial_max],
                tickfont=dict(size=10), gridcolor="#E5E7EB",
                showline=False, layer="below traces",
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color="#222"),
                gridcolor="#E5E7EB", linecolor="#9CA3AF",
                direction="clockwise", rotation=90,
            ),
        ),
        showlegend=show_legend,
    )
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


def bsv_bar_figure(
    values: list[float],
    title: str | None = None,
    y_label: str = "BSV contribution",
    height: int = 380,
    signed: bool = False,
) -> go.Figure:
    labels = [AXIS_LABELS[c] for c in BSV_COMPONENTS]
    colors = [AXIS_COLORS[c] for c in BSV_COMPONENTS]
    if signed:
        # color by sign: up=teal, down=orange, near-zero=grey
        new_colors = []
        for v in values:
            if v > 0.002:
                new_colors.append("#2B8A3E")
            elif v < -0.002:
                new_colors.append("#C0392B")
            else:
                new_colors.append("#9CA3AF")
        colors = new_colors
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>%{y:+.4f}<extra></extra>" if signed
        else "<b>%{x}</b><br>%{y:.4f}<extra></extra>",
    ))
    layout = _base_layout(title, height=height)
    layout["margin"]["b"] = 90
    fig.update_layout(
        **layout,
        yaxis=dict(title=y_label, zeroline=True, zerolinecolor="#9CA3AF"),
        xaxis=dict(title="", tickangle=-30),
        bargap=0.25,
    )
    return fig


def spectra_overlay_figure(
    spectra: list[tuple[str, np.ndarray, np.ndarray]],
    title: str | None = None,
    height: int = 430,
    x_label: str = "Raman shift (cm⁻¹)",
    y_label: str = "Normalized intensity",
) -> go.Figure:
    fig = go.Figure()
    for i, (name, wn, y) in enumerate(spectra):
        color = DEFAULT_LINE_COLORS[i % len(DEFAULT_LINE_COLORS)]
        fig.add_trace(go.Scatter(
            x=wn, y=y, mode="lines",
            name=name, line=dict(width=1.8, color=color),
            hovertemplate=(
                "<b>" + name + "</b><br>"
                "%{x:.0f} cm⁻¹<br>%{y:.3f}<extra></extra>"
            ),
        ))
    fig.update_layout(
        **_base_layout(title, height=height),
        xaxis=dict(title=x_label, gridcolor="#F1F5F9", zeroline=False),
        yaxis=dict(title=y_label, gridcolor="#F1F5F9", zeroline=False),
        hovermode="x unified",
    )
    return fig


def add_atlas_band_shading(
    fig: go.Figure,
    atlas_df: pd.DataFrame,
    axis: str | None = None,
    alpha: float = 0.08,
) -> go.Figure:
    df = atlas_df
    if axis:
        df = df[df["primary_axis"] == axis]
    for _, row in df.iterrows():
        color = AXIS_COLORS.get(row["primary_axis"], "#888888")
        fig.add_vrect(
            x0=row["start_cm1"], x1=row["end_cm1"],
            fillcolor=_rgba(color, alpha),
            line_width=0, layer="below",
        )
    return fig


def delta_heatmap_figure(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str | None = None,
    height: int = 360,
    vmax: float | None = None,
) -> go.Figure:
    if vmax is None:
        vmax = max(0.01, float(np.nanmax(np.abs(matrix))))
    fig = go.Figure(go.Heatmap(
        z=matrix, x=col_labels, y=row_labels,
        zmin=-vmax, zmax=vmax,
        colorscale="RdBu_r", reversescale=False,
        colorbar=dict(
            title=dict(text="ΔBSV", side="right"),
            thickness=12,
        ),
        hovertemplate="<b>%{y}</b><br>%{x}<br>ΔBSV = %{z:+.4f}<extra></extra>",
    ))
    layout = _base_layout(title, height=height)
    layout["margin"]["l"] = 200
    fig.update_layout(
        **layout,
        xaxis=dict(title="Axis", tickangle=-30),
        yaxis=dict(title="", automargin=True),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────
# Misc
# ──────────────────────────────────────────────────────────────────────

def canonical_bsv(row: pd.Series) -> list[float]:
    return [float(row[c]) for c in BSV_COMPONENTS]


def canonical_delta(row: pd.Series) -> list[float]:
    return [float(row[f"delta_bsv_{c}"]) for c in BSV_COMPONENTS]


def canonical_bsv_from_cols(row: pd.Series, prefix: str = "bsv_") -> list[float]:
    return [float(row[f"{prefix}{c}"]) for c in BSV_COMPONENTS]


def format_conc(c) -> str:
    if isinstance(c, str):
        return c
    return f"{float(c):.1f} µM"
