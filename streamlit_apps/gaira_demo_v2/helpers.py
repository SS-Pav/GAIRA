"""GAIRA demo v2 — dark-theme Plotly helpers.

All figures are styled through a single `apply_dark_theme()` pass so labels,
ticks, legends, grids and polar axes stay bright on the dark page.

Data is reused from v1 (`streamlit_apps/gaira_demo/data/`) without modification.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# v2 reuses v1's derived demo assets — single source of truth.
DATA_DIR = (Path(__file__).resolve().parent.parent / "gaira_demo" / "data")


# ──────────────────────────────────────────────────────────────────────
# Dark-theme palette
# ──────────────────────────────────────────────────────────────────────

DARK_BG_PAGE   = "#0B1220"   # page bg
DARK_BG_PANEL  = "#111827"   # panel bg
DARK_BG_PLOT   = "rgba(17, 24, 39, 0.55)"   # plot inner bg
DARK_BG_PAPER  = "rgba(0, 0, 0, 0)"          # let page bg show through

TEXT_PRIMARY    = "#F1F5F9"
TEXT_SECONDARY  = "#CBD5E1"
TEXT_MUTED      = "#94A3B8"
TITLE_COLOR     = "#F8FAFC"

GRID_COLOR      = "rgba(148, 163, 184, 0.18)"
AXIS_LINE_COLOR = "#64748B"
ZERO_LINE_COLOR = "rgba(148, 163, 184, 0.55)"

LEGEND_BG       = "rgba(17, 24, 39, 0.75)"
LEGEND_BORDER   = "rgba(148, 163, 184, 0.25)"


# Canonical BSV order (unchanged from v1)
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
    "membrane_lipid":        "Lipid",
    "protein_backbone":      "Protein",
    "aromatic_amino_acid":   "Aromatic AA",
    "purine_nucleotide":     "Purine",
    "pyrimidine_nucleotide": "Pyrimidine",
    "glycan_carbohydrate":   "Glycan",
    "redox_metabolite":      "Redox",
    "nucleic_acid_backbone": "Nuc.Backbone",
}

# Brighter, dark-bg-friendly axis palette (tailwind 400s).
AXIS_COLORS = {
    "membrane_lipid":        "#60A5FA",   # blue-400
    "protein_backbone":      "#FBBF24",   # amber-400
    "aromatic_amino_acid":   "#34D399",   # emerald-400
    "purine_nucleotide":     "#F87171",   # red-400
    "pyrimidine_nucleotide": "#22D3EE",   # cyan-400
    "glycan_carbohydrate":   "#C084FC",   # purple-400
    "redox_metabolite":      "#FDE68A",   # amber-200 — warm highlight
    "nucleic_acid_backbone": "#F472B6",   # pink-400
}

OVERLAY_COLORS = [
    "#60A5FA", "#FBBF24", "#34D399", "#F87171",
    "#22D3EE", "#C084FC", "#F472B6", "#FDE68A",
    "#A78BFA", "#FB923C", "#4ADE80", "#38BDF8",
]

DIVERGE_POS = "#34D399"    # up  → emerald-400
DIVERGE_NEG = "#F87171"    # down → red-400
DIVERGE_FLAT = "#64748B"   # flat → slate-500


# ──────────────────────────────────────────────────────────────────────
# Loaders (cached, shared with v1 CSVs)
# ──────────────────────────────────────────────────────────────────────

def _csv(name: str) -> pd.DataFrame:
    p = DATA_DIR / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _parquet_or_csv(base: str) -> pd.DataFrame:
    pq = DATA_DIR / f"{base}.parquet"
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception:
            pass
    csv = DATA_DIR / f"{base}.csv"
    return pd.read_csv(csv) if csv.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_atlas_explorer() -> pd.DataFrame:          return _csv("atlas_explorer.csv")
@st.cache_data(show_spinner=False)
def load_corpus_summary() -> pd.DataFrame:          return _csv("grounding_corpus_summary.csv")
@st.cache_data(show_spinner=False)
def load_family_counts() -> pd.DataFrame:           return _csv("grounding_family_counts.csv")
@st.cache_data(show_spinner=False)
def load_axis_coverage() -> pd.DataFrame:           return _csv("atlas_axis_coverage.csv")
@st.cache_data(show_spinner=False)
def load_molecule_index() -> pd.DataFrame:          return _csv("grounding_molecule_index.csv")
@st.cache_data(show_spinner=False)
def load_molecule_bsv() -> pd.DataFrame:            return _csv("grounding_molecule_bsv.csv")
@st.cache_data(show_spinner=False)
def load_molecule_spectra() -> pd.DataFrame:        return _parquet_or_csv("grounding_molecule_spectra")
@st.cache_data(show_spinner=False)
def load_calibration_conditions() -> pd.DataFrame:  return _csv("calibration_conditions.csv")
@st.cache_data(show_spinner=False)
def load_calibration_delta_bsv() -> pd.DataFrame:   return _csv("calibration_delta_bsv.csv")
@st.cache_data(show_spinner=False)
def load_erg_per_conc() -> pd.DataFrame:            return _csv("ergothioneine_bsv_per_concentration.csv")
@st.cache_data(show_spinner=False)
def load_erg_dose_long() -> pd.DataFrame:           return _csv("ergothioneine_dose_response.csv")
@st.cache_data(show_spinner=False)
def load_erg_mean_spectra() -> pd.DataFrame:        return _parquet_or_csv("ergothioneine_spectra_mean")


# ──────────────────────────────────────────────────────────────────────
# Central dark-theme styler — called by every figure factory
# ──────────────────────────────────────────────────────────────────────

def apply_dark_theme(
    fig: go.Figure,
    *,
    title: str | None = None,
    height: int | None = None,
    show_legend: bool = True,
    margin: dict | None = None,
) -> go.Figure:
    """One-stop dark theme application for any Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG_PAPER,
        plot_bgcolor=DARK_BG_PLOT,
        font=dict(
            family="Inter, SF Pro Text, Helvetica, Arial, sans-serif",
            size=13, color=TEXT_PRIMARY,
        ),
        title=dict(
            text=title or "", x=0.02, xanchor="left",
            font=dict(size=15, color=TITLE_COLOR, family="Inter, sans-serif"),
        ),
        margin=margin or dict(l=62, r=28, t=52 if title else 18, b=52),
        height=height or 420,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1.0,
            font=dict(size=11, color=TEXT_SECONDARY),
            bgcolor=LEGEND_BG,
            bordercolor=LEGEND_BORDER, borderwidth=1,
        ),
        showlegend=show_legend,
        hoverlabel=dict(
            bgcolor="#0F172A", bordercolor=AXIS_LINE_COLOR,
            font=dict(color=TEXT_PRIMARY, size=12, family="Inter, sans-serif"),
        ),
    )
    # Cartesian axes — catch any that exist
    fig.update_xaxes(
        gridcolor=GRID_COLOR, zerolinecolor=ZERO_LINE_COLOR,
        linecolor=AXIS_LINE_COLOR, tickcolor=AXIS_LINE_COLOR,
        tickfont=dict(color=TEXT_SECONDARY, size=11),
        title_font=dict(color=TEXT_PRIMARY, size=12),
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR, zerolinecolor=ZERO_LINE_COLOR,
        linecolor=AXIS_LINE_COLOR, tickcolor=AXIS_LINE_COLOR,
        tickfont=dict(color=TEXT_SECONDARY, size=11),
        title_font=dict(color=TEXT_PRIMARY, size=12),
    )
    return fig


def apply_polar_dark(fig: go.Figure, radial_max: float | None = None) -> go.Figure:
    """Force-bright polar axes: angular labels white, radial ticks readable."""
    polar_cfg = dict(
        bgcolor="rgba(15, 23, 42, 0.55)",
        radialaxis=dict(
            visible=True,
            range=[0, radial_max] if radial_max is not None else None,
            gridcolor="rgba(148,163,184,0.22)",
            linecolor=AXIS_LINE_COLOR,
            tickfont=dict(color=TEXT_MUTED, size=10),
            showline=False, layer="below traces",
            angle=90, tickangle=0,
        ),
        angularaxis=dict(
            tickfont=dict(color=TEXT_PRIMARY, size=13,
                          family="Inter, sans-serif"),
            gridcolor="rgba(148,163,184,0.22)",
            linecolor=AXIS_LINE_COLOR,
            direction="clockwise", rotation=90,
        ),
    )
    fig.update_layout(polar=polar_cfg)
    return fig


# ──────────────────────────────────────────────────────────────────────
# Figure factories
# ──────────────────────────────────────────────────────────────────────

def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


def radar_figure(
    traces: list[dict],
    title: str | None = None,
    radial_max: float | None = None,
    height: int = 480,
    show_legend: bool = True,
) -> go.Figure:
    """Canonical-order BSV radar — dark, bright labels, shaded polygons."""
    labels = [AXIS_LABELS[c] for c in BSV_COMPONENTS]
    theta_closed = labels + [labels[0]]

    fig = go.Figure()
    max_v = 0.0
    for tr in traces:
        vals = list(tr["values"])
        r_closed = vals + [vals[0]]
        color = tr.get("color", OVERLAY_COLORS[0])
        alpha = tr.get("fill_alpha", 0.28)
        fig.add_trace(go.Scatterpolar(
            r=r_closed, theta=theta_closed,
            name=tr["name"],
            mode="lines+markers",
            line=dict(color=color, width=2.4),
            marker=dict(color=color, size=7, line=dict(color="#0B1220", width=1)),
            fill="toself",
            fillcolor=_rgba(color, alpha),
            hovertemplate=(
                "<b>%{theta}</b><br>%{r:.3f}"
                "<extra>" + tr["name"] + "</extra>"
            ),
        ))
        max_v = max(max_v, max(vals))

    if radial_max is None:
        radial_max = max(0.15, max_v * 1.18)

    apply_dark_theme(
        fig, title=title, height=height, show_legend=show_legend,
        margin=dict(l=40, r=40, t=56 if title else 20, b=40),
    )
    apply_polar_dark(fig, radial_max=radial_max)
    return fig


def bsv_bar_figure(
    values: list[float],
    title: str | None = None,
    y_label: str = "BSV contribution",
    height: int = 380,
    signed: bool = False,
    y_range: tuple[float, float] | None = None,
) -> go.Figure:
    labels = [AXIS_LABELS[c] for c in BSV_COMPONENTS]
    if signed:
        colors = [
            DIVERGE_POS if v > 0.002
            else (DIVERGE_NEG if v < -0.002 else DIVERGE_FLAT)
            for v in values
        ]
    else:
        colors = [AXIS_COLORS[c] for c in BSV_COMPONENTS]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)", width=0)),
        hovertemplate=(
            "<b>%{x}</b><br>%{y:+.4f}<extra></extra>" if signed
            else "<b>%{x}</b><br>%{y:.4f}<extra></extra>"
        ),
    ))
    apply_dark_theme(
        fig, title=title, height=height,
        margin=dict(l=62, r=28, t=52 if title else 18, b=92),
    )
    fig.update_yaxes(
        title=y_label,
        zeroline=True, zerolinecolor=ZERO_LINE_COLOR, zerolinewidth=1,
        range=list(y_range) if y_range is not None else None,
    )
    fig.update_xaxes(title="", tickangle=-30)
    fig.update_traces(marker_line_color="rgba(255,255,255,0.08)", marker_line_width=0.5)
    fig.update_layout(bargap=0.28)
    return fig


def spectra_overlay_figure(
    spectra: list[tuple[str, np.ndarray, np.ndarray]],
    title: str | None = None,
    height: int = 430,
) -> go.Figure:
    fig = go.Figure()
    for i, (name, wn, y) in enumerate(spectra):
        color = OVERLAY_COLORS[i % len(OVERLAY_COLORS)]
        fig.add_trace(go.Scatter(
            x=wn, y=y, mode="lines",
            name=name, line=dict(width=1.9, color=color),
            hovertemplate=(
                "<b>" + name + "</b><br>%{x:.0f} cm⁻¹<br>%{y:.3f}<extra></extra>"
            ),
        ))
    apply_dark_theme(fig, title=title, height=height)
    fig.update_xaxes(title="Raman shift (cm⁻¹)")
    fig.update_yaxes(title="Normalized intensity")
    fig.update_layout(hovermode="x unified")
    return fig


def add_atlas_band_shading(
    fig: go.Figure, atlas_df: pd.DataFrame,
    axis: str | None = None, alpha: float = 0.14,
) -> go.Figure:
    df = atlas_df if axis is None else atlas_df[atlas_df["primary_axis"] == axis]
    for _, row in df.iterrows():
        color = AXIS_COLORS.get(row["primary_axis"], "#64748B")
        fig.add_vrect(
            x0=row["start_cm1"], x1=row["end_cm1"],
            fillcolor=_rgba(color, alpha),
            line_width=0, layer="below",
        )
    return fig


def delta_heatmap_figure(
    matrix: np.ndarray,
    row_labels: list[str], col_labels: list[str],
    title: str | None = None, height: int = 360,
    vmax: float | None = None,
) -> go.Figure:
    if vmax is None:
        vmax = max(0.01, float(np.nanmax(np.abs(matrix))))
    # Diverging scale readable on dark bg: pink ↔ teal through deep navy
    scale = [
        [0.00, "#F87171"],   # red-400 (strong down)
        [0.35, "#7F1D1D"],   # red-900 (muted down)
        [0.50, "#0F172A"],   # near-zero → deep
        [0.65, "#065F46"],   # emerald-900 (muted up)
        [1.00, "#34D399"],   # emerald-400 (strong up)
    ]
    fig = go.Figure(go.Heatmap(
        z=matrix, x=col_labels, y=row_labels,
        zmin=-vmax, zmax=vmax,
        colorscale=scale,
        colorbar=dict(
            title=dict(text="ΔBSV", side="right",
                        font=dict(color=TEXT_PRIMARY, size=12)),
            thickness=12, outlinewidth=0,
            tickfont=dict(color=TEXT_SECONDARY, size=10),
        ),
        hovertemplate="<b>%{y}</b><br>%{x}<br>ΔBSV = %{z:+.4f}<extra></extra>",
    ))
    apply_dark_theme(
        fig, title=title, height=height,
        margin=dict(l=210, r=40, t=52 if title else 18, b=62),
        show_legend=False,
    )
    fig.update_xaxes(title="Axis", tickangle=-30)
    fig.update_yaxes(title="", automargin=True)
    return fig


# ──────────────────────────────────────────────────────────────────────
# Atlas ruler — bright, high-contrast bands with clear classification
# ──────────────────────────────────────────────────────────────────────

CLASSIFICATION_ALPHA = {
    "anchor":    0.95,
    "secondary": 0.65,
    "ambiguous": 0.38,
}
CLASSIFICATION_BORDER = {
    "anchor":    "rgba(255,255,255,0.75)",
    "secondary": "rgba(255,255,255,0.45)",
    "ambiguous": "rgba(255,255,255,0.22)",
}


def atlas_ruler_figure(
    view: pd.DataFrame,
    axes_unique: list[str],
    height: int | None = None,
) -> go.Figure:
    """High-contrast atlas band ruler for dark bg.

    `view` must have primary_axis, start_cm1, end_cm1, central_cm1,
    classification, display_label, range_label, width_cm1, candidate_axes,
    ambiguity_score, locality_score, source_count.
    """
    axis_to_row = {a: i for i, a in enumerate(axes_unique)}

    fig = go.Figure()

    # Row background stripes for visual rhythm
    for a, i in axis_to_row.items():
        fig.add_shape(
            type="rect",
            x0=440, x1=3100, xref="x",
            y0=i - 0.48, y1=i + 0.48, yref="y",
            fillcolor="rgba(148,163,184,0.055)" if i % 2 == 0 else "rgba(148,163,184,0.02)",
            line=dict(width=0), layer="below",
        )

    # Bands
    for _, row in view.iterrows():
        a = row["primary_axis"]
        if a not in axis_to_row:
            continue
        y = axis_to_row[a]
        color = AXIS_COLORS.get(a, "#64748B")
        cls = row["classification"]
        alpha = CLASSIFICATION_ALPHA.get(cls, 0.5)
        border = CLASSIFICATION_BORDER.get(cls, "rgba(255,255,255,0.3)")
        fig.add_shape(
            type="rect",
            x0=row["start_cm1"], x1=row["end_cm1"],
            y0=y - 0.34, y1=y + 0.34,
            fillcolor=_rgba(color, alpha),
            line=dict(color=border, width=1.0),
            layer="above",
        )
        fig.add_trace(go.Scatter(
            x=[row["central_cm1"]], y=[y],
            mode="markers",
            marker=dict(size=6, color="#F8FAFC",
                        line=dict(color="#0B1220", width=1)),
            showlegend=False,
            hovertemplate=(
                f"<b>{row['display_label']}</b><br>"
                f"Range: {row['range_label']} cm⁻¹  (width {row['width_cm1']})<br>"
                f"Classification: <b>{cls}</b><br>"
                f"Candidate axes: {row['candidate_axes']}<br>"
                f"Ambiguity: {row['ambiguity_score']:.2f} · "
                f"Locality: {row['locality_score']:.2f}<br>"
                f"Sources: {row['source_count']}<extra></extra>"
            ),
        ))

    # Legend entries (dummy scatters) for classification shading
    for cls, alpha in CLASSIFICATION_ALPHA.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(
                size=14, symbol="square",
                color=_rgba("#94A3B8", alpha),
                line=dict(color="rgba(255,255,255,0.55)", width=1),
            ),
            name=cls.capitalize(),
            hoverinfo="skip",
        ))

    h = height or max(260, 66 * max(1, len(axes_unique)) + 70)
    apply_dark_theme(
        fig, height=h, show_legend=True,
        margin=dict(l=140, r=30, t=34, b=60),
        title="Atlas band ruler — hover bands for details",
    )
    fig.update_xaxes(
        title="Raman shift (cm⁻¹)", range=[440, 3100],
        showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
        tickfont=dict(color=TEXT_PRIMARY, size=12),
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(axis_to_row.values()),
        ticktext=[AXIS_LABELS[a] for a in axes_unique],
        range=[-0.8, len(axes_unique) - 0.2],
        automargin=True, showgrid=False, zeroline=False,
        tickfont=dict(color=TEXT_PRIMARY, size=13,
                      family="Inter, sans-serif"),
    )
    fig.update_layout(legend=dict(
        orientation="h", yanchor="bottom", y=1.02, x=1.0, xanchor="right",
        font=dict(color=TEXT_SECONDARY, size=11),
        bgcolor=LEGEND_BG, bordercolor=LEGEND_BORDER, borderwidth=1,
    ))
    return fig


# ──────────────────────────────────────────────────────────────────────
# Utility
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
