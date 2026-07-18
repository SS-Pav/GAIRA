"""GAIRA Demo v1 — plotting helpers (Plotly).

All figures use the dark-scientific palette from `config`. Axes are always
rendered in canonical BSV order so plots are visually comparable across
tabs and modes.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import config as cfg


# ─────────────────────────────────────────────────────────────────────
# Theme appliers
# ─────────────────────────────────────────────────────────────────────

def apply_dark(fig: go.Figure, *, title: str | None = None, height: int | None = None,
                show_legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=cfg.BG_PAPER, plot_bgcolor=cfg.BG_PLOT,
        font=dict(family="Inter, SF Pro Text, Helvetica, Arial, sans-serif",
                  size=13, color=cfg.TEXT_PRIMARY),
        title=dict(text=title or "", x=0.02, xanchor="left",
                    font=dict(size=15, color=cfg.TITLE_COLOR)),
        margin=dict(l=60, r=24, t=52 if title else 18, b=48),
        height=height or 420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                     xanchor="right", x=1.0,
                     font=dict(size=11, color=cfg.TEXT_SECONDARY),
                     bgcolor=cfg.LEGEND_BG, bordercolor=cfg.LEGEND_BORDER, borderwidth=1),
        showlegend=show_legend,
        hoverlabel=dict(bgcolor="#0F172A", bordercolor=cfg.AXIS_LINE_COLOR,
                          font=dict(color=cfg.TEXT_PRIMARY, size=12)),
    )
    fig.update_xaxes(gridcolor=cfg.GRID_COLOR, zerolinecolor=cfg.ZERO_LINE_COLOR,
                      linecolor=cfg.AXIS_LINE_COLOR, tickcolor=cfg.AXIS_LINE_COLOR,
                      tickfont=dict(color=cfg.TEXT_SECONDARY, size=11),
                      title_font=dict(color=cfg.TEXT_PRIMARY, size=12))
    fig.update_yaxes(gridcolor=cfg.GRID_COLOR, zerolinecolor=cfg.ZERO_LINE_COLOR,
                      linecolor=cfg.AXIS_LINE_COLOR, tickcolor=cfg.AXIS_LINE_COLOR,
                      tickfont=dict(color=cfg.TEXT_SECONDARY, size=11),
                      title_font=dict(color=cfg.TEXT_PRIMARY, size=12))
    return fig


def apply_polar(fig: go.Figure, radial_max: float | None = None) -> go.Figure:
    fig.update_layout(polar=dict(
        bgcolor="rgba(15, 23, 42, 0.55)",
        radialaxis=dict(visible=True,
                          range=[0, radial_max] if radial_max is not None else None,
                          gridcolor="rgba(148,163,184,0.22)",
                          linecolor=cfg.AXIS_LINE_COLOR,
                          tickfont=dict(color=cfg.TEXT_MUTED, size=10),
                          showline=False, layer="below traces",
                          angle=90, tickangle=0),
        angularaxis=dict(tickfont=dict(color=cfg.TEXT_PRIMARY, size=12),
                           gridcolor="rgba(148,163,184,0.22)",
                           linecolor=cfg.AXIS_LINE_COLOR,
                           direction="clockwise", rotation=90),
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────
# Radar (BSV)
# ─────────────────────────────────────────────────────────────────────

def radar_figure(traces: list[dict], *, title: str | None = None, height: int = 480,
                  radial_max: float | None = None,
                  line_width: float = 2.5, fill_opacity: float = 0.14) -> go.Figure:
    """Render a closed 11-axis BSV radar.

    Args:
        traces: list of {'name', 'values' (dict axis→float), optional 'color'}
        title / height: standard
        radial_max: cap the radial axis (None ⇒ Plotly auto-fits)
        line_width: line thickness for each trace polygon
        fill_opacity: alpha for the translucent polygon fill (lower when many
            cohorts overlap so the polygons don't visually mask each other)

    The first axis is repeated as the last point so the polygon closes.
    NaN / missing axis values are coerced to 0.0 (since the v11 BSV is non-negative).
    """
    short_labels = [cfg.axis_short(a) for a in cfg.BSV_AXES] + [cfg.axis_short(cfg.BSV_AXES[0])]
    fig = go.Figure()
    for i, t in enumerate(traces):
        vals = []
        for a in cfg.BSV_AXES:
            v = t["values"].get(a, 0.0)
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = 0.0
            if not (v == v):                              # NaN guard
                v = 0.0
            vals.append(max(0.0, v))
        vals.append(vals[0])
        color = t.get("color", cfg.OVERLAY_COLORS[i % len(cfg.OVERLAY_COLORS)])
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=short_labels,
            name=t.get("name", f"trace {i}"),
            line=dict(color=color, width=line_width),
            fill="toself", fillcolor=_alpha(color, fill_opacity),
            hovertemplate="<b>%{theta}</b><br>%{r:.3f}<extra></extra>",
        ))
    apply_dark(fig, title=title, height=height)
    apply_polar(fig, radial_max=radial_max)
    return fig


def _alpha(hex_or_rgba: str, a: float) -> str:
    """Best-effort convert hex/rgba to rgba(...,a)."""
    if hex_or_rgba.startswith("rgba"):
        return hex_or_rgba
    if hex_or_rgba.startswith("#") and len(hex_or_rgba) == 7:
        r = int(hex_or_rgba[1:3], 16)
        g = int(hex_or_rgba[3:5], 16)
        b = int(hex_or_rgba[5:7], 16)
        return f"rgba({r},{g},{b},{a:.2f})"
    return hex_or_rgba


# ─────────────────────────────────────────────────────────────────────
# Spectrum plot
# ─────────────────────────────────────────────────────────────────────

def spectrum_figure(
    traces: list[dict],
    *,
    title: str | None = None,
    height: int = 360,
    highlight_anchors: list[float] | None = None,
    highlight_supports: list[float] | None = None,
    highlight_regions: list[tuple[float, float, str]] | None = None,
) -> go.Figure:
    """traces: list of {'name', 'x', 'y', 'color': optional}."""
    fig = go.Figure()

    if highlight_regions:
        for x0, x1, color in highlight_regions:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.16, line_width=0,
                           layer="below")

    for i, t in enumerate(traces):
        color = t.get("color", cfg.OVERLAY_COLORS[i % len(cfg.OVERLAY_COLORS)])
        fig.add_trace(go.Scatter(
            x=t["x"], y=t["y"], mode="lines",
            name=t.get("name", f"trace {i}"),
            line=dict(color=color, width=2),
            hovertemplate="%{x:.0f} cm⁻¹<br>%{y:.3f}<extra></extra>",
        ))

    if highlight_anchors:
        for cm in highlight_anchors:
            fig.add_vline(x=cm, line=dict(color="#34D399", width=1.4, dash="solid"),
                           opacity=0.85, annotation_text=f"{cm:.0f}",
                           annotation_position="top",
                           annotation_font=dict(color="#34D399", size=10))
    if highlight_supports:
        for cm in highlight_supports:
            fig.add_vline(x=cm, line=dict(color="#A5B4FC", width=1.0, dash="dot"),
                           opacity=0.7)

    apply_dark(fig, title=title, height=height)
    fig.update_xaxes(title="Wavenumber (cm⁻¹)", range=[cfg.WAVENUMBER_MIN, cfg.WAVENUMBER_MAX])
    fig.update_yaxes(title="Normalized intensity")
    return fig


# ─────────────────────────────────────────────────────────────────────
# ΔBSV bar
# ─────────────────────────────────────────────────────────────────────

def delta_bar_figure(delta: dict[str, float], *, title: str | None = None,
                       height: int = 360) -> go.Figure:
    axes = list(cfg.BSV_AXES)
    vals = [float(delta.get(a, 0.0)) for a in axes]
    colors = [cfg.DIVERGE_POS if v >= 0 else cfg.DIVERGE_NEG for v in vals]
    labels = [cfg.axis_short(a) for a in axes]
    fig = go.Figure(go.Bar(
        x=labels, y=vals, marker_color=colors,
        hovertemplate="<b>%{x}</b><br>Δ %{y:+.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=cfg.AXIS_LINE_COLOR, width=1))
    apply_dark(fig, title=title, height=height, show_legend=False)
    fig.update_yaxes(title="ΔBSV (vs reference)")
    return fig


# ─────────────────────────────────────────────────────────────────────
# Family bar (single condition)
# ─────────────────────────────────────────────────────────────────────

def axis_bar_figure(values: dict[str, float], *, title: str | None = None,
                      height: int = 320) -> go.Figure:
    axes = list(cfg.BSV_AXES)
    vals = [float(values.get(a, 0.0)) for a in axes]
    colors = [cfg.axis_color(a) for a in axes]
    labels = [cfg.axis_short(a) for a in axes]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=colors,
                              hovertemplate="<b>%{x}</b><br>%{y:.3f}<extra></extra>"))
    apply_dark(fig, title=title, height=height, show_legend=False)
    fig.update_yaxes(title="BSV value", range=[0, max(0.6, max(vals) * 1.1) if vals else 0.6])
    return fig


# ─────────────────────────────────────────────────────────────────────
# 11-axis biochemical space (PCA/UMAP)
# ─────────────────────────────────────────────────────────────────────

def _confidence_ellipse_points(x: np.ndarray, y: np.ndarray, n_std: float = 1.8,
                                  n_pts: int = 64) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (xs, ys) polygon of an n_std confidence ellipse for the points.

    Requires ≥4 points and non-degenerate covariance. Returns None otherwise so
    the caller can skip drawing a misleading envelope for sparse/colinear families.
    """
    if len(x) < 4:
        return None
    pts = np.column_stack([x, y]).astype(float)
    mu = pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return None
    # Eigendecomposition for ellipse axes
    try:
        vals, vecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    if vals.min() <= 1e-9:
        return None
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.linspace(0.0, 2.0 * np.pi, n_pts)
    radii = n_std * np.sqrt(vals)
    circle = np.column_stack([np.cos(theta) * radii[0], np.sin(theta) * radii[1]])
    ellipse = circle @ vecs.T + mu
    return ellipse[:, 0], ellipse[:, 1]


def biochemical_space_figure(
    coords: pd.DataFrame,
    *,
    title: str | None = None,
    height: int = 540,
    draw_envelopes: bool = True,
    envelope_min_points: int = 5,
) -> go.Figure:
    """coords: DataFrame with columns ['name','category','x','y','dominant_axis','regime',
    'substrate']. Color by dominant_axis. Optionally overlays translucent
    confidence ellipses for families with enough non-degenerate points (so the
    envelope is informative, not decorative)."""
    fig = go.Figure()

    if draw_envelopes:
        for axis in cfg.BSV_AXES:
            sub = coords[coords["dominant_axis"] == axis]
            if len(sub) < envelope_min_points:
                continue
            xs = sub["x"].to_numpy(dtype=float)
            ys = sub["y"].to_numpy(dtype=float)
            ell = _confidence_ellipse_points(xs, ys)
            if ell is None:
                continue
            ex, ey = ell
            fig.add_trace(go.Scatter(
                x=ex, y=ey, mode="lines",
                line=dict(color=cfg.axis_color(axis), width=1.2, dash="solid"),
                fill="toself", fillcolor=_alpha(cfg.axis_color(axis), 0.10),
                hoverinfo="skip", showlegend=False, name=f"{cfg.axis_short(axis)} envelope",
            ))

    for axis in cfg.BSV_AXES:
        sub = coords[coords["dominant_axis"] == axis]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["x"], y=sub["y"], mode="markers",
            name=cfg.axis_short(axis),
            marker=dict(color=cfg.axis_color(axis), size=10, opacity=0.85,
                        line=dict(color="rgba(255,255,255,0.18)", width=0.5)),
            customdata=np.stack([sub["name"], sub["category"], sub["regime"], sub["substrate"]], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "category: %{customdata[1]}<br>"
                "regime: %{customdata[2]}<br>"
                "substrate: %{customdata[3]}<br>"
                f"family: {cfg.axis_short(axis)}"
                "<extra></extra>"
            ),
        ))
    apply_dark(fig, title=title, height=height)
    fig.update_xaxes(title="Component 1", showgrid=True)
    fig.update_yaxes(title="Component 2", showgrid=True)
    return fig


# ─────────────────────────────────────────────────────────────────────
# Pipeline diagram (cards)
# ─────────────────────────────────────────────────────────────────────

PIPELINE_STEPS: list[tuple[str, str]] = [
    ("Grounding spectra + literature", "RamanBioLib · Gobbato · serum Ag-SERS · isotope refs · atlas · domain rules"),
    ("Spectral primitives",             "baseline · smoothing · peaks · widths · prominence · envelope · co-band patterns"),
    ("MSS + motif extraction",          "molecular spectral signatures · anchor/support/anti-evidence · class motifs"),
    ("11 biochemical families",         "noisy-OR aggregation across motifs into the 11-axis BSV"),
    ("Substrate-aware correction",      "Ag-SERS / Raman / matrix rules · physics-informed boosts and dampening"),
    ("BSV radar",                       "compact 11-axis biochemical state vector with uncertainty"),
    ("Evidence-grounded interpretation","direct + supporting evidence · caveats · class- vs candidate-level calls"),
]


def pipeline_figure(highlight_step: int | None = None) -> go.Figure:
    """Vertical-ish stacked card diagram of the GAIRA pipeline."""
    fig = go.Figure()
    n = len(PIPELINE_STEPS)
    width = 0.92
    gap = 0.04
    box_h = (1.0 - (n - 1) * gap) / n
    for i, (title, sub) in enumerate(PIPELINE_STEPS):
        y_top = 1.0 - i * (box_h + gap)
        y_bot = y_top - box_h
        is_hl = (highlight_step is not None and i == highlight_step)
        fill = "rgba(96,165,250,0.18)" if is_hl else "rgba(17,24,39,0.7)"
        stroke = "#60A5FA" if is_hl else "rgba(148,163,184,0.35)"
        fig.add_shape(type="rect", x0=(1 - width)/2, x1=(1 + width)/2, y0=y_bot, y1=y_top,
                        line=dict(color=stroke, width=1.5), fillcolor=fill)
        fig.add_annotation(x=0.5, y=(y_top + y_bot) / 2 + 0.012, xref="x", yref="y",
                            text=f"<b>{title}</b>", showarrow=False,
                            font=dict(color=cfg.TITLE_COLOR, size=14), align="center")
        fig.add_annotation(x=0.5, y=(y_top + y_bot) / 2 - 0.018, xref="x", yref="y",
                            text=sub, showarrow=False,
                            font=dict(color=cfg.TEXT_SECONDARY, size=11), align="center")
        if i < n - 1:
            # arrow
            fig.add_annotation(x=0.5, y=y_bot - gap/2, xref="x", yref="y",
                                ax=0.5, ay=y_bot + 0.005, axref="x", ayref="y",
                                showarrow=True, arrowhead=2, arrowcolor="#60A5FA",
                                arrowsize=0.9, arrowwidth=1.4, text="")
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1])
    apply_dark(fig, height=720, show_legend=False)
    fig.update_layout(margin=dict(l=8, r=8, t=8, b=8))
    return fig


# ─────────────────────────────────────────────────────────────────────
# Atlas ruler (horizontal wavenumber map with regions)
# ─────────────────────────────────────────────────────────────────────

def atlas_ruler_figure(highlight: tuple[int, int] | None = None) -> go.Figure:
    fig = go.Figure()
    palette = cfg.OVERLAY_COLORS
    for i, region in enumerate(cfg.ATLAS_REGIONS):
        color = palette[i % len(palette)]
        opacity = 0.85 if (highlight is None or region["start"] >= highlight[0] and region["end"] <= highlight[1]) else 0.35
        fig.add_shape(type="rect",
                       x0=region["start"], x1=region["end"], y0=0, y1=1,
                       fillcolor=color, opacity=opacity * 0.32,
                       line=dict(color=color, width=1))
        fig.add_annotation(
            x=(region["start"] + region["end"]) / 2, y=0.5,
            text=f"<b>{region['start']}–{region['end']}</b><br>{region['label']}",
            showarrow=False,
            font=dict(color=cfg.TEXT_PRIMARY, size=11),
            align="center",
        )
    apply_dark(fig, height=160, show_legend=False)
    fig.update_xaxes(title="Wavenumber (cm⁻¹)",
                       range=[cfg.WAVENUMBER_MIN, cfg.WAVENUMBER_MAX])
    fig.update_yaxes(visible=False, range=[0, 1])
    return fig


# ─────────────────────────────────────────────────────────────────────
# Dose-response line
# ─────────────────────────────────────────────────────────────────────

def dose_response_figure(df: pd.DataFrame, axis: str, *, title: str | None = None,
                          height: int = 340) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=df["concentration_uM"], y=df[axis], mode="lines+markers",
        line=dict(color=cfg.axis_color(axis), width=2.5),
        marker=dict(color=cfg.axis_color(axis), size=9,
                     line=dict(color="rgba(255,255,255,0.25)", width=0.5)),
        name=cfg.axis_short(axis),
        hovertemplate="%{x:.2f} µM<br>%{y:.3f}<extra></extra>",
    ))
    apply_dark(fig, title=title, height=height, show_legend=False)
    fig.update_xaxes(title="Concentration (µM)")
    fig.update_yaxes(title=f"BSV — {cfg.axis_long(axis)}")
    return fig
