"""Interactive Plotly charts for the Explorer. All other figures are the audit's
publication PNGs, reused directly. Palette + encoding follow the dataviz method:
hero encoding for the 5-representation benchmark (NMF highlighted, others by marker
shape + direct labels), single-hue sequential and green/slate status elsewhere."""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go

from . import theme as T

FONT = dict(family="Inter, system-ui, sans-serif", color=T.INK, size=13)
GRID = "#eceff3"
# non-hero representation marker shapes (secondary encoding for CVD safety)
SHAPES = {"ICA": "square", "PCA": "diamond", "SparseDict": "triangle-up", "Autoencoder": "x"}
OTHER = "#8b97a6"   # muted slate for context series


def _layout(fig, h=430, xtitle="", ytitle="", legend=True):
    fig.update_layout(
        height=h, template="plotly_white", font=FONT,
        margin=dict(l=60, r=24, t=30, b=52), paper_bgcolor=T.CARD, plot_bgcolor=T.CARD,
        xaxis=dict(title=xtitle, gridcolor=GRID, zeroline=False, showline=True,
                   linecolor=T.LINE, ticks="outside", tickcolor=T.LINE),
        yaxis=dict(title=ytitle, gridcolor=GRID, zeroline=False, showline=True,
                   linecolor=T.LINE, ticks="outside", tickcolor=T.LINE),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11.5),
                    bgcolor="rgba(0,0,0,0)") if legend else None,
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=T.LINE))
    return fig


def benchmark_scatter(df):
    """Selection score vs k for every representation. NMF is the hero; the others are
    context (muted, distinct marker shapes, direct-labelled). Tie band shaded."""
    fig = go.Figure()
    top = float(df.total_score.max())
    fig.add_hrect(y0=top - 0.02, y1=top, fillcolor="#f6c945", opacity=0.16, line_width=0,
                  annotation_text="statistical tie band (0.02)", annotation_position="top left",
                  annotation_font_size=10.5)
    hover = ("<b>%{customdata[0]} k=%{x}</b><br>score %{y:.3f}<br>"
             "recon err %{customdata[1]:.3f}<br>stability %{customdata[2]:.3f}<br>"
             "sparsity %{customdata[3]:.3f}<extra></extra>")
    for name, g in df.groupby("representation"):
        g = g.sort_values("k")
        hero = name == "NMF"
        cd = np.c_[[name] * len(g), g.recon_rel_error, g.component_stability, g.loading_sparsity]
        fig.add_trace(go.Scatter(
            x=g.k, y=g.total_score, name=name, mode="lines+markers",
            line=dict(color=T.NAVY if hero else OTHER, width=3.2 if hero else 1.6),
            marker=dict(symbol="circle" if hero else SHAPES.get(name, "circle"),
                        size=11 if hero else 8, color=T.NAVY if hero else OTHER,
                        line=dict(color="white", width=1)),
            opacity=1.0 if hero else 0.85, customdata=cd, hovertemplate=hover))
    # ring the selected NMF k=24
    sel = df[(df.representation == "NMF") & (df.k == 24)]
    if len(sel):
        fig.add_trace(go.Scatter(
            x=[24], y=[float(sel.total_score.iloc[0])], mode="markers+text",
            marker=dict(symbol="circle-open", size=26, color=T.NAVY, line=dict(width=2.5)),
            text=["selected"], textposition="bottom center", textfont=dict(color=T.NAVY_D, size=11),
            showlegend=False, hoverinfo="skip"))
    _layout(fig, xtitle="latent dimension k", ytitle="multi-criteria selection score")
    fig.update_xaxes(tickvals=[4, 8, 12, 16, 24, 32])
    return fig


def transfer_bar(df):
    """Per-analyte pure-Raman → pure-Ag-SERS coordinate cosine. Green = dominant theme
    survives the transfer; slate = it lands on a different theme."""
    df = df.sort_values("coord_cosine")
    colors = [T.GOOD if p else OTHER for p in df.theme_preserved]
    med = float(df.coord_cosine.median())
    fig = go.Figure(go.Bar(
        x=df.coord_cosine, y=df.analyte, orientation="h", marker_color=colors,
        customdata=np.c_[df.raman_theme, df.sers_theme],
        hovertemplate=("<b>%{y}</b><br>coord cosine %{x:.2f}<br>"
                       "Raman theme %{customdata[0]}<br>SERS theme %{customdata[1]}<extra></extra>")))
    fig.add_vline(x=med, line_dash="dash", line_color=T.UP, line_width=1.6,
                  annotation_text=f"median {med:.2f}", annotation_font_size=11)
    _layout(fig, h=max(520, 15 * len(df)), xtitle="cos(Raman coord, Ag-SERS coord) — 1 = signature preserved",
            legend=False)
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(tickfont=dict(size=9.5))
    return fig


def hbar(labels, values, *, color=T.NAVY, xtitle="", height=None, valfmt=".3f"):
    """Single-hue horizontal bar (sequential magnitude)."""
    order = np.argsort(values)
    labels = [labels[i] for i in order]; values = [values[i] for i in order]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=color,
        text=[f"{v:{valfmt}}" for v in values], textposition="outside",
        textfont=dict(size=11, color=T.MUTED),
        hovertemplate="%{y}: %{x:" + valfmt + "}<extra></extra>"))
    _layout(fig, h=height or max(240, 34 * len(labels)), xtitle=xtitle, legend=False)
    fig.update_xaxes(range=[0, max(values) * 1.18 if values else 1])
    return fig


def dose_curve(levels, series, *, name, color, unit="µM", ytitle="theme share"):
    """A single dose-response curve (magnitude over an ordered dose axis)."""
    fig = go.Figure(go.Scatter(
        x=levels, y=series, mode="lines+markers", name=name,
        line=dict(color=color, width=2.6), marker=dict(size=8, color=color, line=dict(color="white", width=1)),
        hovertemplate="%{x} " + unit + " → %{y:.3f}<extra></extra>"))
    _layout(fig, h=360, xtitle=f"concentration ({unit})", ytitle=ytitle, legend=False)
    return fig


def component_map(df):
    """Scatter of the 24 components: variance share (x) vs bootstrap stability (y), sized by
    theme weight, coloured by purity (sequential magnitude). Hover names the chemistry."""
    fig = go.Figure(go.Scatter(
        x=df.variance_share, y=df.stability, mode="markers+text",
        text=[f"c{c}" for c in df.component], textposition="top center",
        textfont=dict(size=9, color=T.FAINT),
        marker=dict(size=8 + 40 * df.top_theme_w, color=df.purity, colorscale="Teal",
                    cmin=float(df.purity.min()), cmax=float(df.purity.max()),
                    colorbar=dict(title="purity", thickness=12, len=0.7),
                    line=dict(color="white", width=1)),
        customdata=np.c_[df.audit_label, df.top_theme, df.top_analyte, df.purity],
        hovertemplate=("<b>c%{text} · %{customdata[0]}</b><br>theme %{customdata[1]}<br>"
                       "top analyte %{customdata[2]}<br>purity %{customdata[3]:.2f}<br>"
                       "stability %{y:.2f}<extra></extra>")))
    _layout(fig, h=460, xtitle="variance share", ytitle="bootstrap stability", legend=False)
    return fig


def delta_bar(labels, deltas, *, xtitle="Δ (treated − control)"):
    """Diverging horizontal bars (signed change): warm pole up, cool pole down."""
    order = np.argsort(deltas)
    labels = [labels[i] for i in order]; deltas = [deltas[i] for i in order]
    colors = [T.UP if d > 0 else T.DOWN for d in deltas]
    fig = go.Figure(go.Bar(
        x=deltas, y=labels, orientation="h", marker_color=colors,
        hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
    fig.add_vline(x=0, line_color=T.MUTED, line_width=1)
    _layout(fig, h=max(240, 30 * len(labels)), xtitle=xtitle, legend=False)
    return fig
