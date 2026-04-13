"""
Spectral query visualization — radar, heatmaps, mean spectra overlays.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from gaira.spectral.window_panel import BSV_COMPONENTS
from gaira.spectral.bsv_projection import CohortBSV


BSV_SHORT = {
    "membrane_lipid": "Lipid", "protein_backbone": "Protein",
    "aromatic_amino_acid": "Aromatic AA", "purine_nucleotide": "Purine",
    "pyrimidine_nucleotide": "Pyrimidine", "glycan_carbohydrate": "Glycan",
    "redox_metabolite": "Redox", "nucleic_acid_backbone": "Nuc. Backbone",
}

COHORT_COLORS = [
    "#2ECC71", "#E74C3C", "#9B59B6", "#F39C12",
    "#3498DB", "#1ABC9C", "#E67E22", "#34495E",
]

BG = "#1a1a2e"


def _hex_to_rgba(h: str, a: float) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def _color_for(idx: int) -> str:
    return COHORT_COLORS[idx % len(COHORT_COLORS)]


def radar_plot(cohort_bsvs: dict[str, CohortBSV], height: int = 380) -> go.Figure:
    """BSV radar overlay — one polygon per cohort."""
    cats = [BSV_SHORT.get(c, c) for c in BSV_COMPONENTS]
    fig = go.Figure()

    for i, (cohort, cbsv) in enumerate(cohort_bsvs.items()):
        vals = [cbsv.mean_bsv[c] for c in BSV_COMPONENTS] + [cbsv.mean_bsv[BSV_COMPONENTS[0]]]
        color = _color_for(i)
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats + [cats[0]],
            fill="toself", fillcolor=_hex_to_rgba(color, 0.10),
            line=dict(color=color, width=2),
            name=cohort.replace("_", " "),
        ))

    fig.update_layout(
        polar=dict(
            bgcolor=BG,
            radialaxis=dict(visible=True, gridcolor="rgba(255,255,255,0.08)",
                            tickfont=dict(size=8, color="rgba(255,255,255,0.4)")),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)",
                             tickfont=dict(size=10, color="rgba(255,255,255,0.7)")),
        ),
        paper_bgcolor=BG, font=dict(color="rgba(255,255,255,0.8)"),
        legend=dict(font=dict(size=10, color="rgba(255,255,255,0.7)"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=60, t=30, b=30), height=height,
        showlegend=len(cohort_bsvs) > 1,
    )
    return fig


def bsv_heatmap(cohort_bsvs: dict[str, CohortBSV], height: int = 280) -> go.Figure:
    """BSV heatmap — rows = cohorts, columns = components."""
    cohorts = list(cohort_bsvs.keys())
    data = np.array([[cohort_bsvs[c].mean_bsv[comp] for comp in BSV_COMPONENTS] for c in cohorts])
    cols = [BSV_SHORT.get(c, c) for c in BSV_COMPONENTS]

    fig = go.Figure(go.Heatmap(
        z=data, x=cols, y=[c.replace("_", " ") for c in cohorts],
        colorscale="YlOrRd", texttemplate="%{z:.4f}", textfont=dict(size=10),
    ))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(color="rgba(255,255,255,0.8)"),
        xaxis=dict(tickfont=dict(size=10)), yaxis=dict(tickfont=dict(size=10)),
    )
    return fig


def delta_heatmap(
    deltas: dict[str, dict[str, float]],
    reference: str,
    height: int = 250,
) -> go.Figure:
    """Delta-vs-reference heatmap."""
    if not deltas:
        return go.Figure()

    cohorts = list(deltas.keys())
    data = np.array([[deltas[c].get(comp, 0) for comp in BSV_COMPONENTS] for c in cohorts])
    cols = [BSV_SHORT.get(c, c) for c in BSV_COMPONENTS]
    vmax = max(abs(data.min()), abs(data.max())) * 1.1 or 0.01

    fig = go.Figure(go.Heatmap(
        z=data, x=cols, y=[c.replace("_", " ") for c in cohorts],
        colorscale="RdBu_r", zmid=0, zmin=-vmax, zmax=vmax,
        texttemplate="%{z:.4f}", textfont=dict(size=10),
    ))
    fig.update_layout(
        title=dict(text=f"Delta vs {reference.replace('_', ' ')}", font=dict(size=12)),
        paper_bgcolor=BG, plot_bgcolor=BG, height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(color="rgba(255,255,255,0.8)"),
    )
    return fig


def mean_spectra_plot(
    X_norm: np.ndarray,
    wavenumbers: np.ndarray,
    cohorts: np.ndarray,
    height: int = 350,
) -> go.Figure:
    """Mean spectra overlay by cohort."""
    fig = go.Figure()
    for i, cohort in enumerate(sorted(set(cohorts))):
        mask = cohorts == cohort
        mean_spec = X_norm[mask].mean(axis=0)
        color = _color_for(i)
        fig.add_trace(go.Scatter(
            x=wavenumbers, y=mean_spec, mode="lines",
            name=cohort.replace("_", " "), line=dict(color=color, width=1.5),
        ))

    fig.update_layout(
        xaxis_title="Wavenumber (cm⁻¹)", yaxis_title="Mean Intensity (norm.)",
        paper_bgcolor=BG, plot_bgcolor=BG, height=height,
        margin=dict(l=50, r=20, t=30, b=40),
        font=dict(color="rgba(255,255,255,0.8)"),
        legend=dict(font=dict(size=9, color="rgba(255,255,255,0.7)"), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
    )
    return fig
