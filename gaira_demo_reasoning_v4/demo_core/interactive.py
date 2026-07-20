"""Interactive (plotly) visualizations — hover + filter where static figures can't.

Used sparingly, only where interactivity is scientifically useful: the reference
family map (explore analytes) and the components→MSS→themes flow (see the
many-to-many structure). Everything else is publication-quality matplotlib.
"""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from . import theme as T

# colourblind-safe qualitative palette for family identity (position + colour + hover)
FAMILY_PALETTE = px.colors.qualitative.Safe


def pca_2d(X):
    """Deterministic 2-D PCA (sign-fixed) for visualisation only."""
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    for i in range(2):
        if Vt[i][np.argmax(np.abs(Vt[i]))] < 0:
            Vt[i] = -Vt[i]
    proj = Xc @ Vt[:2].T
    var = (S[:2] ** 2) / (S ** 2).sum()
    return proj, var


def reference_pca(ref_map, show_families=None, label_analytes=None):
    """Interactive PCA of the 167 reference analytes in frozen component space,
    coloured by biochemical family. Explanatory only — NOT the inference model."""
    proj, var = pca_2d(ref_map["coords"])
    analytes = np.array(ref_map["analytes"])
    families = np.array(ref_map["families"])
    fig = go.Figure()
    fam_order = sorted(set(families))
    for k, fam in enumerate(fam_order):
        if show_families and fam not in show_families:
            continue
        m = families == fam
        fig.add_trace(go.Scatter(
            x=proj[m, 0], y=proj[m, 1], mode="markers", name=fam,
            marker=dict(size=8, color=FAMILY_PALETTE[k % len(FAMILY_PALETTE)],
                        line=dict(width=0.5, color="white")),
            text=analytes[m], customdata=families[m],
            hovertemplate="<b>%{text}</b><br>family: %{customdata}"
                          "<br>PC1 %{x:.3f} · PC2 %{y:.3f}<extra></extra>"))
    if label_analytes:
        sel = np.isin(analytes, label_analytes)
        for a, x, y in zip(analytes[sel], proj[sel, 0], proj[sel, 1]):
            fig.add_annotation(x=x, y=y, text=a, showarrow=False, font=dict(size=9, color=T.INK),
                               yshift=10)
    fig.update_layout(
        template="simple_white", height=560,
        xaxis_title=f"PC1 ({var[0]:.0%} of variance)",
        yaxis_title=f"PC2 ({var[1]:.0%} of variance)",
        legend_title="biochemical family", margin=dict(l=40, r=20, t=30, b=40),
        font=dict(color=T.INK))
    return fig, var


def component_theme_sankey(sankey):
    """Components → MSS motifs → biochemical themes flow (many-to-many)."""
    from .figures import THEME_SHORT
    labels = (sankey["comp_nodes"]
              + sankey["motif_nodes"]
              + [THEME_SHORT.get(t, t) for t in sankey["theme_nodes"]])
    ncomp = len(sankey["comp_nodes"]); nmot = len(sankey["motif_nodes"])
    node_colors = ([T.PRIMARY_SOFT] * ncomp + [T.UP] * nmot
                   + [T.SECONDARY] * len(sankey["theme_nodes"]))
    src = [l[0] for l in sankey["links"]]
    tgt = [l[1] for l in sankey["links"]]
    val = [l[2] for l in sankey["links"]]
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=labels, color=node_colors, pad=10, thickness=12,
                  line=dict(width=0.4, color="white")),
        link=dict(source=src, target=tgt, value=val,
                  color="rgba(150,170,190,0.35)")))
    fig.update_layout(template="simple_white", height=680,
                      margin=dict(l=10, r=10, t=20, b=10), font=dict(color=T.INK, size=10))
    return fig
