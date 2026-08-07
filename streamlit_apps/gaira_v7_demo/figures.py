"""GAIRA V7 Phase 11 — Plotly figures.

Every figure here is drawn from an `InferenceResult` produced by the frozen engine, or from the
frozen reference dictionaries loaded for display. **Nothing in this module computes a scientific
quantity**: no projection, no similarity, no calibration, no aggregation. If a value is not in
the result it is not plotted.

The one thing this module reads outside a result is the LSM/CSM motif *spectra*, which are
reference data rather than results — the engine needs them to project, the demo needs them to
draw a motif. They are loaded from the same pinned frozen artifacts the engine reads and their
digests are verified against `gaira.v7.runtime.freeze.EXPECTED_DIGESTS` before use.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from . import theme as T


def _fig(**kw) -> go.Figure:
    f = go.Figure()
    f.update_layout(**T.plotly_layout(**kw))
    return f


def _pretty(axis: str) -> str:
    return axis.replace("_", " ")


# ── spectra ──────────────────────────────────────────────────────────────────
def raw_spectrum(x, y, title=None) -> go.Figure:
    f = _fig(height=330, margin=dict(l=48, r=24, t=30, b=44))
    if title:
        f.update_layout(title=dict(text=title, font=dict(size=13, color=T.INK)))
    f.add_trace(go.Scatter(
        x=x, y=y, mode="lines", name="raw",
        line=dict(color=T.INK_2, width=1.3),
        hovertemplate="%{x:.1f} cm⁻¹<br>%{y:.4g}<extra></extra>"))
    f.update_xaxes(title="wavenumber (cm⁻¹)")
    f.update_yaxes(title="intensity (as supplied)")
    return f


def processed_spectrum(grid, processed, bands=None, title=None) -> go.Figure:
    """The engine's own processed vector. Bands are the diagnostic wavenumbers it reported."""
    f = _fig(height=352, margin=dict(l=48, r=24, t=34, b=44))
    if title:
        f.update_layout(title=dict(text=title, font=dict(size=13, color=T.INK)))
    f.add_trace(go.Scatter(
        x=grid, y=processed, mode="lines", name="processed",
        line=dict(color=T.ACCENT, width=1.6), fill="tozeroy",
        fillcolor="rgba(91,140,255,.10)",
        hovertemplate="%{x:.0f} cm⁻¹<br>%{y:.4f}<extra></extra>"))
    if bands:
        top = float(np.max(processed)) if len(processed) else 1.0
        for b in bands:
            f.add_vline(x=b, line=dict(color=T.AMBER, width=1, dash="dot"), opacity=.5)
        f.add_trace(go.Scatter(
            x=list(bands), y=[top * 1.04] * len(bands), mode="markers",
            marker=dict(color=T.AMBER, size=7, symbol="triangle-down"),
            name="diagnostic bands",
            hovertemplate="diagnostic band<br>%{x:.0f} cm⁻¹<extra></extra>"))
    f.update_xaxes(title="wavenumber (cm⁻¹)")
    f.update_yaxes(title="normalised intensity")
    return f


def preprocessing_stages(x_raw, y_raw, grid, processed, stage: int) -> go.Figure:
    """Raw → canonical grid → processed, revealed one stage at a time.

    Only two real curves exist: what the caller supplied and what the ENGINE returned. The
    animation reveals them; it never interpolates a fictional intermediate.
    """
    f = _fig(height=380, title=dict(font=dict(size=13, color=T.INK)))
    yr = np.asarray(y_raw, float)
    yr = yr / (np.abs(yr).max() + 1e-12)      # display scaling only, stated in the caption
    if stage >= 0:
        f.add_trace(go.Scatter(
            x=x_raw, y=yr, mode="lines", name="raw (scaled for display)",
            line=dict(color=T.INK_3, width=1.1),
            opacity=1.0 if stage == 0 else .30,
            hovertemplate="%{x:.1f} cm⁻¹<extra>raw</extra>"))
    if stage >= 1:
        f.add_vrect(x0=450, x1=1800, fillcolor="rgba(91,140,255,.05)", line_width=0,
                    annotation_text="canonical window 450–1800 cm⁻¹",
                    annotation_position="top left",
                    annotation_font=dict(size=10, color=T.INK_3))
    if stage >= 2:
        f.add_trace(go.Scatter(
            x=grid, y=processed, mode="lines", name="processed by the engine",
            line=dict(color=T.ACCENT, width=1.7), fill="tozeroy",
            fillcolor="rgba(91,140,255,.10)",
            hovertemplate="%{x:.0f} cm⁻¹<br>%{y:.4f}<extra>processed</extra>"))
    f.update_xaxes(title="wavenumber (cm⁻¹)")
    f.update_yaxes(title="intensity (normalised)")
    titles = ["Raw, as supplied", "Cropped and resampled to the canonical grid",
              "Baseline removed, smoothed, L2-normalised"]
    f.update_layout(title=dict(text=titles[min(stage, 2)]))
    return f


def reconstruction(grid, processed, recon, opacity=0.85) -> go.Figure:
    """Query, reconstruction and residual on synchronised axes."""
    from plotly.subplots import make_subplots
    p, r = np.asarray(processed, float), np.asarray(recon, float)
    f = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.68, .32],
                      vertical_spacing=.07,
                      subplot_titles=("query and CSM reconstruction", "residual"))
    f.add_trace(go.Scatter(x=grid, y=p, mode="lines", name="query",
                           line=dict(color=T.INK, width=1.5),
                           hovertemplate="%{x:.0f}<br>%{y:.4f}<extra>query</extra>"), 1, 1)
    f.add_trace(go.Scatter(x=grid, y=r, mode="lines", name="reconstruction",
                           line=dict(color=T.CYAN, width=1.5), opacity=opacity,
                           hovertemplate="%{x:.0f}<br>%{y:.4f}<extra>reconstruction</extra>"),
                1, 1)
    f.add_trace(go.Scatter(x=grid, y=p - r, mode="lines", name="residual",
                           line=dict(color=T.ROSE, width=1.1), fill="tozeroy",
                           fillcolor="rgba(251,113,133,.13)",
                           hovertemplate="%{x:.0f}<br>%{y:+.4f}<extra>residual</extra>"), 2, 1)
    f.update_layout(**T.plotly_layout(height=470, margin=dict(l=48, r=24, t=52, b=44)))
    f.update_xaxes(title="wavenumber (cm⁻¹)", row=2, col=1, gridcolor=T.STROKE,
                   tickfont=dict(color=T.INK_3))
    f.update_xaxes(gridcolor=T.STROKE, row=1, col=1, tickfont=dict(color=T.INK_3))
    for r_ in (1, 2):
        f.update_yaxes(gridcolor=T.STROKE, row=r_, col=1, tickfont=dict(color=T.INK_3))
    for a in f.layout.annotations:
        a.font = dict(size=11, color=T.INK_3)
    return f


def motif_spectrum(grid, motif, motif_id, bands=None, assignment="") -> go.Figure:
    """One LSM or CSM basis spectrum from the frozen dictionary."""
    f = _fig(height=300, title=dict(text=f"{motif_id}", font=dict(size=13, color=T.INK)))
    f.add_trace(go.Scatter(
        x=grid, y=motif, mode="lines", name=motif_id,
        line=dict(color=T.CYAN, width=1.6), fill="tozeroy",
        fillcolor="rgba(34,211,238,.11)",
        hovertemplate="%{x:.0f} cm⁻¹<br>%{y:.4f}<extra></extra>"))
    for b in (bands or []):
        f.add_vline(x=b, line=dict(color=T.AMBER, width=1, dash="dot"), opacity=.55)
    f.update_xaxes(title="wavenumber (cm⁻¹)")
    f.update_yaxes(title="motif amplitude")
    if assignment:
        f.add_annotation(text=assignment[:110], xref="paper", yref="paper", x=0, y=1.14,
                         showarrow=False, font=dict(size=10, color=T.INK_3), align="left")
    return f


def overlay(grid, query, reference, ref_name, bands=None) -> go.Figure:
    f = _fig(height=340, title=dict(text=f"query vs {ref_name}",
                                    font=dict(size=13, color=T.INK)))
    f.add_trace(go.Scatter(x=grid, y=query, mode="lines", name="query",
                           line=dict(color=T.INK, width=1.5),
                           hovertemplate="%{x:.0f}<br>%{y:.4f}<extra>query</extra>"))
    f.add_trace(go.Scatter(x=grid, y=reference, mode="lines", name=ref_name,
                           line=dict(color=T.VIOLET, width=1.5),
                           hovertemplate="%{x:.0f}<br>%{y:.4f}<extra>reference</extra>"))
    for b in (bands or []):
        f.add_vline(x=b, line=dict(color=T.AMBER, width=1, dash="dot"), opacity=.45)
    f.update_xaxes(title="wavenumber (cm⁻¹)")
    f.update_yaxes(title="normalised intensity")
    return f


# ── chemistry evidence ───────────────────────────────────────────────────────
AXIS_NOTE = {
    "acylglycerol": "mono-, di- and tri-glycerides — glycerol backbone with ester-linked chains",
    "carboxylic_acid_metabolite": "small organic acids of central metabolism",
    "chromophore_pigment": "conjugated, strongly resonant pigments",
    "fatty_acid": "free fatty acids — long aliphatic chains with a terminal carboxyl",
    "free_amino_acid": "individual amino acids, not peptide-bonded",
    "mono_oligosaccharide": "simple sugars and short sugar chains",
    "nucleic_acid_polymer": "DNA and RNA polymers",
    "peptide_protein": "peptide-bonded chains — amide backbone dominates",
    "phosphate_metabolite": "phosphorylated small molecules and nucleotides",
    "phospholipid_sphingolipid": "membrane lipids with a phosphate head group",
    "polysaccharide": "long sugar polymers",
    "purine": "two-ring nitrogen heterocycles",
    "pyrimidine": "single-ring nitrogen heterocycles",
    "small_nitrogenous": "small nitrogen-bearing metabolites",
    "sterol_steroid": "fused four-ring systems",
    "sulfur_thiol_cofactor": "sulfur-bearing cofactors and thiols",
}


def chemistry_radar(chem: dict, animate_to: float = 1.0) -> go.Figure:
    names = [_pretty(a) for a in chem["axis_names"]]
    vals = [v * animate_to for v in chem["evidence_l1"]]
    conf = chem["calibrated_probability"]
    f = _fig(height=452, margin=dict(l=20, r=20, t=30, b=20))
    f.add_trace(go.Scatterpolar(
        r=vals + vals[:1], theta=names + names[:1], fill="toself", name="evidence",
        line=dict(color=T.VIOLET, width=2), fillcolor="rgba(167,139,250,.20)",
        marker=dict(size=5, color=T.VIOLET),
        customdata=[[c, AXIS_NOTE.get(a, "")] for a, c in zip(chem["axis_names"], conf)]
                   + [[conf[0], AXIS_NOTE.get(chem["axis_names"][0], "")]],
        hovertemplate=("<b>%{theta}</b><br>relative evidence %{r:.3f}"
                       "<br>calibrated confidence %{customdata[0]:.3f}"
                       "<br><i>%{customdata[1]}</i><extra></extra>")))
    f.update_layout(
        polar=dict(bgcolor="rgba(255,255,255,.018)",
                   domain=dict(x=[0.22, 0.78], y=[0.04, 0.90]),
                   radialaxis=dict(visible=True, showticklabels=False, gridcolor=T.STROKE,
                                   linecolor=T.STROKE, range=[0, max(vals + [0.01]) * 1.15]),
                   angularaxis=dict(gridcolor=T.STROKE, linecolor=T.STROKE,
                                    tickfont=dict(size=9, color=T.INK_3))),
        showlegend=False)
    return f


def chemistry_bars(chem: dict, horizontal=True) -> go.Figure:
    order = np.argsort(chem["evidence_l1"])
    names = [_pretty(chem["axis_names"][i]) for i in order]
    vals = [chem["evidence_l1"][i] for i in order]
    conf = [chem["calibrated_probability"][i] for i in order]
    notes = [AXIS_NOTE.get(chem["axis_names"][i], "") for i in order]
    cols = [T.AXIS_COLORS[i % len(T.AXIS_COLORS)] for i in order]
    f = _fig(height=452, margin=dict(l=170, r=30, t=30, b=44))
    f.add_trace(go.Bar(
        x=vals, y=names, orientation="h", marker=dict(color=cols, line_width=0),
        customdata=list(zip(conf, notes)),
        hovertemplate=("<b>%{y}</b><br>relative evidence %{x:.3f}"
                       "<br>calibrated confidence %{customdata[0]:.3f}"
                       "<br><i>%{customdata[1]}</i><extra></extra>")))
    f.update_xaxes(title="share of total evidence")
    f.update_yaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=10.5))
    return f


def chemistry_polar_bars(chem: dict) -> go.Figure:
    names = [_pretty(a) for a in chem["axis_names"]]
    vals = list(chem["evidence_l1"])
    f = _fig(height=452, margin=dict(l=20, r=20, t=30, b=20))
    f.add_trace(go.Barpolar(
        r=vals, theta=names, marker=dict(color=T.AXIS_COLORS[:len(vals)], line_width=0),
        customdata=[AXIS_NOTE.get(a, "") for a in chem["axis_names"]],
        hovertemplate="<b>%{theta}</b><br>%{r:.3f}<br><i>%{customdata}</i><extra></extra>"))
    f.update_layout(
        polar=dict(bgcolor="rgba(255,255,255,.018)",
                   domain=dict(x=[0.22, 0.78], y=[0.04, 0.90]),
                   radialaxis=dict(showticklabels=False, gridcolor=T.STROKE,
                                   linecolor=T.STROKE),
                   angularaxis=dict(gridcolor=T.STROKE, linecolor=T.STROKE,
                                    tickfont=dict(size=9, color=T.INK_3))),
        showlegend=False)
    return f


# ── motif activations ────────────────────────────────────────────────────────
def activation_bars(activation, ids, title, colour=T.CYAN, height=340,
                    highlight: str | None = None) -> go.Figure:
    a = np.asarray(activation, float)
    order = np.argsort(-a)
    f = _fig(height=height, margin=dict(l=48, r=24, t=52, b=70),
             title=dict(text=title, font=dict(size=13, color=T.INK)))
    cols = [T.AMBER if highlight and ids[i] == highlight
            else (colour if a[i] > 0 else T.STROKE) for i in order]
    f.add_trace(go.Bar(
        x=[ids[i] for i in order], y=[a[i] for i in order],
        marker=dict(color=cols, line_width=0),
        hovertemplate="<b>%{x}</b><br>activation %{y:.4f}<extra></extra>"))
    f.update_xaxes(title="", tickangle=-60, tickfont=dict(size=8))
    f.update_yaxes(title="activation")
    return f


def activation_heatmap(activation, ids, title) -> go.Figure:
    a = np.asarray(activation, float)
    n = len(a)
    cols = 10
    rows = int(np.ceil(n / cols))
    grid = np.full((rows, cols), np.nan)
    labels = np.full((rows, cols), "", dtype=object)
    order = np.argsort(-a)
    for k, idx in enumerate(order):
        grid[k // cols, k % cols] = a[idx]
        labels[k // cols, k % cols] = ids[idx]
    f = _fig(height=90 + 42 * rows, margin=dict(l=24, r=24, t=52, b=24),
             title=dict(text=title, font=dict(size=13, color=T.INK)))
    f.add_trace(go.Heatmap(
        z=grid, customdata=labels, colorscale=[[0, "rgba(34,211,238,.05)"], [1, T.CYAN]],
        showscale=True, xgap=3, ygap=3,
        colorbar=dict(thickness=10, len=.7, tickfont=dict(color=T.INK_3, size=9),
                      outlinewidth=0),
        hovertemplate="<b>%{customdata}</b><br>activation %{z:.4f}<extra></extra>"))
    f.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    f.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, autorange="reversed")
    return f


# ── retrieval ────────────────────────────────────────────────────────────────
def retrieval_bars(hits) -> go.Figure:
    names = [f"{h['rank']}. {h['molecule']}" for h in hits][::-1]
    sims = [h["similarity"] for h in hits][::-1]
    classes = [_pretty(h["chemistry_class"]) for h in hits][::-1]
    f = _fig(height=max(300, 34 * len(hits) + 90), margin=dict(l=190, r=30, t=52, b=44),
             title=dict(text="Grounded Evidence Retrieval",
                        font=dict(size=13, color=T.INK)))
    f.add_trace(go.Bar(
        x=sims, y=names, orientation="h",
        marker=dict(color=[T.ACCENT if i == len(sims) - 1 else "rgba(91,140,255,.45)"
                           for i in range(len(sims))], line_width=0),
        customdata=classes,
        hovertemplate=("<b>%{y}</b><br>CSM cosine similarity %{x:.4f}"
                       "<br>%{customdata}<extra></extra>")))
    f.update_xaxes(title="CSM cosine similarity", range=[0, 1])
    f.update_yaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=10.5))
    return f


def csm_contribution_waterfall(hit: dict) -> go.Figure:
    """How one retrieval score decomposes. The bars sum to the score exactly."""
    cs = hit["supporting_csms"]
    f = _fig(height=300, margin=dict(l=48, r=24, t=52, b=60),
             title=dict(text=f"score decomposition — {hit['molecule']}",
                        font=dict(size=13, color=T.INK)))
    f.add_trace(go.Bar(
        x=[c["csm_id"] for c in cs], y=[c["contribution"] for c in cs],
        marker=dict(color=T.CYAN, line_width=0),
        customdata=[[c["share_of_similarity"],
                     ", ".join(f"{b:.0f}" for b in c["diagnostic_bands"][:4]) or "—"]
                    for c in cs],
        hovertemplate=("<b>%{x}</b><br>contribution %{y:.4f}"
                       "<br>share of similarity %{customdata[0]:.1%}"
                       "<br>bands %{customdata[1]} cm⁻¹<extra></extra>")))
    f.add_hline(y=0, line=dict(color=T.STROKE, width=1))
    f.update_xaxes(title="", tickangle=-35, tickfont=dict(size=9))
    f.update_yaxes(title="contribution to similarity")
    return f


# ── confidence ───────────────────────────────────────────────────────────────
def confidence_gauge(conf: dict) -> go.Figure:
    v = conf["overall"]
    f = go.Figure(go.Indicator(
        mode="gauge+number", value=v, number=dict(valueformat=".3f",
                                                  font=dict(size=34, color=T.INK)),
        gauge=dict(
            axis=dict(range=[0, 1], tickcolor=T.INK_3, tickfont=dict(size=9, color=T.INK_3)),
            bar=dict(color=T.ACCENT, thickness=.7),
            bgcolor="rgba(255,255,255,.03)", borderwidth=0,
            steps=[dict(range=[0, .4], color="rgba(251,113,133,.13)"),
                   dict(range=[.4, .7], color="rgba(251,191,36,.13)"),
                   dict(range=[.7, 1], color="rgba(52,211,153,.13)")])))
    f.update_layout(**T.plotly_layout(height=210, margin=dict(l=24, r=24, t=18, b=8)))
    return f


def confidence_factors(conf: dict) -> go.Figure:
    """The two factors whose PRODUCT is the confidence, shown as such."""
    ev = conf["reconstruction_explained_variance"]
    s1 = conf["top1_confidence"]
    f = _fig(height=230, margin=dict(l=170, r=40, t=52, b=44),
             title=dict(text="confidence = explained variance × top-1 similarity",
                        font=dict(size=12.5, color=T.INK)))
    f.add_trace(go.Bar(
        x=[ev, s1, conf["overall"]],
        y=["CSM explained variance", "top-1 similarity", "→ overall confidence"],
        orientation="h",
        marker=dict(color=[T.CYAN, T.VIOLET, T.ACCENT], line_width=0),
        text=[f"{ev:.3f}", f"{s1:.3f}", f"{conf['overall']:.3f}"],
        textposition="outside", textfont=dict(color=T.INK, size=11),
        hovertemplate="<b>%{y}</b><br>%{x:.4f}<extra></extra>"))
    f.update_xaxes(range=[0, 1.12], title="")
    f.update_yaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=10.5))
    return f


# ── provenance ───────────────────────────────────────────────────────────────
def provenance_sankey(prov: dict, chem: dict) -> go.Figure:
    """spectrum → LSM → CSM → chemistry → molecule, with real weights."""
    labels, colors = ["spectrum"], [T.INK_3]
    src, tgt, val, lcol = [], [], [], []

    lsm = prov["lsm_layer"][:6]
    csm = prov["csm_layer"][:6]
    che = prov["chemistry_layer"][:4]
    mol = prov["molecule_layer"][:4]

    i0 = 1
    for n in lsm:
        labels.append(n["identifier"]); colors.append(T.CYAN)
    i1 = len(labels)
    for n in csm:
        labels.append(n["identifier"]); colors.append(T.ACCENT)
    i2 = len(labels)
    for n in che:
        labels.append(_pretty(n["identifier"])); colors.append(T.VIOLET)
    i3 = len(labels)
    for n in mol:
        labels.append(n["identifier"]); colors.append(T.GREEN)

    for k, n in enumerate(lsm):
        src.append(0); tgt.append(i0 + k); val.append(max(n["weight"], 1e-4))
        lcol.append("rgba(34,211,238,.22)")
    for k, n in enumerate(csm):
        src.append(i0 + min(k, len(lsm) - 1) if lsm else 0)
        tgt.append(i1 + k); val.append(max(n["weight"], 1e-4))
        lcol.append("rgba(91,140,255,.22)")
    for k, n in enumerate(che):
        src.append(i1); tgt.append(i2 + k); val.append(max(n["weight"], 1e-4))
        lcol.append("rgba(167,139,250,.22)")
    for k, n in enumerate(mol):
        src.append(i2); tgt.append(i3 + k); val.append(max(n["weight"], 1e-4))
        lcol.append("rgba(52,211,153,.22)")

    f = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=labels, color=colors, pad=16, thickness=13,
                  line=dict(color="rgba(255,255,255,.10)", width=.5)),
        link=dict(source=src, target=tgt, value=val, color=lcol)))
    f.update_layout(**T.plotly_layout(height=430, margin=dict(l=10, r=10, t=44, b=10),
                                      title=dict(text="Evidence chain",
                                                 font=dict(size=13, color=T.INK)),
                                      font=dict(size=10, color=T.INK_2)))
    return f


# ── architecture page ────────────────────────────────────────────────────────
ARCH_STAGES = [
    ("Raw spectrum", "any grid, any intensity scale",
     "A Raman spectrum as the instrument produced it. GAIRA makes no assumption about its "
     "sampling, its range or its units."),
    ("Preprocessing", "450–1800 cm⁻¹ · 676 bins · asLS · SG(9,3) · L2",
     "Crop, resample, remove the fluorescence background by asymmetric least squares, smooth "
     "with a Savitzky–Golay filter, normalise to unit length. The last step discards absolute "
     "intensity — which is why no output can be read as a concentration."),
    ("50 Local Spectral Motifs", "non-negative basis, learned within chemistry class",
     "Express the spectrum as a sum of learned building blocks that may only be ADDED, never "
     "subtracted. About ten of fifty are active for a typical sample. Diagnostic only — no "
     "later stage reads it."),
    ("49 Consensus Spectral Motifs", "THE canonical representation",
     "The merged, deduplicated dictionary. These 49 numbers are the coordinates every "
     "downstream layer reads, and the only ones. Chemistry accuracy on unseen molecules peaks "
     "here at 0.855 and falls for every layer built above it."),
    ("Grounded retrieval", "cosine over 154 reference molecules",
     "Rank reference molecules by how closely their activation pattern points in the same "
     "direction. Because a cosine is an inner product, each score decomposes EXACTLY into "
     "per-motif contributions — which is what makes the ranking explainable."),
    ("Chemistry Evidence", "16 axes, hierarchical model, calibrated",
     "Collapse the activation onto 16 chemistry families, letting coarse chemistry gently "
     "inform fine chemistry without ever excluding it. The result is RELATIVE evidence."),
    ("Report", "confidence · audit · provenance · deterministic text",
     "Confidence is explained variance times top-1 similarity — deliberately unforgiving. "
     "Provenance resolves every claim back to specific wavenumbers. The interpretation "
     "paragraph is template-driven; no language model is involved."),
]


def architecture_flow(selected: int | None = None) -> go.Figure:
    """Seven stages left to right. Sub-labels live in the detail card, not in the boxes —
    seven captions across one row collided at every width worth supporting."""
    n = len(ARCH_STAGES)
    f = _fig(height=170, margin=dict(l=14, r=14, t=22, b=22))
    xs = np.linspace(0.075, 0.925, n)
    half = 0.055
    for i, (name, _sub, _body) in enumerate(ARCH_STAGES):
        active = selected == i
        f.add_shape(type="rect", x0=xs[i] - half, x1=xs[i] + half, y0=.32, y1=.78,
                    xref="paper", yref="paper",
                    fillcolor="rgba(91,140,255,.20)" if active else "rgba(255,255,255,.045)",
                    line=dict(color=T.ACCENT if active else "rgba(255,255,255,.11)",
                              width=1.7 if active else 1))
        f.add_annotation(x=xs[i], y=.55, xref="paper", yref="paper",
                         text=f"<b>{name.replace(' ', '<br>')}</b>", showarrow=False,
                         font=dict(size=8.8, color=T.INK if active else T.INK_2),
                         align="center")
        if i:
            f.add_annotation(x=xs[i] - half, y=.55, ax=-22, ay=0,
                             xref="paper", yref="paper", showarrow=True, arrowhead=2,
                             arrowsize=1.0, arrowwidth=1.1, arrowcolor=T.INK_3, text="")
    f.update_xaxes(visible=False, range=[0, 1])
    f.update_yaxes(visible=False, range=[0, 1])
    return f
