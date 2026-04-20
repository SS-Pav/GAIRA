"""GAIRA v4 — Spectral Query Page.

Improvements over v3:
- Clearer expected comparator section with card grid
- Expected-comparator trust graphs per cohort (literature-grounded)
- Overlay radars demoted and clearly labeled "visual comparison only"
- Similarity matrix + alignment + delta-shift as primary validation figures
- Consistent terminology throughout
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

from gaira.spectral.dataset_registry import discover_datasets
from gaira.spectral.dataset_loader import load_dataset
from gaira.spectral.preprocessing import preprocess
from gaira.spectral.window_panel import extract_window_features, BSV_COMPONENTS
from gaira.spectral.bsv_projection import project_to_bsv, compute_cohort_bsvs, compute_deltas
from gaira.spectral.plots import (
    radar_plot, bsv_heatmap, delta_heatmap, mean_spectra_plot,
    BSV_SHORT, BG, _hex_to_rgba, _color_for,
)
from gaira.spectral.band_drivers import compute_per_cohort_window_importance
from gaira.spectral.explanation import annotate_windows, THEME_DISPLAY
from gaira.spectral.expected_bsv import build_expected_comparators, get_cohort_display_name
from gaira.spectral.comparison import (
    compute_delta_comparison, compute_cross_matrix_normalized,
    get_substrate_context, generate_interpretation,
)

# Reuse text-query trust graph builder for expected comparator traversals
from gaira.retrieval.text_query_retriever import TextQueryRetriever
from gaira.retrieval.motif_theme_mapper import map_evidence_to_motifs_themes_bsv as tq_mtb_map
from gaira.retrieval.literature_bsv_builder import build_literature_bsv_profile
from gaira.retrieval.trust_graph_builder import build_per_condition_traversals
from gaira.retrieval.trust_graph_render import render_trust_graph

import warnings; warnings.filterwarnings("ignore", category=FutureWarning)


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


@st.cache_resource
def get_text_retriever():
    r = TextQueryRetriever()
    r.load_sources()
    return r


# ── Page ───────────────────────────────────────────────────────────────

st.header("🔬 Spectral Query")
st.caption("Observed spectral BSV is primary. Expected literature comparator is post-hoc.")

datasets = discover_datasets()
if not datasets:
    st.error("No target datasets found. Is the SSD mounted?")
    st.stop()

S = st.session_state


def _dn(coh):
    return get_cohort_display_name(S.get("sq4_dsid", ""), coh)


with st.sidebar:
    st.header("Spectral Query Settings")
    dm = {d.display_name: d for d in datasets}
    sel = dm[st.selectbox("Dataset:", list(dm.keys()))]
    st.caption(f"`{sel.dataset_id}` · {sel.n_spectra} spectra")
    if sel.note:
        st.caption(sel.note)
    for c, n in sorted(sel.cohorts.items()):
        st.markdown(f"- {get_cohort_display_name(sel.dataset_id, c)} (n={n})")

    st.divider()
    cl = sorted(sel.cohorts.keys())
    dr_default = "healthy_control" if "healthy_control" in cl else cl[0]
    ref = st.selectbox("Reference cohort:", cl, index=cl.index(dr_default))

    st.divider()
    use_gemini = st.toggle("Use Gemini for interpretation", value=False)

    st.divider()
    run = st.button("Run Spectral Query", type="primary")

# ── Pipeline ───────────────────────────────────────────────────────────

if run:
    dsid = sel.dataset_id
    with st.spinner("Processing..."):
        ds = load_dataset(dsid)
        Xn, prep = preprocess(ds)
        wf = extract_window_features(Xn, ds.wavenumbers)
        bm = project_to_bsv(wf)
        cb = compute_cohort_bsvs(bm, ds.cohorts)
        dt = compute_deltas(cb, ref)

        drv = compute_per_cohort_window_importance(wf, ds.cohorts, ref)
        ann = {c: annotate_windows(w) for c, w in drv.items()}

        exp = build_expected_comparators(dsid, ds.cohort_names)
        sub = get_substrate_context(dsid)
        means = {c: v.mean_bsv for c, v in cb.items()}

        # Sample-level similarity
        ss = {}
        for c, v in cb.items():
            e = exp.get(c)
            if not e or not e.bsv:
                continue
            ev = np.array([e.bsv[k] for k in BSV_COMPONENTS])
            own = [_cos(v.sample_bsv[i], ev) for i in range(v.n_spectra)]
            alts = {}
            for oc, oe in exp.items():
                if oc == c or not oe.bsv:
                    continue
                ov = np.array([oe.bsv[k] for k in BSV_COMPONENTS])
                alts[oc] = [_cos(v.sample_bsv[i], ov) for i in range(v.n_spectra)]
            ss[c] = {"own": own, "alts": alts}

        # Delta comparison
        re = exp.get(ref)
        dr_res = {}
        if re and re.bsv:
            for c in ds.cohort_names:
                if c == ref:
                    continue
                e = exp.get(c)
                if e and e.bsv:
                    dr_res[c] = compute_delta_comparison(
                        cb[c].mean_bsv, cb[ref].mean_bsv, e.bsv, re.bsv)

        cross = compute_cross_matrix_normalized(means, exp, "raw")
        interp = generate_interpretation(cross["alignment_summary"], dr_res, sub)

        pca_obj = PCA(n_components=2)
        pp = pca_obj.fit_transform(bm)

        # Build expected-comparator trust graphs (literature-side only)
        # Each cohort's expected comparator gets its own trust graph
        tq_retriever = get_text_retriever()
        expected_trust_graphs = {}
        for coh in ds.cohort_names:
            e = exp.get(coh)
            if not e or not e.bsv:
                continue
            # Build a literature trust graph for this comparator's condition
            comp_name = e.comparator_name  # e.g. "HCC", "healthy_control"
            # Retrieve evidence for this specific literature comparator
            lookup_query = f"{comp_name.replace('_', ' ')} biochemical profile serum"
            lit_items = tq_retriever.retrieve(lookup_query, top_k=6)
            lit_mtb = tq_mtb_map(lit_items)
            lit_bsv = build_literature_bsv_profile(comp_name.replace("_", " "), lit_items)
            traversals = build_per_condition_traversals(
                lookup_query, lit_items, lit_mtb, lit_bsv,
                [comp_name] if comp_name in lit_bsv.get("conditions", []) else [],
                retriever=tq_retriever,
            )
            if traversals:
                expected_trust_graphs[coh] = {
                    "comparator": comp_name,
                    "graph": traversals[0]["graph"],
                    "n_evidence": traversals[0]["n_evidence"],
                }
            elif lit_items:
                # Fallback: build a generic trust graph
                from gaira.retrieval.trust_graph_builder import build_trust_graph as _btg
                expected_trust_graphs[coh] = {
                    "comparator": comp_name,
                    "graph": _btg(lookup_query, lit_items, lit_mtb, lit_bsv),
                    "n_evidence": len(lit_items),
                }

        # Optional Gemini interpretation
        gemini_interp = None
        if use_gemini:
            try:
                from gaira.llm.gemini_client import generate_text
                prompt_parts = [
                    "You are GAIRA, a Raman/SERS biochemical interpretation engine.",
                    "Write a concise scientific discussion (3-5 sentences) of these spectral query results.",
                    f"Dataset: {sel.display_name} ({sel.n_spectra} spectra, substrate: {sub.get('substrate', '?')})",
                    f"Reference cohort: {_dn(ref)}",
                ]
                for c, v in cb.items():
                    top = sorted(v.mean_bsv.items(), key=lambda x: -x[1])[:3]
                    prompt_parts.append(f"Observed BSV {_dn(c)}: top = {top}")
                for a in cross.get("alignment_summary", []):
                    prompt_parts.append(
                        f"Alignment {_dn(a['cohort'])}: margin={a['margin']:+.3f}")
                for c, d in dr_res.items():
                    prompt_parts.append(f"Delta cosine {_dn(c)}: {d['delta_cosine']:+.3f}")
                prompt_parts.append("Be cautious. Focus on BSV axes and shift agreement.")

                with st.spinner("Querying Gemini..."):
                    result = generate_text("\n".join(prompt_parts))
                    gemini_interp = result.text
            except Exception as e:
                gemini_interp = f"Gemini unavailable: {e}"

    S.update({
        "sq4_dsid": dsid, "sq4_ds": ds, "sq4_Xn": Xn, "sq4_prep": prep,
        "sq4_wf": wf, "sq4_bm": bm, "sq4_cb": cb, "sq4_dt": dt, "sq4_ref": ref,
        "sq4_drv": drv, "sq4_ann": ann, "sq4_exp": exp, "sq4_sub": sub,
        "sq4_ss": ss, "sq4_dr": dr_res, "sq4_cross": cross,
        "sq4_interp": interp, "sq4_gemini": gemini_interp,
        "sq4_pp": pp, "sq4_pca": pca_obj,
        "sq4_expected_graphs": expected_trust_graphs,
    })

if "sq4_ds" not in S:
    st.stop()

# ── Render ─────────────────────────────────────────────────────────────

ds = S["sq4_ds"]; Xn = S["sq4_Xn"]; prep = S["sq4_prep"]
cb = S["sq4_cb"]; dt = S["sq4_dt"]; ref = S["sq4_ref"]
ann_d = S["sq4_ann"]; exp = S["sq4_exp"]; sub = S["sq4_sub"]
ss = S["sq4_ss"]; dr_res = S["sq4_dr"]; cross = S["sq4_cross"]
interp = S["sq4_interp"]; gemini_interp = S.get("sq4_gemini")
pp = S["sq4_pp"]; pca_obj = S["sq4_pca"]
expected_graphs = S.get("sq4_expected_graphs", {})
cats = [BSV_SHORT.get(c, c) for c in BSV_COMPONENTS]

# ══════════════════════════════════════════════════════════════
# SECTION 1 — MEASURED SPECTRAL STRUCTURE
# ══════════════════════════════════════════════════════════════

st.divider()
st.header("1 · Measured Spectral Structure")
st.caption("Direct: spectra → windows → BSV. No literature influence.")

c1, c2, c3 = st.columns(3)
c1.metric("Spectra", ds.n_spectra)
c2.metric("Cohorts", len(ds.cohort_names))
c3.metric("Substrate", sub.get("substrate", "?"))

with st.expander("Preprocessing"):
    st.markdown(f"**Baseline:** {prep.baseline}  \n**Smoothing:** {prep.smoothing}  \n"
                f"**Normalization:** {prep.normalization}")

st.plotly_chart(mean_spectra_plot(Xn, ds.wavenumbers, ds.cohorts),
                use_container_width=True, config={"displayModeBar": False})

st.markdown("##### Observed spectral BSV")
st.plotly_chart(radar_plot(cb), use_container_width=True, config={"displayModeBar": False})

ca, cc = st.columns(2)
with ca:
    st.plotly_chart(bsv_heatmap(cb, height=250),
                    use_container_width=True, config={"displayModeBar": False})
with cc:
    if dt:
        st.plotly_chart(delta_heatmap(dt, ref, height=220),
                        use_container_width=True, config={"displayModeBar": False})

# PCA
fig_pca = go.Figure()
for j, coh in enumerate(sorted(set(ds.cohorts))):
    m = ds.cohorts == coh
    fig_pca.add_trace(go.Scatter(
        x=pp[m, 0], y=pp[m, 1], mode="markers",
        marker=dict(color=_color_for(j), size=5, opacity=0.5), name=_dn(coh),
    ))
fig_pca.update_layout(
    xaxis_title=f"PC1 ({pca_obj.explained_variance_ratio_[0]*100:.1f}%)",
    yaxis_title=f"PC2 ({pca_obj.explained_variance_ratio_[1]*100:.1f}%)",
    paper_bgcolor=BG, plot_bgcolor=BG, height=280,
    margin=dict(l=50, r=20, t=10, b=40),
    font=dict(color="rgba(255,255,255,0.8)"),
    legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
)
st.plotly_chart(fig_pca, use_container_width=True, config={"displayModeBar": False})

# ══════════════════════════════════════════════════════════════
# SECTION 2 — SPECTRAL BAND DRIVERS
# ══════════════════════════════════════════════════════════════

if ann_d:
    st.divider()
    st.header("2 · Spectral Band Drivers")
    st.caption(f"Top windows driving observed BSV differences vs {_dn(ref)}.")

    for coh, wins in ann_d.items():
        top = wins[:8]
        fig_bd = go.Figure(go.Bar(
            x=[f"{w['wavenumber_start']}-{w['wavenumber_end']}" for w in top],
            y=[w["effect_size"] for w in top],
            marker_color=["#2ECC71" if w["effect_size"] > 0 else "#E74C3C" for w in top],
            text=[BSV_SHORT.get(w["bsv_component"], "?")[:6] for w in top],
            textposition="outside", textfont=dict(size=7),
        ))
        fig_bd.update_layout(
            title=dict(text=f"{_dn(coh)} vs {_dn(ref)}",
                       font=dict(size=10, color="rgba(255,255,255,0.7)")),
            paper_bgcolor=BG, plot_bgcolor=BG, height=220,
            margin=dict(l=40, r=20, t=30, b=40),
            font=dict(color="rgba(255,255,255,0.8)", size=9),
            xaxis=dict(tickangle=-45),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)",
                       zeroline=True, zerolinecolor="rgba(255,255,255,0.2)"),
        )
        st.plotly_chart(fig_bd, use_container_width=True, config={"displayModeBar": False})

        with st.expander(f"Band annotations ({_dn(coh)})"):
            st.table([{
                "Window": f"{w['wavenumber_start']}-{w['wavenumber_end']}",
                "BSV": BSV_SHORT.get(w["bsv_component"], "?"),
                "Effect": f"{w['effect_size']:+.3f}",
                "Motif(s)": " / ".join(a["motif"] for a in w.get("annotations", [])) or "—",
            } for w in top])

# ══════════════════════════════════════════════════════════════
# SECTION 3 — EXPECTED LITERATURE COMPARATOR
# ══════════════════════════════════════════════════════════════

st.divider()
st.header("3 · Expected Literature Comparator")
st.caption("Literature-grounded reference profiles per cohort. Does not drive spectral BSV.")

# Card grid
comparator_rows = []
for coh in ds.cohort_names:
    e = exp.get(coh)
    if not e:
        continue
    compat_badge = {"favorable": "🟢", "mixed": "🟡", "uncertain": "⚪"}.get(
        sub.get("compatibility"), "⚪")
    comparator_rows.append({
        "Cohort": _dn(coh),
        "Expected Comparator": e.comparator_name.replace("_", " "),
        "Match": e.match_type,
        "Confidence": e.confidence,
        "Substrate": f"{compat_badge} {sub.get('compatibility', '?')}",
        "Rationale": e.explanation[:100] + ("..." if len(e.explanation) > 100 else ""),
    })
st.table(comparator_rows)

# Expected literature trust graphs — one per cohort
if expected_graphs:
    st.markdown("##### Expected literature traversals")
    st.caption(
        "Each cohort's expected comparator has its own literature trust graph below — "
        "showing which evidence sources and motifs/themes support that expected profile."
    )

    for coh in ds.cohort_names:
        if coh not in expected_graphs:
            continue
        eg = expected_graphs[coh]
        comparator = eg["comparator"].replace("_", " ")
        n_ev = eg["n_evidence"]
        st.markdown(f"---")
        st.markdown(
            f"##### Expected literature traversal — **{_dn(coh)}** "
            f"(comparator: {comparator}, {n_ev} evidence items)"
        )
        fig_etg = render_trust_graph(eg["graph"])
        st.plotly_chart(fig_etg, use_container_width=True,
                         config={"displayModeBar": False})

# ══════════════════════════════════════════════════════════════
# SECTION 4 — OBSERVED VS EXPECTED COMPARISON
# ══════════════════════════════════════════════════════════════

st.divider()
st.header("4 · Observed vs Expected Comparison")
st.caption(
    "How to read: **raw cosine** = profile similarity. "
    "**margin** = own expected vs best alternative gap. "
    "**delta cosine** = disease-vs-reference direction agreement (primary metric)."
)

# PRIMARY 1: Similarity matrix
if cross["matrix"]:
    st.markdown("##### Similarity matrix")
    st.caption("Rows = observed cohort BSV. Columns = expected comparator BSV. Diagonal-dominant = discriminative.")
    mat = np.array(cross["matrix"])
    fig_cm = go.Figure(go.Heatmap(
        z=mat,
        x=[c.replace("_", " ") for c in cross["expected_labels"]],
        y=[_dn(c) for c in cross["observed_labels"]],
        colorscale="RdBu_r", zmid=0,
        texttemplate="%{z:.3f}", textfont=dict(size=12),
    ))
    fig_cm.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        height=max(200, 90 * len(cross["observed_labels"])),
        margin=dict(l=10, r=10, t=20, b=10),
        font=dict(color="rgba(255,255,255,0.8)"),
    )
    st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})

# PRIMARY 2: Alignment summary table
if cross["alignment_summary"]:
    st.markdown("##### Alignment summary")
    rows = []
    for a in cross["alignment_summary"]:
        mi = "🟢" if a["margin"] > 0.05 else "🟡" if a["margin"] > 0 else "🔴"
        dc = dr_res.get(a["cohort"], {}).get("delta_cosine")
        sd = ss.get(a["cohort"])
        sep = "—"
        if sd:
            om = float(np.mean(sd["own"]))
            bam = max((float(np.mean(v)) for v in sd["alts"].values()), default=0)
            sep = f"{om - bam:+.3f}"
        rows.append({
            "Cohort": _dn(a["cohort"]),
            "Expected": a["own_expected"].replace("_", " "),
            "Match": a["match_type"],
            "Raw cos": f"{a['own_cosine']:+.3f}",
            "Margin": f"{mi} {a['margin']:+.3f}",
            "Δ cos": f"{dc:+.3f}" if dc is not None else "—",
            "Sample sep": sep,
        })
    st.table(rows)

# PRIMARY 3: Disease-vs-reference shift comparison (the star figure)
if dr_res:
    st.markdown("##### Disease-vs-reference shift comparison")
    st.caption(
        f"**Primary validation metric.** Observed Δ(cohort − {_dn(ref)}) vs expected literature Δ. "
        "Positive delta cosine = shift direction is consistent with expected biology."
    )

    for coh, dr in dr_res.items():
        obs_d = [dr["observed_delta"].get(c, 0) for c in BSV_COMPONENTS]
        exp_d = [dr["expected_delta"].get(c, 0) for c in BSV_COMPONENTS]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=cats, y=obs_d, name="Observed Δ", marker_color="#3498DB"))
        fig.add_trace(go.Bar(x=cats, y=exp_d, name="Expected Δ", marker_color="#E74C3C", opacity=0.7))
        fig.update_layout(
            barmode="group",
            title=dict(text=f"{_dn(coh)} — delta cosine = {dr['delta_cosine']:+.3f}",
                       font=dict(size=11, color="rgba(255,255,255,0.7)")),
            paper_bgcolor=BG, plot_bgcolor=BG, height=240,
            margin=dict(l=40, r=20, t=35, b=30),
            font=dict(color="rgba(255,255,255,0.8)", size=9),
            yaxis=dict(title="Δ BSV", gridcolor="rgba(255,255,255,0.06)",
                       zeroline=True, zerolinecolor="rgba(255,255,255,0.2)"),
            legend=dict(font=dict(size=8), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        ci = {"aligned": "🟢", "partial": "🟡", "divergent": "🔴", "weak": "⚪"}
        with st.expander(f"Per-axis agreement ({_dn(coh)})"):
            st.table([{
                "Axis": BSV_SHORT.get(a["component"], a["component"]),
                "Obs Δ": f"{a['obs_delta']:+.5f}",
                "Exp Δ": f"{a['exp_delta']:+.4f}",
                "Status": f"{ci.get(a['category'], '?')} {a['category']}",
            } for a in dr["per_axis"]])

# Sample-level alignment histograms (primary validation support)
if ss:
    st.markdown("##### Sample-level alignment")
    st.caption("Per-sample cosine to own expected vs alternative expected comparator.")
    for coh, sd in ss.items():
        own = np.array(sd["own"])
        fig_sl = go.Figure()
        fig_sl.add_trace(go.Histogram(
            x=own, name=f"→ own ({exp[coh].comparator_name.replace('_', ' ')})",
            marker_color="#3498DB", opacity=0.7, nbinsx=25,
        ))
        for ac, av in sd["alts"].items():
            fig_sl.add_trace(go.Histogram(
                x=av, name=f"→ {exp[ac].comparator_name.replace('_', ' ')}",
                marker_color="#E74C3C", opacity=0.5, nbinsx=25,
            ))
        om = float(own.mean())
        bam = max((float(np.mean(v)) for v in sd["alts"].values()), default=0)
        sep = om - bam
        fig_sl.update_layout(
            barmode="overlay",
            title=dict(text=f"{_dn(coh)} — own={om:.3f}, alt={bam:.3f}, sep={sep:+.3f}",
                       font=dict(size=10, color="rgba(255,255,255,0.7)")),
            paper_bgcolor=BG, plot_bgcolor=BG, height=200,
            margin=dict(l=40, r=20, t=40, b=30),
            font=dict(color="rgba(255,255,255,0.8)", size=9),
            xaxis=dict(title="Cosine"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            legend=dict(font=dict(size=8), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_sl, use_container_width=True, config={"displayModeBar": False})

# DEMOTED: Overlay radars (moved to bottom, clearly labeled)
st.markdown("---")
with st.expander("📊 Optional: observed vs expected overlay radars (visual comparison only)"):
    st.caption(
        "⚠️ **Visual comparison only.** Observed spectral BSV and literature-expected BSV "
        "live in different numerical spaces (spectral intensity vs literature support weights). "
        "These overlays are not a primary validation figure — use the similarity matrix, "
        "alignment summary, and delta-shift comparison above instead."
    )
    for coh in ds.cohort_names:
        e = exp.get(coh)
        if not e or not e.bsv:
            continue
        obs_bsv = cb[coh].mean_bsv
        obs_vals = [obs_bsv.get(c, 0) for c in BSV_COMPONENTS]
        exp_vals = [e.bsv.get(c, 0) for c in BSV_COMPONENTS]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=obs_vals + [obs_vals[0]], theta=cats + [cats[0]],
            fill="toself", fillcolor=_hex_to_rgba("#3498DB", 0.10),
            line=dict(color="#3498DB", width=2), name="Observed (spectral)",
        ))
        fig.add_trace(go.Scatterpolar(
            r=exp_vals + [exp_vals[0]], theta=cats + [cats[0]],
            fill="toself", fillcolor=_hex_to_rgba("#E74C3C", 0.10),
            line=dict(color="#E74C3C", width=2, dash="dash"),
            name=f"Expected ({e.comparator_name.replace('_', ' ')})",
        ))
        fig.update_layout(
            title=dict(text=f"{_dn(coh)} — raw overlay",
                       font=dict(size=11, color="rgba(255,255,255,0.6)")),
            polar=dict(bgcolor=BG,
                       radialaxis=dict(visible=True, gridcolor="rgba(255,255,255,0.08)"),
                       angularaxis=dict(gridcolor="rgba(255,255,255,0.1)",
                                        tickfont=dict(size=10, color="rgba(255,255,255,0.7)"))),
            paper_bgcolor=BG, font=dict(color="rgba(255,255,255,0.8)"),
            legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=60, r=60, t=40, b=30), height=260,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ══════════════════════════════════════════════════════════════
# INTERPRETATION
# ══════════════════════════════════════════════════════════════

if interp:
    st.divider()
    st.subheader("Interpretation")
    st.markdown(interp)

if gemini_interp:
    st.divider()
    st.subheader("Gemini Interpretation")
    st.caption("Generated by Gemini LLM based on the analysis results above.")
    st.markdown(gemini_interp)

st.caption(
    "Observed spectral BSV = direct projection. "
    "Expected literature BSV = post-hoc comparator. "
    "Primary validation metric is disease-vs-reference delta cosine."
)
