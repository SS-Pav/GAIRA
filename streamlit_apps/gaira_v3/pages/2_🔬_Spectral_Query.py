"""GAIRA v3 — Spectral Query Page.

Dataset-grounded BSV composition with optional Gemini interpretation.
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

import warnings; warnings.filterwarnings("ignore", category=FutureWarning)


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


# ── Page ───────────────────────────────────────────────────────────────

st.header("🔬 Spectral Query")
st.caption("Direct spectral BSV → band drivers → literature validation.")

datasets = discover_datasets()
if not datasets:
    st.error("No target datasets found. Is the SSD mounted?")
    st.stop()

S = st.session_state


def _dn(coh):
    return get_cohort_display_name(S.get("sq_dsid", ""), coh)


# Sidebar
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
    dr = "healthy_control" if "healthy_control" in cl else cl[0]
    ref = st.selectbox("Reference:", cl, index=cl.index(dr))

    st.divider()
    use_gemini = st.toggle("Use Gemini for interpretation", value=False,
                            help="Generate a Gemini-written scientific interpretation of the results.")

    st.divider()
    run = st.button("Run Spectral Query", type="primary")

# Pipeline
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

        # Sample-level
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

        # Optional Gemini interpretation
        gemini_interp = None
        if use_gemini:
            try:
                from gaira.llm.gemini_client import generate_text
                # Build a concise prompt from the analysis results
                prompt_parts = [
                    "You are GAIRA, a Raman/SERS biochemical interpretation engine.",
                    "Write a concise scientific discussion (3-5 sentences) interpreting these spectral query results.",
                    f"Dataset: {sel.display_name} ({sel.n_spectra} spectra, substrate: {sub.get('substrate', '?')})",
                    f"Cohorts: {', '.join(f'{_dn(c)}(n={n})' for c, n in sorted(sel.cohorts.items()))}",
                    f"Reference cohort: {_dn(ref)}",
                ]
                for c, v in cb.items():
                    top = sorted(v.mean_bsv.items(), key=lambda x: -x[1])[:3]
                    prompt_parts.append(f"Observed BSV {_dn(c)}: top = {top}")
                for a in cross.get("alignment_summary", []):
                    prompt_parts.append(
                        f"Alignment {_dn(a['cohort'])}: margin={a['margin']:+.3f}, "
                        f"expected={a['own_expected']}")
                for c, d in dr_res.items():
                    prompt_parts.append(f"Delta cosine {_dn(c)}: {d['delta_cosine']:+.3f}")

                prompt_parts.append(
                    "Be concise, scientific, cautious. Do not overclaim. "
                    "Focus on which BSV axes show cohort differences and whether "
                    "the spectral shifts are consistent with expected biology."
                )

                with st.spinner("Querying Gemini..."):
                    result = generate_text("\n".join(prompt_parts))
                    gemini_interp = result.text
            except Exception as e:
                gemini_interp = f"Gemini unavailable: {e}"

    S.update({
        "sq_dsid": dsid, "sq_ds": ds, "sq_Xn": Xn, "sq_prep": prep,
        "sq_wf": wf, "sq_bm": bm, "sq_cb": cb, "sq_dt": dt, "sq_ref": ref,
        "sq_drv": drv, "sq_ann": ann, "sq_exp": exp, "sq_sub": sub,
        "sq_ss": ss, "sq_dr": dr_res, "sq_cross": cross,
        "sq_interp": interp, "sq_gemini": gemini_interp,
        "sq_pp": pp, "sq_pca": pca_obj,
    })

if "sq_ds" not in S:
    st.stop()

# ── Render ─────────────────────────────────────────────────────────────

ds = S["sq_ds"]; Xn = S["sq_Xn"]; prep = S["sq_prep"]
cb = S["sq_cb"]; dt = S["sq_dt"]; ref = S["sq_ref"]
ann_d = S["sq_ann"]; exp = S["sq_exp"]; sub = S["sq_sub"]
ss = S["sq_ss"]; dr_res = S["sq_dr"]; cross = S["sq_cross"]
interp = S["sq_interp"]; gemini_interp = S.get("sq_gemini")
pp = S["sq_pp"]; pca_obj = S["sq_pca"]
cats = [BSV_SHORT.get(c, c) for c in BSV_COMPONENTS]

# ── Section 1: Measured Structure ──────────────────────────────────────

st.divider()
st.subheader("1 · Measured Spectral Structure")
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
    paper_bgcolor=BG, plot_bgcolor=BG, height=300,
    margin=dict(l=50, r=20, t=10, b=40),
    font=dict(color="rgba(255,255,255,0.8)"),
    legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
)
st.plotly_chart(fig_pca, use_container_width=True, config={"displayModeBar": False})

# ── Section 2: Band Drivers ───────────────────────────────────────────

if ann_d:
    st.divider()
    st.subheader("2 · Spectral Band Drivers")
    st.caption(f"Top windows driving BSV differences vs {_dn(ref)}.")

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

# ── Section 3: Expected Comparator ────────────────────────────────────

st.divider()
st.subheader("3 · Expected Literature Comparator")
st.caption("Post-hoc comparison. Does not influence spectral BSV.")

for coh in ds.cohort_names:
    e = exp.get(coh)
    if not e:
        continue
    sc = {"direct": "green", "approximate": "orange", "unavailable": "red"}
    st.markdown(
        f"**{_dn(coh)}** → `{e.comparator_name.replace('_', ' ')}` "
        f":{sc.get(e.match_type, 'gray')}[{e.match_type}]"
    )
    st.caption(e.explanation)

# ── Section 4: Validation ─────────────────────────────────────────────

st.divider()
st.subheader("4 · Observed vs Expected")

if dr_res:
    for coh, dr in dr_res.items():
        obs_d = [dr["observed_delta"].get(c, 0) for c in BSV_COMPONENTS]
        exp_d = [dr["expected_delta"].get(c, 0) for c in BSV_COMPONENTS]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=cats, y=obs_d, name="Observed Δ", marker_color="#3498DB"))
        fig.add_trace(go.Bar(x=cats, y=exp_d, name="Expected Δ", marker_color="#E74C3C", opacity=0.7))
        fig.update_layout(
            barmode="group",
            title=dict(text=f"{_dn(coh)} — delta cosine = {dr['delta_cosine']:+.3f}",
                       font=dict(size=10, color="rgba(255,255,255,0.7)")),
            paper_bgcolor=BG, plot_bgcolor=BG, height=220,
            margin=dict(l=40, r=20, t=35, b=30),
            font=dict(color="rgba(255,255,255,0.8)", size=9),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)",
                       zeroline=True, zerolinecolor="rgba(255,255,255,0.2)"),
            legend=dict(font=dict(size=8), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

if cross["matrix"]:
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

# Alignment table
if cross["alignment_summary"]:
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
            "Margin": f"{mi} {a['margin']:+.3f}",
            "Δ Cos": f"{dc:+.3f}" if dc is not None else "—",
            "Sample Sep": sep,
        })
    st.table(rows)

# ── Interpretation ─────────────────────────────────────────────────────

st.divider()
st.subheader("Interpretation")
st.markdown(interp)

if gemini_interp:
    st.divider()
    st.subheader("Gemini Interpretation")
    st.caption("Generated by Gemini LLM based on the analysis results above.")
    st.markdown(gemini_interp)

st.caption("Observed BSV = direct spectral projection. Positive delta cosine = shift consistent with expected.")
