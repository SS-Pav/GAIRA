"""GAIRA Spectral Query v2.5 — Best-Approach Build.

Architecture:
  Section 1: Measured spectral structure (PRIMARY)
  Section 2: Spectral band drivers (EXPLANATION)
  Section 3: Expected literature comparator (SECONDARY)
  Section 4: Observed vs expected validation (VALIDATION)

Direct spectral BSV is primary. Motifs annotate windows only.
Expected BSV is post-hoc. Delta-shift agreement is the main comparison metric.

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src streamlit run streamlit_apps/gaira_lfm_v2_5_spectral_query.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

from gaira.spectral.dataset_registry import discover_datasets
from gaira.spectral.dataset_loader import load_dataset
from gaira.spectral.preprocessing import preprocess
from gaira.spectral.window_panel import extract_window_features, BSV_COMPONENTS, WINDOW_DEFS
from gaira.spectral.bsv_projection import project_to_bsv, compute_cohort_bsvs, compute_deltas
from gaira.spectral.plots import (
    radar_plot, bsv_heatmap, delta_heatmap, mean_spectra_plot,
    BSV_SHORT, BG, _hex_to_rgba, _color_for,
)
from gaira.spectral.band_drivers import compute_per_cohort_window_importance
from gaira.spectral.explanation import annotate_windows, THEME_DISPLAY
from gaira.spectral.expected_bsv import (
    build_expected_comparators, get_cohort_display_name,
)
from gaira.spectral.comparison import (
    compute_delta_comparison, compute_cross_matrix_normalized,
    get_substrate_context, generate_interpretation,
)

import warnings; warnings.filterwarnings("ignore", category=FutureWarning)


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


# ── App ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="GAIRA v2.5 Spectral Query", layout="wide")
st.title("GAIRA Spectral Query v2.5")
st.caption("Direct spectral BSV → band drivers → literature validation.")

datasets = discover_datasets()
if not datasets:
    st.error("No target datasets found."); st.stop()

S = st.session_state  # shorthand


def _dn(coh):
    return get_cohort_display_name(S.get("dsid", ""), coh)


# ── Sidebar ────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Target Dataset")
    dm = {d.display_name: d for d in datasets}
    sel = dm[st.selectbox("Dataset:", list(dm.keys()))]
    st.caption(f"`{sel.dataset_id}` · {sel.n_spectra} spectra · {sel.family}")
    if sel.note:
        st.caption(sel.note)
    for c, n in sorted(sel.cohorts.items()):
        st.markdown(f"- {get_cohort_display_name(sel.dataset_id, c)} (n={n})")

    st.divider()
    cl = sorted(sel.cohorts.keys())
    dr = "healthy_control" if "healthy_control" in cl else cl[0]
    ref = st.selectbox("Reference:", cl, index=cl.index(dr))
    st.divider()
    run = st.button("Run", type="primary")

# ── Pipeline ───────────────────────────────────────────────────────────

if run:
    dsid = sel.dataset_id
    with st.spinner("Processing..."):
        ds = load_dataset(dsid)
        X_norm, prep = preprocess(ds)
        wf = extract_window_features(X_norm, ds.wavenumbers)
        bm = project_to_bsv(wf)
        cb = compute_cohort_bsvs(bm, ds.cohorts)
        dt = compute_deltas(cb, ref)

        # Band drivers + annotations
        drv = compute_per_cohort_window_importance(wf, ds.cohorts, ref)
        ann = {c: annotate_windows(w) for c, w in drv.items()}

        # Expected
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
        pca = PCA(n_components=2)
        pp = pca.fit_transform(bm)

    S.update({
        "dsid": dsid, "ds": ds, "Xn": X_norm, "prep": prep,
        "wf": wf, "bm": bm, "cb": cb, "dt": dt, "ref": ref,
        "drv": drv, "ann": ann, "exp": exp, "sub": sub,
        "ss": ss, "dr": dr_res, "cross": cross, "interp": interp,
        "pp": pp, "pca": pca,
    })

# ── Rendering ──────────────────────────────────────────────────────────

if "ds" not in S:
    st.stop()

ds = S["ds"]; Xn = S["Xn"]; prep = S["prep"]
wf = S["wf"]; bm = S["bm"]; cb = S["cb"]; dt = S["dt"]; ref = S["ref"]
drv = S["drv"]; ann_d = S["ann"]; exp = S["exp"]; sub = S["sub"]
ss = S["ss"]; dr_res = S["dr"]; cross = S["cross"]; interp = S["interp"]
pp = S["pp"]; pca_obj = S["pca"]
cats = [BSV_SHORT.get(c, c) for c in BSV_COMPONENTS]

# ══════════════════════════════════════════════════════════════
# SECTION 1 — MEASURED SPECTRAL STRUCTURE (PRIMARY)
# ══════════════════════════════════════════════════════════════

st.divider()
st.header("1 · Measured Spectral Structure")
st.caption("Direct: preprocessed spectra → spectral windows → BSV. No literature influence.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Spectra", ds.n_spectra)
c2.metric("Cohorts", len(ds.cohort_names))
c3.metric("Substrate", sub.get("substrate", "?"))
c4.metric("Pipeline", prep.pipeline)

with st.expander("Preprocessing"):
    st.markdown(f"- **Baseline:** {prep.baseline}")
    st.markdown(f"- **Smoothing:** {prep.smoothing}")
    st.markdown(f"- **Normalization:** {prep.normalization}")
    st.caption(prep.notes)

# Mean spectra
st.plotly_chart(mean_spectra_plot(Xn, ds.wavenumbers, ds.cohorts),
                use_container_width=True, config={"displayModeBar": False})

# BSV radar + heatmaps
st.subheader("Observed BSV Composition")
st.plotly_chart(radar_plot(cb), use_container_width=True, config={"displayModeBar": False})

ca, cc = st.columns(2)
with ca:
    st.plotly_chart(bsv_heatmap(cb, height=250),
                    use_container_width=True, config={"displayModeBar": False})
with cc:
    if dt:
        st.plotly_chart(delta_heatmap(dt, ref, height=220),
                        use_container_width=True, config={"displayModeBar": False})

# Pairwise delta heatmap (for multi-cohort)
if len(ds.cohort_names) > 2:
    st.subheader("Pairwise Cohort Deltas")
    pairs = []
    pair_labels = []
    for i, c1n in enumerate(sorted(ds.cohort_names)):
        for c2n in sorted(ds.cohort_names)[i+1:]:
            d = {comp: round(cb[c1n].mean_bsv[comp] - cb[c2n].mean_bsv[comp], 6)
                 for comp in BSV_COMPONENTS}
            pairs.append([d[c] for c in BSV_COMPONENTS])
            pair_labels.append(f"{_dn(c1n)} − {_dn(c2n)}")
    if pairs:
        pw = np.array(pairs)
        vm = max(abs(pw.min()), abs(pw.max())) * 1.1 or 0.01
        fig_pw = go.Figure(go.Heatmap(
            z=pw, x=cats, y=pair_labels, colorscale="RdBu_r", zmid=0,
            zmin=-vm, zmax=vm, texttemplate="%{z:.4f}", textfont=dict(size=10),
        ))
        fig_pw.update_layout(paper_bgcolor=BG, plot_bgcolor=BG,
                             height=max(180, 50*len(pairs)),
                             margin=dict(l=10, r=10, t=10, b=10),
                             font=dict(color="rgba(255,255,255,0.8)"))
        st.plotly_chart(fig_pw, use_container_width=True, config={"displayModeBar": False})

# Sample distributions
st.subheader("Sample-Level BSV")
# Pick top 4 axes by max delta magnitude
if dt:
    axis_ranks = sorted(range(8), key=lambda i: max(
        abs(dt.get(c, {}).get(BSV_COMPONENTS[i], 0)) for c in dt
    ), reverse=True)[:4]
else:
    axis_ranks = list(range(4))

fig_box = go.Figure()
for ai, ci in enumerate(axis_ranks):
    comp = BSV_COMPONENTS[ci]
    for j, coh in enumerate(sorted(set(ds.cohorts))):
        fig_box.add_trace(go.Box(
            y=cb[coh].sample_bsv[:, ci], name=f"{BSV_SHORT[comp]} · {_dn(coh)}",
            marker_color=_color_for(j), boxmean=True,
            visible=True if ai == 0 else "legendonly",
        ))
fig_box.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=280,
                       margin=dict(l=40, r=20, t=10, b=30),
                       font=dict(color="rgba(255,255,255,0.8)", size=9),
                       yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                       legend=dict(font=dict(size=8), bgcolor="rgba(0,0,0,0)"))
st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})

# PCA
st.subheader("PCA in BSV Space")
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
    paper_bgcolor=BG, plot_bgcolor=BG, height=320,
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
    st.caption(f"Which windows drive BSV differences vs {_dn(ref)}. "
               "Motif annotations are interpretive — they did not compute BSV.")

    for coh, wins in ann_d.items():
        top = wins[:10]
        st.markdown(f"##### {_dn(coh)} vs {_dn(ref)}")

        # Bar chart
        fig_bd = go.Figure(go.Bar(
            x=[f"{w['wavenumber_start']}-{w['wavenumber_end']}" for w in top],
            y=[w["effect_size"] for w in top],
            marker_color=["#2ECC71" if w["effect_size"] > 0 else "#E74C3C" for w in top],
            text=[BSV_SHORT.get(w["bsv_component"], "?")[:6] for w in top],
            textposition="outside", textfont=dict(size=7),
        ))
        fig_bd.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG, height=220,
            margin=dict(l=40, r=20, t=10, b=40),
            font=dict(color="rgba(255,255,255,0.8)", size=9),
            xaxis=dict(tickangle=-45, title="Window (cm⁻¹)"),
            yaxis=dict(title="Effect size", gridcolor="rgba(255,255,255,0.06)",
                       zeroline=True, zerolinecolor="rgba(255,255,255,0.2)"),
        )
        st.plotly_chart(fig_bd, use_container_width=True, config={"displayModeBar": False})

        # Annotation table
        with st.expander(f"Band annotations ({_dn(coh)})"):
            rows = []
            for w in top[:8]:
                a = w.get("annotations", [])
                motifs = " / ".join(x["motif"] for x in a) if a else "unresolved"
                themes = " / ".join(THEME_DISPLAY.get(x["theme"], x["theme"]) for x in a) if a else "—"
                amb = "⚠️" if w.get("multi_assignment") else ""
                rows.append({
                    "Window": f"{w['wavenumber_start']}-{w['wavenumber_end']}",
                    "BSV": BSV_SHORT.get(w["bsv_component"], "?"),
                    "Effect": f"{w['effect_size']:+.3f}",
                    "Dir": w["direction"],
                    "Motif(s)": f"{amb}{motifs}",
                    "Theme(s)": themes,
                })
            st.table(rows)

# ══════════════════════════════════════════════════════════════
# SECTION 3 — EXPECTED LITERATURE COMPARATOR
# ══════════════════════════════════════════════════════════════

st.divider()
st.header("3 · Expected Literature Comparator")
st.caption("Literature-grounded. Does not influence observed spectral BSV.")

for coh in ds.cohort_names:
    e = exp.get(coh)
    if not e:
        continue
    sc = {"direct": "green", "approximate": "orange", "unavailable": "red"}
    ci = {"favorable": "🟢", "mixed": "🟡", "uncertain": "⚪"}.get(sub.get("compatibility"), "⚪")
    st.markdown(
        f"**{_dn(coh)}** → `{e.comparator_name.replace('_', ' ')}` "
        f":{sc.get(e.match_type, 'gray')}[{e.match_type}] · {ci}"
    )
    st.caption(e.explanation)

# ══════════════════════════════════════════════════════════════
# SECTION 4 — OBSERVED VS EXPECTED VALIDATION
# ══════════════════════════════════════════════════════════════

st.divider()
st.header("4 · Observed vs Expected Validation")
st.caption("Does the measured spectral structure track expected literature biology?")

# Delta-shift comparison (PRIMARY validation metric)
if dr_res:
    st.subheader("Disease-vs-Reference Shift Agreement")
    st.caption(f"Observed Δ(cohort − {_dn(ref)}) vs expected literature Δ. "
               "Delta cosine measures directional agreement.")

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
            paper_bgcolor=BG, plot_bgcolor=BG, height=230,
            margin=dict(l=40, r=20, t=40, b=30),
            font=dict(color="rgba(255,255,255,0.8)", size=9),
            yaxis=dict(title="Δ BSV", gridcolor="rgba(255,255,255,0.06)",
                       zeroline=True, zerolinecolor="rgba(255,255,255,0.2)"),
            legend=dict(font=dict(size=8), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        ci = {"aligned": "🟢", "partial": "🟡", "divergent": "🔴", "weak": "⚪"}
        with st.expander(f"Per-axis ({_dn(coh)})"):
            st.table([{
                "Axis": BSV_SHORT.get(a["component"], a["component"]),
                "Obs Δ": f"{a['obs_delta']:+.5f}",
                "Exp Δ": f"{a['exp_delta']:+.4f}",
                "": f"{ci.get(a['category'], '?')} {a['category']}",
            } for a in dr["per_axis"]])

# Cross-similarity matrix
if cross["matrix"]:
    st.subheader("Similarity Matrix")
    st.caption("Raw cosine. Rows = observed. Columns = expected.")
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

# Sample-level similarity
if ss:
    st.subheader("Sample-Level Alignment")
    st.caption("Per-sample cosine to own expected vs alternative expected.")

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
            xaxis=dict(title="Cosine"), yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            legend=dict(font=dict(size=8), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_sl, use_container_width=True, config={"displayModeBar": False})

# Alignment summary table
if cross["alignment_summary"]:
    st.subheader("Alignment Summary")
    rows = []
    for a in cross["alignment_summary"]:
        mi = "🟢" if a["margin"] > 0.05 else "🟡" if a["margin"] > 0 else "🔴"
        dc = dr_res.get(a["cohort"], {}).get("delta_cosine")
        sd = ss.get(a["cohort"])
        if sd:
            om = float(np.mean(sd["own"]))
            bam = max((float(np.mean(v)) for v in sd["alts"].values()), default=0)
            sep_str = f"{om - bam:+.3f}"
        else:
            sep_str = "—"
        rows.append({
            "Cohort": _dn(a["cohort"]),
            "Expected": a["own_expected"].replace("_", " "),
            "Match": a["match_type"],
            "Cos": f"{a['own_cosine']:+.3f}",
            "Alt": f"{a['best_alt_cosine']:+.3f}" if a["best_alt_expected"] else "—",
            "Margin": f"{mi} {a['margin']:+.3f}",
            "Δ Cos": f"{dc:+.3f}" if dc is not None else "—",
            "Sample Sep": sep_str,
        })
    st.table(rows)

# ══════════════════════════════════════════════════════════════
# INTERPRETATION
# ══════════════════════════════════════════════════════════════

if interp:
    st.divider()
    st.subheader("Interpretation")
    st.markdown(interp)
    st.caption(
        "Observed BSV = direct spectral projection. "
        "Positive margin = cohort-level preferential alignment. "
        "Positive delta cosine = disease shift direction consistent with expected."
    )
