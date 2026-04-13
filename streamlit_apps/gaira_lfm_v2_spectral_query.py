"""GAIRA_LFM_v2.2 — Spectral Query Demo.

Dataset-grounded biochemical composition + literature-expected alignment.
Observed spectral BSV is computed from measured spectra only.
Expected comparators are applied post-hoc for discriminative comparison.

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src streamlit run streamlit_apps/gaira_lfm_v2_spectral_query.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from gaira.spectral.dataset_registry import discover_datasets
from gaira.spectral.dataset_loader import load_dataset
from gaira.spectral.preprocessing import preprocess
from gaira.spectral.window_panel import extract_window_features, BSV_COMPONENTS
from gaira.spectral.bsv_projection import project_to_bsv, compute_cohort_bsvs, compute_deltas
from gaira.spectral.plots import (
    radar_plot, bsv_heatmap, delta_heatmap, mean_spectra_plot,
    BSV_SHORT, BG, _hex_to_rgba,
)
from gaira.spectral.trust_graph import build_cohort_graph, render_cohort_graph
from gaira.spectral.expected_bsv import (
    build_expected_comparators, compute_observed_vs_expected,
    compute_cross_similarity_matrix, generate_interpretation,
    get_cohort_display_name, COHORT_DISPLAY_NAMES,
)


# ── App ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="GAIRA_LFM_v2.2 Spectral Query", layout="wide")

st.title("GAIRA_LFM_v2.2 — Spectral Query")
st.caption(
    "Dataset-grounded biochemical composition + literature-expected alignment analysis. "
    "Spectral BSV from measured spectra. Expected BSV applied post-hoc."
)
st.info(
    "**Spectral query** projects measured spectra into BSV space, then compares "
    "each cohort's observed profile against its expected literature counterpart. "
    "No disease priors in the spectral projection.",
    icon="🔬",
)

datasets = discover_datasets()
if not datasets:
    st.error("No target datasets found. Check data paths.")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Dataset")
    ds_map = {d.display_name: d for d in datasets}
    sel_name = st.selectbox("Select target dataset:", list(ds_map.keys()))
    sel = ds_map[sel_name]

    st.markdown(f"**ID:** `{sel.dataset_id}`")
    st.markdown(f"**Spectra:** {sel.n_spectra} · **Family:** {sel.family}")
    if sel.note:
        st.caption(sel.note)

    st.markdown("**Cohorts:**")
    for coh, n in sorted(sel.cohorts.items()):
        disp = get_cohort_display_name(sel.dataset_id, coh)
        label = f"{disp} (`{coh}`, n={n})" if disp != coh.replace("_", " ") else f"{disp} (n={n})"
        st.markdown(f"- {label}")

    if sel.dataset_id in COHORT_DISPLAY_NAMES:
        st.caption("Cohort labels corrected for semantic clarity.")

    st.divider()
    cohort_list = sorted(sel.cohorts.keys())
    default_ref = "healthy_control" if "healthy_control" in cohort_list else cohort_list[0]
    reference = st.selectbox("Reference cohort:", cohort_list,
                             index=cohort_list.index(default_ref))

    st.divider()
    run = st.button("Run Spectral Query", type="primary")


# ── Pipeline ───────────────────────────────────────────────────────────

if run:
    dsid = sel.dataset_id

    with st.spinner("Loading & analyzing..."):
        ds = load_dataset(dsid)
        X_norm, prep = preprocess(ds)
        win_feats = extract_window_features(X_norm, ds.wavenumbers)
        bsv_matrix = project_to_bsv(win_feats)
        cohort_bsvs = compute_cohort_bsvs(bsv_matrix, ds.cohorts)
        deltas = compute_deltas(cohort_bsvs, reference)

        expected = build_expected_comparators(dsid, ds.cohort_names)
        obs_vs_exp = {}
        for coh, cbsv in cohort_bsvs.items():
            if coh in expected:
                r = compute_observed_vs_expected(cbsv.mean_bsv, expected[coh])
                if r:
                    obs_vs_exp[coh] = r

        cohort_mean_bsvs = {c: cb.mean_bsv for c, cb in cohort_bsvs.items()}
        cross_matrix = compute_cross_similarity_matrix(cohort_mean_bsvs, expected)
        interp = generate_interpretation(cross_matrix["alignment_summary"], dsid)

    st.session_state.update({
        "v2_ds": ds, "v2_dsid": dsid, "v2_X_norm": X_norm, "v2_prep": prep,
        "v2_win_feats": win_feats, "v2_bsv_matrix": bsv_matrix,
        "v2_cohort_bsvs": cohort_bsvs, "v2_deltas": deltas,
        "v2_reference": reference, "v2_expected": expected,
        "v2_obs_vs_exp": obs_vs_exp, "v2_cross": cross_matrix,
        "v2_interp": interp,
    })


# ── Rendering ──────────────────────────────────────────────────────────

def _dn(coh: str) -> str:
    return get_cohort_display_name(st.session_state["v2_dsid"], coh)


def _render():
    ds = st.session_state["v2_ds"]
    dsid = st.session_state["v2_dsid"]
    X_norm = st.session_state["v2_X_norm"]
    prep = st.session_state["v2_prep"]
    win_feats = st.session_state["v2_win_feats"]
    cohort_bsvs = st.session_state["v2_cohort_bsvs"]
    deltas = st.session_state["v2_deltas"]
    ref = st.session_state["v2_reference"]
    expected = st.session_state["v2_expected"]
    obs_vs_exp = st.session_state["v2_obs_vs_exp"]
    cross = st.session_state["v2_cross"]
    interp = st.session_state["v2_interp"]

    # ── 1. Dataset summary ─────────────────────────────────────
    st.divider()
    st.subheader("Dataset Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Spectra", ds.n_spectra)
    c2.metric("Cohorts", len(ds.cohort_names))
    c3.metric("Range", f"{ds.wavenumbers.min():.0f}–{ds.wavenumbers.max():.0f} cm⁻¹")

    with st.expander("Preprocessing", expanded=False):
        st.markdown(f"- **Normalization:** {prep.normalization}")
        st.markdown(f"- **Baseline:** {prep.baseline}")
        st.caption(prep.notes)

    # ── 2. Mean spectra ────────────────────────────────────────
    st.divider()
    st.subheader("Mean Spectra by Cohort")
    st.plotly_chart(mean_spectra_plot(X_norm, ds.wavenumbers, ds.cohorts),
                    use_container_width=True, config={"displayModeBar": False})

    # ── 3. Spectral BSV ────────────────────────────────────────
    st.divider()
    st.subheader("Spectral BSV Composition")
    st.caption("Derived from measured spectra only. No literature influence.")
    st.plotly_chart(radar_plot(cohort_bsvs),
                    use_container_width=True, config={"displayModeBar": False})

    ca, cb = st.columns(2)
    with ca:
        st.markdown("##### BSV Values")
        st.plotly_chart(bsv_heatmap(cohort_bsvs),
                        use_container_width=True, config={"displayModeBar": False})
    with cb:
        st.markdown(f"##### Delta vs {_dn(ref)}")
        if deltas:
            st.plotly_chart(delta_heatmap(deltas, ref),
                            use_container_width=True, config={"displayModeBar": False})

    # ── 4. Cohort traversals ───────────────────────────────────
    st.divider()
    st.subheader("Cohort Traversals")
    st.caption("Cohort → Windows → Motifs → Themes → BSV. Spectral derivation only.")

    for coh, cbsv in cohort_bsvs.items():
        mask = ds.cohorts == coh
        g = build_cohort_graph(coh, cbsv, win_feats, mask)
        st.markdown("---")
        st.markdown(
            f"##### {_dn(coh)}  "
            f"<small>({cbsv.n_spectra} spectra · {g['n_motifs']} motifs · "
            f"{g['n_themes']} themes · {g['n_bsv']} BSV)</small>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(render_cohort_graph(g, height=260),
                        use_container_width=True, config={"displayModeBar": False})

    # ── 5. Expected comparators ────────────────────────────────
    st.divider()
    st.subheader("Expected Literature Comparators")
    st.caption("Post-hoc comparison layer. Does NOT influence spectral BSV above.")

    for coh in ds.cohort_names:
        exp = expected.get(coh)
        if not exp:
            continue
        sc = {"direct": "green", "approximate": "orange", "unavailable": "red"}
        st.markdown(
            f"**{_dn(coh)}** → `{exp.comparator_name.replace('_', ' ')}` "
            f":{sc.get(exp.match_type, 'gray')}[{exp.match_type}] "
            f"(confidence: {exp.confidence})"
        )
        st.caption(exp.explanation)

    # ── 6. Observed vs Expected per cohort ─────────────────────
    if obs_vs_exp:
        st.divider()
        st.subheader("Observed vs Expected")

        for coh, result in obs_vs_exp.items():
            exp = expected[coh]
            cats = [BSV_SHORT.get(c, c) for c in BSV_COMPONENTS]

            fig = go.Figure()
            obs_v = [result["observed"].get(c, 0) for c in BSV_COMPONENTS] + \
                    [result["observed"].get(BSV_COMPONENTS[0], 0)]
            exp_v = [result["expected"].get(c, 0) for c in BSV_COMPONENTS] + \
                    [result["expected"].get(BSV_COMPONENTS[0], 0)]

            fig.add_trace(go.Scatterpolar(
                r=obs_v, theta=cats + [cats[0]],
                fill="toself", fillcolor=_hex_to_rgba("#3498DB", 0.10),
                line=dict(color="#3498DB", width=2), name="Observed",
            ))
            fig.add_trace(go.Scatterpolar(
                r=exp_v, theta=cats + [cats[0]],
                fill="toself", fillcolor=_hex_to_rgba("#E74C3C", 0.10),
                line=dict(color="#E74C3C", width=2, dash="dash"),
                name=f"Expected ({exp.comparator_name.replace('_', ' ')})",
            ))
            fig.update_layout(
                title=dict(text=f"{_dn(coh)} — cosine = {result['cosine']:.3f}",
                           font=dict(size=13, color="rgba(255,255,255,0.8)")),
                polar=dict(bgcolor=BG,
                           radialaxis=dict(visible=True, gridcolor="rgba(255,255,255,0.08)",
                                           tickfont=dict(size=8, color="rgba(255,255,255,0.4)")),
                           angularaxis=dict(gridcolor="rgba(255,255,255,0.1)",
                                            tickfont=dict(size=10, color="rgba(255,255,255,0.7)"))),
                paper_bgcolor=BG, font=dict(color="rgba(255,255,255,0.8)"),
                legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=60, r=60, t=50, b=30), height=300,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── 7. Cross-similarity matrix ─────────────────────────────
    if cross["matrix"]:
        st.divider()
        st.subheader("Cohort × Expected Similarity Matrix")
        st.caption(
            "Rows = observed spectral cohorts. Columns = expected literature profiles. "
            "Values = cosine similarity. Diagonal-dominant = discriminative alignment."
        )

        matrix = np.array(cross["matrix"])
        obs_labels = [_dn(c) for c in cross["observed_labels"]]
        exp_labels = [c.replace("_", " ") for c in cross["expected_labels"]]

        fig = go.Figure(go.Heatmap(
            z=matrix, x=exp_labels, y=obs_labels,
            colorscale="RdBu_r", zmid=0,
            texttemplate="%{z:.3f}", textfont=dict(size=11),
        ))
        fig.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG, height=max(200, 80 * len(obs_labels)),
            margin=dict(l=10, r=10, t=30, b=10),
            font=dict(color="rgba(255,255,255,0.8)"),
            xaxis=dict(title="Expected profile", tickfont=dict(size=10)),
            yaxis=dict(title="Observed cohort", tickfont=dict(size=10)),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── 8. Alignment summary table ─────────────────────────────
    if cross["alignment_summary"]:
        st.divider()
        st.subheader("Alignment Summary")

        rows = []
        for a in cross["alignment_summary"]:
            margin_color = "🟢" if a["margin"] > 0.05 else "🟡" if a["margin"] > 0 else "🔴"
            rows.append({
                "Cohort": _dn(a["cohort"]),
                "Own Expected": a["own_expected"].replace("_", " "),
                "Match": a["match_type"],
                "Own Cosine": f"{a['own_cosine']:.3f}",
                "Best Alt": a["best_alt_expected"].replace("_", " ") if a["best_alt_expected"] else "—",
                "Alt Cosine": f"{a['best_alt_cosine']:.3f}" if a["best_alt_expected"] else "—",
                "Margin": f"{margin_color} {a['margin']:+.3f}",
            })
        st.table(rows)

    # ── 9. Interpretation ──────────────────────────────────────
    if interp:
        st.divider()
        st.subheader("Scientific Interpretation")
        st.markdown(interp)
        st.caption(
            "This interpretation is based on cosine similarity between observed spectral BSV "
            "and literature-expected BSV profiles. Positive margin means the cohort aligns "
            "preferentially with its own expected profile over alternatives."
        )


if "v2_ds" in st.session_state:
    _render()
