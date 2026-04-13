"""GAIRA_LFM_v2.3 — Spectral Query (Raw-Primary + Min-Max Comparator).

Raw observed spectral BSV is the primary biological output.
Expected literature BSV is a post-hoc comparator using shared min-max scaling.
Delta comparison: observed disease-vs-reference shift vs expected shift.

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src streamlit run streamlit_apps/gaira_lfm_v2_3_spectral_query.py
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
    build_expected_comparators, get_cohort_display_name, COHORT_DISPLAY_NAMES,
)
from gaira.spectral.comparison import (
    shared_minmax_scale, compute_delta_comparison,
    compute_all_mode_similarities, compute_cross_matrix_normalized,
    get_substrate_context, generate_interpretation,
)


st.set_page_config(page_title="GAIRA_LFM_v2.3", layout="wide")

st.title("GAIRA_LFM_v2.3 — Spectral Query")
st.caption(
    "Raw-primary spectral BSV + post-hoc literature comparator with shared min-max scaling."
)
st.info(
    "**Section 1** shows biochemical structure derived from measured spectra only. "
    "**Section 2** compares observed profiles to literature-expected BSV — "
    "this is a post-hoc comparison layer that does not alter the spectral analysis.",
    icon="🔬",
)

datasets = discover_datasets()
if not datasets:
    st.error("No target datasets found.")
    st.stop()


def _dn(coh): return get_cohort_display_name(st.session_state.get("v23_dsid", ""), coh)


# ── Sidebar ────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Dataset")
    ds_map = {d.display_name: d for d in datasets}
    sel_name = st.selectbox("Target dataset:", list(ds_map.keys()))
    sel = ds_map[sel_name]

    st.markdown(f"`{sel.dataset_id}` · {sel.n_spectra} spectra · {sel.family}")
    if sel.note:
        st.caption(sel.note)

    st.markdown("**Cohorts:**")
    for coh, n in sorted(sel.cohorts.items()):
        disp = get_cohort_display_name(sel.dataset_id, coh)
        st.markdown(f"- {disp} (n={n})")

    st.divider()
    coh_list = sorted(sel.cohorts.keys())
    def_ref = "healthy_control" if "healthy_control" in coh_list else coh_list[0]
    reference = st.selectbox("Reference cohort:", coh_list, index=coh_list.index(def_ref))

    st.divider()
    run = st.button("Run Spectral Query", type="primary")

# ── Pipeline ───────────────────────────────────────────────────────────

if run:
    dsid = sel.dataset_id

    with st.spinner("Analyzing..."):
        ds = load_dataset(dsid)
        X_norm, prep = preprocess(ds)
        win_feats = extract_window_features(X_norm, ds.wavenumbers)
        bsv_matrix = project_to_bsv(win_feats)
        cohort_bsvs = compute_cohort_bsvs(bsv_matrix, ds.cohorts)
        deltas = compute_deltas(cohort_bsvs, reference)

        expected = build_expected_comparators(dsid, ds.cohort_names)
        substrate = get_substrate_context(dsid)
        cohort_means = {c: cb.mean_bsv for c, cb in cohort_bsvs.items()}

        # Per-cohort similarities
        per_sims = {}
        for coh, cbsv in cohort_bsvs.items():
            exp = expected.get(coh)
            if exp and exp.bsv:
                per_sims[coh] = compute_all_mode_similarities(cbsv.mean_bsv, exp.bsv)

        # Shared min-max overlays
        minmax_pairs = {}
        for coh, cbsv in cohort_bsvs.items():
            exp = expected.get(coh)
            if exp and exp.bsv:
                obs_s, exp_s = shared_minmax_scale(cbsv.mean_bsv, exp.bsv, cohort_means)
                minmax_pairs[coh] = (obs_s, exp_s)

        # Delta comparison (disease cohorts vs reference)
        delta_results = {}
        ref_exp = expected.get(reference)
        if ref_exp and ref_exp.bsv and reference in cohort_bsvs:
            for coh in ds.cohort_names:
                if coh == reference:
                    continue
                exp = expected.get(coh)
                if exp and exp.bsv:
                    delta_results[coh] = compute_delta_comparison(
                        cohort_bsvs[coh].mean_bsv, cohort_bsvs[reference].mean_bsv,
                        exp.bsv, ref_exp.bsv,
                    )

        cross = compute_cross_matrix_normalized(cohort_means, expected, "raw")
        interp = generate_interpretation(cross["alignment_summary"], delta_results, substrate)

    st.session_state.update({
        "v23_ds": ds, "v23_dsid": dsid, "v23_X_norm": X_norm, "v23_prep": prep,
        "v23_win_feats": win_feats, "v23_cohort_bsvs": cohort_bsvs,
        "v23_deltas": deltas, "v23_ref": reference,
        "v23_expected": expected, "v23_substrate": substrate,
        "v23_per_sims": per_sims, "v23_minmax": minmax_pairs,
        "v23_delta_results": delta_results, "v23_cross": cross,
        "v23_interp": interp,
    })

# ── Rendering ──────────────────────────────────────────────────────────

def _render():
    ds = st.session_state["v23_ds"]
    dsid = st.session_state["v23_dsid"]
    X_norm = st.session_state["v23_X_norm"]
    prep = st.session_state["v23_prep"]
    win_feats = st.session_state["v23_win_feats"]
    cohort_bsvs = st.session_state["v23_cohort_bsvs"]
    deltas = st.session_state["v23_deltas"]
    ref = st.session_state["v23_ref"]
    expected = st.session_state["v23_expected"]
    substrate = st.session_state["v23_substrate"]
    per_sims = st.session_state["v23_per_sims"]
    minmax_pairs = st.session_state["v23_minmax"]
    delta_results = st.session_state["v23_delta_results"]
    cross = st.session_state["v23_cross"]
    interp = st.session_state["v23_interp"]
    cats = [BSV_SHORT.get(c, c) for c in BSV_COMPONENTS]

    # ══════════════════════════════════════════════════════════
    # SECTION 1 — MEASURED SPECTRAL STRUCTURE
    # ══════════════════════════════════════════════════════════

    st.divider()
    st.header("Section 1 — Measured Spectral Structure")
    st.caption("All outputs below derived from measured spectra only.")

    # Dataset summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spectra", ds.n_spectra)
    c2.metric("Cohorts", len(ds.cohort_names))
    c3.metric("Substrate", substrate.get("substrate", "?"))
    c4.metric("Range", f"{ds.wavenumbers.min():.0f}–{ds.wavenumbers.max():.0f}")

    with st.expander("Preprocessing", expanded=False):
        st.markdown(f"- **Normalization:** {prep.normalization}")
        st.markdown(f"- **Baseline:** {prep.baseline}")
        st.caption(prep.notes)

    # Mean spectra
    st.subheader("Mean Spectra by Cohort")
    st.plotly_chart(mean_spectra_plot(X_norm, ds.wavenumbers, ds.cohorts),
                    use_container_width=True, config={"displayModeBar": False})

    # Raw BSV radar
    st.subheader("Raw Spectral BSV")
    st.plotly_chart(radar_plot(cohort_bsvs),
                    use_container_width=True, config={"displayModeBar": False})

    # Heatmaps
    ca, cb = st.columns(2)
    with ca:
        st.plotly_chart(bsv_heatmap(cohort_bsvs, height=250),
                        use_container_width=True, config={"displayModeBar": False})
    with cb:
        if deltas:
            st.plotly_chart(delta_heatmap(deltas, ref, height=220),
                            use_container_width=True, config={"displayModeBar": False})

    # Cohort traversals
    st.subheader("Cohort Traversals")
    st.caption("Cohort → Windows → Motifs → Themes → BSV. Spectral derivation only.")

    for coh, cbsv in cohort_bsvs.items():
        mask = ds.cohorts == coh
        g = build_cohort_graph(coh, cbsv, win_feats, mask)
        with st.expander(f"{_dn(coh)} ({cbsv.n_spectra} spectra · {g['n_motifs']} motifs · {g['n_bsv']} BSV)", expanded=False):
            st.plotly_chart(render_cohort_graph(g, height=260),
                            use_container_width=True, config={"displayModeBar": False})
            st.caption("Traversal derived from measured spectra only.")

    # ══════════════════════════════════════════════════════════
    # SECTION 2 — LITERATURE COMPARATOR
    # ══════════════════════════════════════════════════════════

    st.divider()
    st.header("Section 2 — Literature-Expected Comparator")
    st.caption(
        "Post-hoc comparison layer. Does NOT alter the spectral BSV above. "
        "Shared min-max scaling for cross-space visualization."
    )

    # Comparator summary
    st.subheader("Expected Comparators")
    for coh in ds.cohort_names:
        exp = expected.get(coh)
        if not exp:
            continue
        sc = {"direct": "green", "approximate": "orange", "unavailable": "red"}
        compat_icon = {"favorable": "🟢", "mixed": "🟡", "uncertain": "⚪"}.get(
            substrate.get("compatibility", "?"), "⚪")
        st.markdown(
            f"**{_dn(coh)}** → `{exp.comparator_name.replace('_', ' ')}` "
            f":{sc.get(exp.match_type, 'gray')}[{exp.match_type}] · "
            f"substrate: {compat_icon}"
        )
        st.caption(exp.explanation)

    # Shared min-max overlay radars
    if minmax_pairs:
        st.subheader("Observed vs Expected (Shared Min-Max)")
        st.caption("Both profiles scaled to a shared [0,1] range per axis for visual comparison.")

        for coh in ds.cohort_names:
            if coh not in minmax_pairs:
                continue
            obs_s, exp_s = minmax_pairs[coh]
            exp = expected[coh]
            sims = per_sims.get(coh, {})
            raw_cos = sims.get("raw", "?")

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=list(obs_s) + [obs_s[0]], theta=cats + [cats[0]],
                fill="toself", fillcolor=_hex_to_rgba("#3498DB", 0.10),
                line=dict(color="#3498DB", width=2), name="Observed (scaled)",
            ))
            fig.add_trace(go.Scatterpolar(
                r=list(exp_s) + [exp_s[0]], theta=cats + [cats[0]],
                fill="toself", fillcolor=_hex_to_rgba("#E74C3C", 0.10),
                line=dict(color="#E74C3C", width=2, dash="dash"),
                name=f"Expected ({exp.comparator_name.replace('_', ' ')})",
            ))
            fig.update_layout(
                title=dict(text=f"{_dn(coh)} — raw cosine = {raw_cos}",
                           font=dict(size=12, color="rgba(255,255,255,0.8)")),
                polar=dict(bgcolor=BG, radialaxis=dict(visible=True, range=[0, 1.05],
                    gridcolor="rgba(255,255,255,0.08)",
                    tickfont=dict(size=8, color="rgba(255,255,255,0.4)")),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.1)",
                    tickfont=dict(size=10, color="rgba(255,255,255,0.7)"))),
                paper_bgcolor=BG, font=dict(color="rgba(255,255,255,0.8)"),
                legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=60, r=60, t=50, b=30), height=300,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Delta comparison: observed shift vs expected shift
    if delta_results:
        st.subheader(f"Disease-vs-Reference Shift Comparison")
        st.caption(
            f"Observed shift (cohort − {_dn(ref)}) vs expected literature shift. "
            "Delta cosine measures whether the direction of change matches."
        )

        for coh, dr in delta_results.items():
            exp = expected[coh]
            obs_d = [dr["observed_delta"].get(c, 0) for c in BSV_COMPONENTS]
            exp_d = [dr["expected_delta"].get(c, 0) for c in BSV_COMPONENTS]

            fig = go.Figure()
            fig.add_trace(go.Bar(x=cats, y=obs_d, name="Observed shift",
                                  marker_color="#3498DB"))
            fig.add_trace(go.Bar(x=cats, y=exp_d, name="Expected shift",
                                  marker_color="#E74C3C", opacity=0.7))
            fig.update_layout(
                barmode="group",
                title=dict(text=f"{_dn(coh)} vs {_dn(ref)} — delta cosine = {dr['delta_cosine']:+.3f}",
                           font=dict(size=11, color="rgba(255,255,255,0.7)")),
                paper_bgcolor=BG, plot_bgcolor=BG, height=240,
                margin=dict(l=40, r=20, t=40, b=30),
                font=dict(color="rgba(255,255,255,0.8)", size=10),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=True,
                           zerolinecolor="rgba(255,255,255,0.2)", title="Δ BSV"),
                legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # Per-axis agreement table
            cat_icons = {"aligned": "🟢", "partial": "🟡", "divergent": "🔴", "weak": "⚪"}
            rows = [{
                "Axis": BSV_SHORT.get(a["component"], a["component"]),
                "Obs Δ": f"{a['obs_delta']:+.4f}",
                "Exp Δ": f"{a['exp_delta']:+.4f}",
                "Status": f"{cat_icons.get(a['category'], '?')} {a['category']}",
            } for a in dr["per_axis"]]
            with st.expander(f"Per-axis agreement ({_dn(coh)})", expanded=False):
                st.table(rows)

    # Similarity matrix
    if cross["matrix"]:
        st.subheader("Similarity Matrix")
        st.caption("Raw cosine. Rows = observed cohorts. Columns = expected profiles.")

        mat = np.array(cross["matrix"])
        obs_l = [_dn(c) for c in cross["observed_labels"]]
        exp_l = [c.replace("_", " ") for c in cross["expected_labels"]]

        fig = go.Figure(go.Heatmap(
            z=mat, x=exp_l, y=obs_l, colorscale="RdBu_r", zmid=0,
            texttemplate="%{z:.3f}", textfont=dict(size=11),
        ))
        fig.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG, height=max(200, 80 * len(obs_l)),
            margin=dict(l=10, r=10, t=30, b=10),
            font=dict(color="rgba(255,255,255,0.8)"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Alignment summary
    if cross["alignment_summary"]:
        st.subheader("Alignment Summary")
        rows = []
        for a in cross["alignment_summary"]:
            mi = "🟢" if a["margin"] > 0.05 else "🟡" if a["margin"] > 0 else "🔴"
            dc = delta_results.get(a["cohort"], {}).get("delta_cosine", "—")
            dc_str = f"{dc:+.3f}" if isinstance(dc, float) else dc
            rows.append({
                "Cohort": _dn(a["cohort"]),
                "Expected": a["own_expected"].replace("_", " "),
                "Match": a["match_type"],
                "Raw Cos": f"{a['own_cosine']:+.3f}",
                "Best Alt": a["best_alt_expected"].replace("_", " ") if a["best_alt_expected"] else "—",
                "Alt Cos": f"{a['best_alt_cosine']:+.3f}" if a["best_alt_expected"] else "—",
                "Margin": f"{mi} {a['margin']:+.3f}",
                "Δ Cosine": dc_str,
            })
        st.table(rows)

    # Interpretation
    if interp:
        st.subheader("Scientific Interpretation")
        st.markdown(interp)
        st.caption(
            "Positive margin = cohort aligns preferentially with its own expected profile. "
            "Positive delta cosine = disease-vs-reference shift direction matches expected."
        )


if "v23_ds" in st.session_state:
    _render()
