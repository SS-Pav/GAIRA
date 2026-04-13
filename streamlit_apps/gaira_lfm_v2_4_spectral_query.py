"""GAIRA Spectral Query v2.4 — Replication-First Build.

Faithful reproduction of original v1/v3 spectral query pipeline.
Observed BSV from direct spectral projection (no motifs).
Trust graphs for expected literature comparator only.
Sample-level similarity distributions included.

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src streamlit run streamlit_apps/gaira_lfm_v2_4_spectral_query.py
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
from gaira.spectral.expected_bsv import (
    build_expected_comparators, get_cohort_display_name,
)
from gaira.spectral.comparison import (
    compute_delta_comparison, compute_cross_matrix_normalized,
    get_substrate_context, generate_interpretation,
)

import warnings; warnings.filterwarnings("ignore", category=FutureWarning)


def _cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


# ── App ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="GAIRA v2.4 Spectral Query", layout="wide")
st.title("GAIRA Spectral Query v2.4")
st.caption("Replication of original v1/v3 pipeline. Direct spectral → BSV. No motif computation.")

datasets = discover_datasets()
if not datasets:
    st.error("No target datasets found."); st.stop()


def _dn(coh):
    return get_cohort_display_name(st.session_state.get("v24_dsid", ""), coh)


# ── Sidebar ────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Dataset")
    ds_map = {d.display_name: d for d in datasets}
    sel_name = st.selectbox("Target:", list(ds_map.keys()))
    sel = ds_map[sel_name]
    st.caption(f"`{sel.dataset_id}` · {sel.n_spectra} spectra · {sel.family}")
    if sel.note:
        st.caption(sel.note)

    for coh, n in sorted(sel.cohorts.items()):
        st.markdown(f"- {get_cohort_display_name(sel.dataset_id, coh)} (n={n})")

    st.divider()
    coh_list = sorted(sel.cohorts.keys())
    def_ref = "healthy_control" if "healthy_control" in coh_list else coh_list[0]
    reference = st.selectbox("Reference:", coh_list, index=coh_list.index(def_ref))
    st.divider()
    run = st.button("Run", type="primary")

# ── Pipeline ───────────────────────────────────────────────────────────

if run:
    dsid = sel.dataset_id
    with st.spinner("Processing..."):
        ds = load_dataset(dsid)
        X_norm, prep = preprocess(ds)
        win_feats = extract_window_features(X_norm, ds.wavenumbers)
        bsv_matrix = project_to_bsv(win_feats)
        cbsvs = compute_cohort_bsvs(bsv_matrix, ds.cohorts)
        deltas = compute_deltas(cbsvs, reference)

        # Band drivers
        drivers = compute_per_cohort_window_importance(win_feats, ds.cohorts, reference)

        # Expected
        expected = build_expected_comparators(dsid, ds.cohort_names)
        substrate = get_substrate_context(dsid)
        means = {c: cb.mean_bsv for c, cb in cbsvs.items()}

        # Sample-level similarity distributions
        sample_sims = {}
        for coh, cbsv in cbsvs.items():
            exp = expected.get(coh)
            if not exp or not exp.bsv:
                continue
            exp_vec = np.array([exp.bsv[c] for c in BSV_COMPONENTS])
            sims_own = [_cosine(cbsv.sample_bsv[i], exp_vec) for i in range(cbsv.n_spectra)]

            # Also compute vs all other expected profiles
            alt_sims = {}
            for other_coh, other_exp in expected.items():
                if other_coh == coh or not other_exp.bsv:
                    continue
                other_vec = np.array([other_exp.bsv[c] for c in BSV_COMPONENTS])
                alt_sims[other_coh] = [_cosine(cbsv.sample_bsv[i], other_vec)
                                        for i in range(cbsv.n_spectra)]

            sample_sims[coh] = {"own": sims_own, "alternatives": alt_sims}

        # Delta comparison
        ref_exp = expected.get(reference)
        delta_results = {}
        if ref_exp and ref_exp.bsv:
            for coh in ds.cohort_names:
                if coh == reference:
                    continue
                exp = expected.get(coh)
                if exp and exp.bsv:
                    delta_results[coh] = compute_delta_comparison(
                        cbsvs[coh].mean_bsv, cbsvs[reference].mean_bsv,
                        exp.bsv, ref_exp.bsv)

        cross = compute_cross_matrix_normalized(means, expected, "raw")
        interp = generate_interpretation(cross["alignment_summary"], delta_results, substrate)

        # PCA
        pca_obj = PCA(n_components=2)
        pca_proj = pca_obj.fit_transform(bsv_matrix)

    st.session_state.update({
        "v24_ds": ds, "v24_dsid": dsid, "v24_X_norm": X_norm, "v24_prep": prep,
        "v24_win": win_feats, "v24_bsv": bsv_matrix,
        "v24_cbsvs": cbsvs, "v24_deltas": deltas, "v24_ref": reference,
        "v24_drivers": drivers, "v24_expected": expected, "v24_substrate": substrate,
        "v24_sample_sims": sample_sims, "v24_delta_results": delta_results,
        "v24_cross": cross, "v24_interp": interp,
        "v24_pca": pca_proj, "v24_pca_obj": pca_obj,
    })


# ── Rendering ──────────────────────────────────────────────────────────

def _render():
    ds = st.session_state["v24_ds"]
    dsid = st.session_state["v24_dsid"]
    X_norm = st.session_state["v24_X_norm"]
    prep = st.session_state["v24_prep"]
    win_feats = st.session_state["v24_win"]
    bsv_matrix = st.session_state["v24_bsv"]
    cbsvs = st.session_state["v24_cbsvs"]
    deltas = st.session_state["v24_deltas"]
    ref = st.session_state["v24_ref"]
    drivers = st.session_state["v24_drivers"]
    expected = st.session_state["v24_expected"]
    substrate = st.session_state["v24_substrate"]
    sample_sims = st.session_state["v24_sample_sims"]
    delta_results = st.session_state["v24_delta_results"]
    cross = st.session_state["v24_cross"]
    interp = st.session_state["v24_interp"]
    pca_proj = st.session_state["v24_pca"]
    pca_obj = st.session_state["v24_pca_obj"]
    cats = [BSV_SHORT.get(c, c) for c in BSV_COMPONENTS]

    # ══════════════════════════════════════════════════════════
    # SECTION 1 — OBSERVED SPECTRAL STRUCTURE
    # ══════════════════════════════════════════════════════════

    st.divider()
    st.header("1 — Observed Spectral Structure")
    st.caption("Direct: preprocessed spectra → spectral windows → BSV projection.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spectra", ds.n_spectra)
    c2.metric("Cohorts", len(ds.cohort_names))
    c3.metric("Substrate", substrate.get("substrate", "?"))
    c4.metric("Pipeline", prep.pipeline)

    with st.expander("Preprocessing"):
        st.markdown(f"- **Baseline:** {prep.baseline}")
        st.markdown(f"- **Smoothing:** {prep.smoothing}")
        st.markdown(f"- **Normalization:** {prep.normalization}")
        st.caption(prep.notes)

    st.subheader("Mean Spectra")
    st.plotly_chart(mean_spectra_plot(X_norm, ds.wavenumbers, ds.cohorts),
                    use_container_width=True, config={"displayModeBar": False})

    st.subheader("Observed BSV")
    st.plotly_chart(radar_plot(cbsvs),
                    use_container_width=True, config={"displayModeBar": False})

    ca, cb = st.columns(2)
    with ca:
        st.plotly_chart(bsv_heatmap(cbsvs, height=250),
                        use_container_width=True, config={"displayModeBar": False})
    with cb:
        if deltas:
            st.plotly_chart(delta_heatmap(deltas, ref, height=220),
                            use_container_width=True, config={"displayModeBar": False})

    # Sample-level BSV distributions
    st.subheader("Sample-Level BSV Distributions")
    top_axes = sorted(range(8), key=lambda i: max(
        abs(cbsvs[c].mean_bsv[BSV_COMPONENTS[i]] - cbsvs[ref].mean_bsv[BSV_COMPONENTS[i]])
        for c in ds.cohort_names if c != ref
    ) if len(ds.cohort_names) > 1 else 0, reverse=True)[:4]

    fig_dist = go.Figure()
    for ai, ci in enumerate(top_axes):
        comp = BSV_COMPONENTS[ci]
        for j, coh in enumerate(sorted(set(ds.cohorts))):
            vals = cbsvs[coh].sample_bsv[:, ci]
            fig_dist.add_trace(go.Box(
                y=vals, name=f"{BSV_SHORT.get(comp, comp)} · {_dn(coh)}",
                marker_color=_color_for(j), boxmean=True,
                visible=True if ai == 0 else "legendonly",
            ))

    fig_dist.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, height=300,
        margin=dict(l=40, r=20, t=20, b=30),
        font=dict(color="rgba(255,255,255,0.8)", size=9),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="BSV value"),
        legend=dict(font=dict(size=8), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})

    # PCA
    st.subheader("PCA in BSV Space")
    fig_pca = go.Figure()
    for j, coh in enumerate(sorted(set(ds.cohorts))):
        mask = ds.cohorts == coh
        fig_pca.add_trace(go.Scatter(
            x=pca_proj[mask, 0], y=pca_proj[mask, 1], mode="markers",
            marker=dict(color=_color_for(j), size=5, opacity=0.5),
            name=_dn(coh),
        ))
    fig_pca.update_layout(
        xaxis_title=f"PC1 ({pca_obj.explained_variance_ratio_[0]*100:.1f}%)",
        yaxis_title=f"PC2 ({pca_obj.explained_variance_ratio_[1]*100:.1f}%)",
        paper_bgcolor=BG, plot_bgcolor=BG, height=320,
        margin=dict(l=50, r=20, t=20, b=40),
        font=dict(color="rgba(255,255,255,0.8)"),
        legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
    )
    st.plotly_chart(fig_pca, use_container_width=True, config={"displayModeBar": False})

    # Band drivers
    if drivers:
        st.subheader("Spectral Band Drivers")
        st.caption(f"Top windows driving BSV differences vs {_dn(ref)}.")
        for coh, wins in drivers.items():
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
                margin=dict(l=40, r=20, t=35, b=40),
                font=dict(color="rgba(255,255,255,0.8)", size=9),
                xaxis=dict(tickangle=-45),
                yaxis=dict(title="Effect size", gridcolor="rgba(255,255,255,0.06)",
                           zeroline=True, zerolinecolor="rgba(255,255,255,0.2)"),
            )
            st.plotly_chart(fig_bd, use_container_width=True, config={"displayModeBar": False})

    # ══════════════════════════════════════════════════════════
    # SECTION 2 — EXPECTED LITERATURE COMPARATOR
    # ══════════════════════════════════════════════════════════

    st.divider()
    st.header("2 — Expected Literature Comparator")
    st.caption("Literature-grounded BSV. Does not influence observed spectral analysis.")

    for coh in ds.cohort_names:
        exp = expected.get(coh)
        if not exp:
            continue
        sc = {"direct": "green", "approximate": "orange", "unavailable": "red"}
        compat = {"favorable": "🟢", "mixed": "🟡", "uncertain": "⚪"}.get(
            substrate.get("compatibility"), "⚪")
        st.markdown(
            f"**{_dn(coh)}** → `{exp.comparator_name.replace('_', ' ')}` "
            f":{sc.get(exp.match_type, 'gray')}[{exp.match_type}] · {compat}"
        )
        st.caption(exp.explanation)

    # ══════════════════════════════════════════════════════════
    # SECTION 3 — OBSERVED VS EXPECTED COMPARISON
    # ══════════════════════════════════════════════════════════

    st.divider()
    st.header("3 — Observed vs Expected")

    # Cross-similarity matrix
    if cross["matrix"]:
        st.subheader("Similarity Matrix")
        st.caption("Raw cosine. Rows = observed cohorts. Columns = expected profiles.")
        mat = np.array(cross["matrix"])
        fig = go.Figure(go.Heatmap(
            z=mat,
            x=[c.replace("_", " ") for c in cross["expected_labels"]],
            y=[_dn(c) for c in cross["observed_labels"]],
            colorscale="RdBu_r", zmid=0,
            texttemplate="%{z:.3f}", textfont=dict(size=12),
        ))
        fig.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG,
            height=max(200, 90 * len(cross["observed_labels"])),
            margin=dict(l=10, r=10, t=30, b=10),
            font=dict(color="rgba(255,255,255,0.8)"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Sample-level similarity distributions
    if sample_sims:
        st.subheader("Sample-Level Alignment")
        st.caption("Per-sample cosine to own expected vs alternative expected profiles.")

        for coh, sdata in sample_sims.items():
            own = np.array(sdata["own"])
            fig_sl = go.Figure()
            fig_sl.add_trace(go.Histogram(
                x=own, name=f"→ own ({expected[coh].comparator_name.replace('_', ' ')})",
                marker_color="#3498DB", opacity=0.7, nbinsx=25,
            ))
            for alt_coh, alt_vals in sdata["alternatives"].items():
                fig_sl.add_trace(go.Histogram(
                    x=alt_vals, name=f"→ {expected[alt_coh].comparator_name.replace('_', ' ')}",
                    marker_color="#E74C3C", opacity=0.5, nbinsx=25,
                ))

            own_mean = float(own.mean())
            alt_means = {k: float(np.mean(v)) for k, v in sdata["alternatives"].items()}
            best_alt = max(alt_means.values()) if alt_means else 0
            sep = own_mean - best_alt

            fig_sl.update_layout(
                barmode="overlay",
                title=dict(text=f"{_dn(coh)} samples — own={own_mean:.3f}, "
                           f"best alt={best_alt:.3f}, separation={sep:+.3f}",
                           font=dict(size=10, color="rgba(255,255,255,0.7)")),
                paper_bgcolor=BG, plot_bgcolor=BG, height=220,
                margin=dict(l=40, r=20, t=40, b=30),
                font=dict(color="rgba(255,255,255,0.8)", size=9),
                xaxis=dict(title="Cosine similarity"),
                yaxis=dict(title="Count", gridcolor="rgba(255,255,255,0.06)"),
                legend=dict(font=dict(size=8), bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_sl, use_container_width=True, config={"displayModeBar": False})

    # Delta shift comparison
    if delta_results:
        st.subheader("Disease-vs-Reference Shift")
        st.caption(f"Observed Δ(cohort − {_dn(ref)}) vs expected literature Δ.")

        for coh, dr in delta_results.items():
            obs_d = [dr["observed_delta"].get(c, 0) for c in BSV_COMPONENTS]
            exp_d = [dr["expected_delta"].get(c, 0) for c in BSV_COMPONENTS]

            fig = go.Figure()
            fig.add_trace(go.Bar(x=cats, y=obs_d, name="Observed Δ", marker_color="#3498DB"))
            fig.add_trace(go.Bar(x=cats, y=exp_d, name="Expected Δ", marker_color="#E74C3C", opacity=0.7))
            fig.update_layout(
                barmode="group",
                title=dict(text=f"{_dn(coh)} — delta cosine = {dr['delta_cosine']:+.3f}",
                           font=dict(size=10, color="rgba(255,255,255,0.7)")),
                paper_bgcolor=BG, plot_bgcolor=BG, height=230,
                margin=dict(l=40, r=20, t=40, b=30),
                font=dict(color="rgba(255,255,255,0.8)", size=9),
                yaxis=dict(title="Δ BSV", gridcolor="rgba(255,255,255,0.06)",
                           zeroline=True, zerolinecolor="rgba(255,255,255,0.2)"),
                legend=dict(font=dict(size=8), bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # Per-axis agreement
            cat_icons = {"aligned": "🟢", "partial": "🟡", "divergent": "🔴", "weak": "⚪"}
            with st.expander(f"Per-axis ({_dn(coh)})"):
                st.table([{
                    "Axis": BSV_SHORT.get(a["component"], a["component"]),
                    "Obs Δ": f"{a['obs_delta']:+.5f}",
                    "Exp Δ": f"{a['exp_delta']:+.4f}",
                    "": f"{cat_icons.get(a['category'], '?')} {a['category']}",
                } for a in dr["per_axis"]])

    # Alignment summary
    if cross["alignment_summary"]:
        st.subheader("Alignment Summary")
        rows = []
        for a in cross["alignment_summary"]:
            mi = "🟢" if a["margin"] > 0.05 else "🟡" if a["margin"] > 0 else "🔴"
            dc = delta_results.get(a["cohort"], {}).get("delta_cosine")

            # Sample-level separation
            ss = sample_sims.get(a["cohort"])
            if ss:
                own_m = float(np.mean(ss["own"]))
                alt_m = max((float(np.mean(v)) for v in ss["alternatives"].values()), default=0)
                sep = f"{own_m - alt_m:+.3f}"
            else:
                sep = "—"

            rows.append({
                "Cohort": _dn(a["cohort"]),
                "Expected": a["own_expected"].replace("_", " "),
                "Match": a["match_type"],
                "Cohort Cos": f"{a['own_cosine']:+.3f}",
                "Alt Cos": f"{a['best_alt_cosine']:+.3f}" if a["best_alt_expected"] else "—",
                "Margin": f"{mi} {a['margin']:+.3f}",
                "Δ Cos": f"{dc:+.3f}" if dc is not None else "—",
                "Sample Sep": sep,
            })
        st.table(rows)

    # Interpretation
    if interp:
        st.subheader("Interpretation")
        st.markdown(interp)
        st.caption(
            "Observed BSV = direct spectral projection (no motifs). "
            "Positive margin = cohort-level preferential alignment. "
            "Positive sample separation = sample-level preferential alignment. "
            "Positive delta cosine = shift direction consistent with expected."
        )


if "v24_ds" in st.session_state:
    _render()
