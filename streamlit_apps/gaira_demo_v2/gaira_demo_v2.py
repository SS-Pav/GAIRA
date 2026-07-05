"""GAIRA — polished demo v2 (dark theme · readability pass).

Preserves the architecture of v1 (streamlit_apps/gaira_demo/gaira_demo.py)
and reuses its derived assets. Differences:
  - dark-first CSS chrome
  - centralized apply_dark_theme() on every Plotly figure
  - brighter axis palette tuned for dark bg
  - explicit polar-axis styling so radar labels are readable
  - atlas ruler with classification legend + visible borders

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/streamlit run streamlit_apps/gaira_demo_v2/gaira_demo_v2.py \\
        --theme.base dark --theme.backgroundColor "#0B1220" \\
        --theme.secondaryBackgroundColor "#111827" \\
        --theme.primaryColor "#60A5FA" --theme.textColor "#F1F5F9"
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from helpers import (
    AXIS_COLORS, AXIS_LABELS, BSV_COMPONENTS,
    DARK_BG_PAPER, DARK_BG_PLOT, GRID_COLOR,
    LEGEND_BG, LEGEND_BORDER,
    OVERLAY_COLORS, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, TITLE_COLOR,
    add_atlas_band_shading, apply_dark_theme, atlas_ruler_figure,
    bsv_bar_figure, canonical_bsv, canonical_bsv_from_cols,
    delta_heatmap_figure, format_conc,
    load_atlas_explorer, load_axis_coverage, load_calibration_conditions,
    load_calibration_delta_bsv, load_corpus_summary, load_erg_dose_long,
    load_erg_per_conc, load_family_counts,
    load_molecule_bsv, load_molecule_index, load_molecule_spectra,
    radar_figure, spectra_overlay_figure,
)


st.set_page_config(
    page_title="GAIRA v2 — demo",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────
# Page-level CSS (enforces dark theme even without config.toml)
# ──────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(ellipse at top, #101B31 0%, #0B1220 55%);
        color: #F1F5F9;
    }
    .block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1320px; }
    h1, h2, h3, h4 { color: #F8FAFC; letter-spacing: -0.01em; }
    p, li, label { color: #CBD5E1; }
    .caption-muted { color: #94A3B8; font-size: 0.95rem; line-height: 1.55; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(17, 24, 39, 0.6);
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 10px;
        padding: 10px 14px;
    }
    [data-testid="stMetricValue"] { color: #F8FAFC; }
    [data-testid="stMetricLabel"] { color: #94A3B8; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid rgba(148,163,184,0.18); }
    .stTabs [data-baseweb="tab"] {
        background: rgba(17, 24, 39, 0.6);
        color: #CBD5E1;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        border: 1px solid rgba(148,163,184,0.12);
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background: #1E293B;
        color: #F8FAFC;
        border-color: rgba(96,165,250,0.55);
        border-bottom: 2px solid #60A5FA;
    }

    /* Widgets */
    div[data-baseweb="select"] > div, input, textarea {
        background: #111827 !important; color: #F1F5F9 !important;
        border: 1px solid rgba(148,163,184,0.25) !important;
    }
    div[data-baseweb="popover"] { background: #111827 !important; color: #F1F5F9 !important; }
    div[data-testid="stExpander"] {
        background: rgba(17, 24, 39, 0.55);
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 8px;
    }

    /* Inline pills */
    .pill {
        display: inline-block; padding: 2px 10px; margin-right: 6px;
        border-radius: 999px; font-size: 0.78rem; font-weight: 600;
        background: rgba(99,102,241,0.2); color: #A5B4FC;
        border: 1px solid rgba(99,102,241,0.35);
    }
    .pill-ok   { background: rgba(16,185,129,0.18); color: #6EE7B7; border-color: rgba(52,211,153,0.4); }
    .pill-warn { background: rgba(245,158,11,0.18); color: #FBBF24; border-color: rgba(251,191,36,0.4); }
    .pill-bad  { background: rgba(239,68,68,0.2);   color: #FCA5A5; border-color: rgba(248,113,113,0.4); }

    .box {
        background: rgba(17, 24, 39, 0.62);
        border: 1px solid rgba(148,163,184,0.22);
        border-radius: 10px; padding: 14px 18px; color: #E2E8F0;
    }
    .box code {
        background: rgba(148,163,184,0.12); color: #FDE68A;
        padding: 1px 6px; border-radius: 4px;
    }

    hr { border-color: rgba(148,163,184,0.22); }

    /* Dataframe surface */
    [data-testid="stDataFrame"] div { color: #E2E8F0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────

st.title("GAIRA — biochemical reasoning from Raman / SERS")
st.markdown(
    '<div class="caption-muted">'
    "Domain-aware, uncertainty-aware evidence engine. "
    "Spectra are treated as mixtures, not fingerprints. "
    "Interpretation is region-based, multi-axis, and ambiguity-tracked."
    '<span style="color:#60A5FA; margin-left:10px;">· v2 · dark readability pass</span>'
    "</div>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs([
    "  Methods / Pipeline  ",
    "  Grounding  ",
    "  Calibration  ",
    "  Regression / Dose-response  ",
])


# ──────────────────────────────────────────────────────────────────────
# TAB 1 — METHODS / PIPELINE
# ──────────────────────────────────────────────────────────────────────

with tab1:
    st.subheader("Pipeline at a glance")

    repo_root = Path(__file__).resolve().parents[2]
    hero_path = repo_root / "figures" / "gaira_three_phase_master_figure.png"
    pipe_path = repo_root / "figures" / "phase2_bsv_pipeline_figure.png"
    bsv_vs_peak_path = repo_root / "figures" / "phase2_bsv_vs_peak_matching_explainer.png"

    if hero_path.exists():
        st.image(
            str(hero_path), width="stretch",
            caption="GAIRA three-phase system: grounding → representation (BSV) → structured evidence.",
        )
    else:
        st.info("Hero figure not found (figures/gaira_three_phase_master_figure.png).")

    st.markdown(
        '<div class="caption-muted">'
        "<b style='color:#F1F5F9;'>One-line description.</b> GAIRA projects a measured spectrum into an "
        "8-axis Biochemical State Vector (BSV), then interprets changes (ΔBSV) "
        "against a curated Raman physics atlas and a grounding corpus of pure molecules."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Stages")
    c1, c2, c3, c4 = st.columns(4)
    stage_html = lambda n, title, blurb: (
        f"<div class='box' style='min-height:112px;'>"
        f"<div style='color:#60A5FA; font-weight:700; letter-spacing:.02em;'>{n}</div>"
        f"<div style='color:#F8FAFC; font-weight:600; margin:4px 0 6px 0;'>{title}</div>"
        f"<div style='color:#94A3B8; font-size:0.88rem;'>{blurb}</div>"
        f"</div>"
    )
    c1.markdown(stage_html("①", "Grounding",
        "Pure molecules (RamanBioLib) + axis-linked literature fix axis meanings."), unsafe_allow_html=True)
    c2.markdown(stage_html("②", "Raman physics atlas",
        "Bands with ranges, primary axis, companions, ambiguity, locality."), unsafe_allow_html=True)
    c3.markdown(stage_html("③", "Preprocess → window panel",
        "AsLS + SG + L2 → 22 windows → 8-axis BSV."), unsafe_allow_html=True)
    c4.markdown(stage_html("④", "Scorer · BSV · ΔBSV",
        "Calibration contrasts scored on testable axes only (R7c, SAEL)."), unsafe_allow_html=True)

    if pipe_path.exists():
        with st.expander("BSV pipeline schematic", expanded=False):
            st.image(str(pipe_path), width="stretch")
    if bsv_vs_peak_path.exists():
        with st.expander("BSV vs peak-matching explainer", expanded=False):
            st.image(str(bsv_vs_peak_path), width="stretch")

    st.markdown("---")

    # ── Grounding corpus visualizer ─────────────────────────────
    st.subheader("Grounding corpus")

    summary = load_corpus_summary()
    family_counts = load_family_counts()

    if not summary.empty:
        m = dict(zip(summary["metric"], summary["value"]))
        kc1, kc2, kc3, kc4 = st.columns(4)
        kc1.metric("Pure-molecule spectra", int(m.get("n_molecule_spectra", 0)))
        kc2.metric("Atlas bands", int(m.get("n_atlas_bands", 0)))
        kc3.metric("Canonical axes", int(m.get("n_atlas_axes", 0)))
        rng = f"{int(m.get('wavenumber_min_cm1', 0))} – {int(m.get('wavenumber_max_cm1', 0))}"
        kc4.metric("Spectral range (cm⁻¹)", rng)

    col_a, col_b = st.columns([1.1, 1.0])
    with col_a:
        if not family_counts.empty:
            fig = go.Figure(go.Bar(
                x=family_counts["n_molecules"],
                y=family_counts["family"],
                orientation="h",
                marker=dict(color="#60A5FA",
                             line=dict(color="rgba(255,255,255,0.25)", width=0.5)),
                hovertemplate="<b>%{y}</b><br>%{x} molecules<extra></extra>",
            ))
            apply_dark_theme(
                fig, title="Grounding corpus by family",
                height=340, show_legend=False,
                margin=dict(l=130, r=30, t=40, b=40),
            )
            fig.update_xaxes(title="Pure molecules")
            fig.update_yaxes(title="", autorange="reversed", automargin=True)
            st.plotly_chart(fig, width="stretch")

    with col_b:
        axis_cov = load_axis_coverage()
        if not axis_cov.empty:
            cols = [c for c in ["anchor", "secondary", "ambiguous"] if c in axis_cov.columns]
            color_for = {"anchor": "#34D399", "secondary": "#60A5FA", "ambiguous": "#F87171"}
            fig = go.Figure()
            for c in cols:
                fig.add_trace(go.Bar(
                    name=c.capitalize(),
                    x=axis_cov["primary_axis"].map(lambda a: AXIS_LABELS.get(a, a)),
                    y=axis_cov[c],
                    marker=dict(color=color_for.get(c, "#64748B"),
                                 line=dict(color="rgba(255,255,255,0.25)", width=0.4)),
                    hovertemplate="<b>%{x}</b><br>" + c + ": %{y}<extra></extra>",
                ))
            apply_dark_theme(
                fig, title="Atlas band coverage by axis",
                height=340,
                margin=dict(l=50, r=30, t=40, b=80),
            )
            fig.update_layout(barmode="stack")
            fig.update_xaxes(title="", tickangle=-30)
            fig.update_yaxes(title="Number of bands")
            st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    # ── Atlas explorer ───────────────────────────────────────────
    st.subheader("Raman physics atlas — band explorer")
    atlas = load_atlas_explorer()
    if atlas.empty:
        st.warning("Atlas table not found.")
    else:
        ec1, ec2, ec3 = st.columns([1.0, 1.0, 1.0])
        axes_opts = ["(all axes)"] + sorted(atlas["primary_axis"].unique().tolist())
        class_opts = ["(all classes)"] + sorted(atlas["classification"].unique().tolist())
        axis_pick = ec1.selectbox("Primary axis", axes_opts, index=0, key="atlas_axis")
        class_pick = ec2.selectbox("Classification", class_opts, index=0, key="atlas_class")
        show_companion_only = ec3.checkbox(
            "Only bands with companion axes", value=False,
            help="Bands where more than one primary axis is candidate (multi-assignment).",
        )

        view = atlas.copy()
        if axis_pick != "(all axes)":
            view = view[view["primary_axis"] == axis_pick]
        if class_pick != "(all classes)":
            view = view[view["classification"] == class_pick]
        if show_companion_only:
            view = view[view["has_companion"]]

        st.caption(f"Showing {len(view)} / {len(atlas)} atlas bands.")

        axes_unique = [a for a in BSV_COMPONENTS if a in view["primary_axis"].unique()]
        if axes_unique:
            fig = atlas_ruler_figure(view, axes_unique)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No bands match the current filter.")

        with st.expander("Band table", expanded=False):
            preview = view[[
                "display_label", "range_label", "classification",
                "candidate_axes", "ambiguity_score", "locality_score",
                "source_count", "priority_tags",
            ]].rename(columns={
                "display_label": "Band", "range_label": "Range (cm⁻¹)",
                "classification": "Class", "candidate_axes": "Candidate axes",
                "ambiguity_score": "Ambiguity", "locality_score": "Locality",
                "source_count": "Sources", "priority_tags": "Tags",
            })
            st.dataframe(preview, width="stretch", hide_index=True)


# ──────────────────────────────────────────────────────────────────────
# TAB 2 — GROUNDING
# ──────────────────────────────────────────────────────────────────────

with tab2:
    st.subheader("Pure molecule grounding")

    mol_index = load_molecule_index()
    mol_bsv = load_molecule_bsv()
    mol_spec = load_molecule_spectra()

    if mol_index.empty or mol_bsv.empty or mol_spec.empty:
        st.warning("Grounding assets missing.  Run build_demo_assets.py (v1).")
    else:
        default_names = [
            n for n in ["L-ergothioneine", "Hypoxanthine", "Uric acid"]
            if n in mol_index["component"].values
        ] or mol_index["component"].head(2).tolist()

        left, right = st.columns([1.0, 0.95])
        with left:
            fam_opts = ["(all families)"] + sorted(mol_index["family"].unique().tolist())
            fam_pick = st.selectbox("Family filter", fam_opts, index=0, key="grd_family")
            pool = mol_index
            if fam_pick != "(all families)":
                pool = pool[pool["family"] == fam_pick]
            pool = pool.sort_values("component")
            selected = st.multiselect(
                "Molecules (up to 5)",
                pool["component"].tolist(),
                default=default_names[:3],
                max_selections=5,
                key="grd_molecules",
            )

        if not selected:
            st.info("Pick at least one molecule above.")
        else:
            sel_rows = mol_index[mol_index["component"].isin(selected)]
            sel_ids = sel_rows["id"].tolist()

            spectra_plot = []
            for mol_id in sel_ids:
                sub = mol_spec[mol_spec["id"] == mol_id].sort_values("wavenumber")
                if sub.empty:
                    continue
                name = mol_index.set_index("id").loc[mol_id, "component"]
                spectra_plot.append(
                    (str(name), sub["wavenumber"].to_numpy(), sub["intensity_norm"].to_numpy())
                )

            atlas = load_atlas_explorer()
            fig_spec = spectra_overlay_figure(
                spectra_plot,
                title="Processed spectra (min-max normalized · display only)",
                height=430,
            )
            fig_spec = add_atlas_band_shading(
                fig_spec,
                atlas[atlas["classification"] == "anchor"],
                alpha=0.12,
            )
            with right:
                st.caption(
                    "Gentle shaded stripes mark atlas **anchor** bands, "
                    "colored by primary axis."
                )
            st.plotly_chart(fig_spec, width="stretch")

            st.markdown("")
            colL, colR = st.columns([1.05, 1.0])

            sel_bsv_rows = mol_bsv[mol_bsv["id"].isin(sel_ids)]
            if len(sel_bsv_rows) == 1:
                row = sel_bsv_rows.iloc[0]
                bsv_vals = canonical_bsv(row)
                bar_title = f"BSV contribution — {row['component']}"
            else:
                bsv_vals = sel_bsv_rows[BSV_COMPONENTS].mean(axis=0).tolist()
                bar_title = "BSV contribution — mean across selected"

            with colL:
                fig_bar = bsv_bar_figure(
                    bsv_vals, title=bar_title,
                    y_label="BSV (pure-molecule · no baseline)",
                    signed=False, height=380,
                )
                st.plotly_chart(fig_bar, width="stretch")
                st.caption(
                    "**Pure molecules have no biological baseline → ΔBSV is not meaningful here.** "
                    "We show BSV contribution instead."
                )

            with colR:
                traces = []
                for i, mol_id in enumerate(sel_ids):
                    brow = sel_bsv_rows[sel_bsv_rows["id"] == mol_id]
                    if brow.empty:
                        continue
                    vals = canonical_bsv(brow.iloc[0])
                    color = OVERLAY_COLORS[i % len(OVERLAY_COLORS)]
                    traces.append({
                        "name": str(brow.iloc[0]["component"]),
                        "values": vals, "color": color,
                        "fill_alpha": 0.22 if len(sel_ids) > 1 else 0.34,
                    })
                radial_max = max(
                    0.15, float(sel_bsv_rows[BSV_COMPONENTS].values.max()) * 1.18,
                )
                fig_radar = radar_figure(
                    traces,
                    title="BSV radar (canonical axis order)",
                    radial_max=radial_max, height=460,
                )
                st.plotly_chart(fig_radar, width="stretch")

            st.markdown("### Context")
            ctx = sel_bsv_rows.merge(
                mol_index[["id", "type"]], on="id", how="left"
            )[["component", "family", "type", "dominant_axis", "dominant_weight"]].rename(
                columns={
                    "component": "Molecule", "family": "Family", "type": "Type",
                    "dominant_axis": "Dominant axis", "dominant_weight": "Weight",
                }
            )
            ctx["Weight"] = ctx["Weight"].map(lambda v: f"{v:.3f}")
            ctx["Dominant axis"] = ctx["Dominant axis"].map(
                lambda a: AXIS_LABELS.get(a, a)
            )
            st.dataframe(ctx, width="stretch", hide_index=True)
            st.caption(
                "**Caveat.** GAIRA represents mixtures, not fingerprints. "
                "Dominant-axis summaries describe spectral projection only, "
                "not exact molecule identity."
            )


# ──────────────────────────────────────────────────────────────────────
# TAB 3 — CALIBRATION
# ──────────────────────────────────────────────────────────────────────

with tab3:
    st.subheader("Calibration — representation-space response")

    conds = load_calibration_conditions()
    deltas = load_calibration_delta_bsv()

    if conds.empty or deltas.empty:
        st.warning("Calibration assets missing.")
    else:
        label_map = dict(zip(conds["contrast_id"], conds["display_name"]))

        c1, c2 = st.columns([1.3, 1.0])
        with c1:
            selected = st.multiselect(
                "Calibration contrasts (compare up to 4)",
                conds["contrast_id"].tolist(),
                default=[conds["contrast_id"].iloc[0], conds["contrast_id"].iloc[1]],
                format_func=lambda cid: label_map.get(cid, cid),
                max_selections=4, key="cal_contrasts",
            )
        with c2:
            view_mode = st.radio(
                "ΔBSV view",
                ["Bar per contrast", "Heatmap across contrasts", "Radar overlay"],
                horizontal=False, index=0, key="cal_view_mode",
            )

        if not selected:
            st.info("Pick at least one calibration contrast.")
        else:
            rows, labels, vmax = [], [], 0.0
            testable_map = {}
            for cid in selected:
                subset = deltas[deltas["contrast_id"] == cid]
                if subset.empty:
                    continue
                axis_to_val = dict(zip(subset["axis"], subset["observed_delta"]))
                axis_to_test = dict(zip(subset["axis"], subset["testable"]))
                vals = [float(axis_to_val.get(a, 0.0)) for a in BSV_COMPONENTS]
                rows.append(vals)
                labels.append(label_map.get(cid, cid))
                testable_map[cid] = axis_to_test
                vmax = max(vmax, max(abs(v) for v in vals))
            vmax = max(0.005, vmax * 1.1)

            if view_mode == "Heatmap across contrasts":
                matrix = np.asarray(rows)
                fig = delta_heatmap_figure(
                    matrix,
                    row_labels=labels,
                    col_labels=[AXIS_LABELS[c] for c in BSV_COMPONENTS],
                    title="ΔBSV — observed delta by condition",
                    height=max(260, 60 * len(rows) + 120),
                    vmax=vmax,
                )
                st.plotly_chart(fig, width="stretch")

            elif view_mode == "Bar per contrast":
                for i, (cid, vals) in enumerate(zip(selected, rows)):
                    st.markdown(f"**{label_map.get(cid, cid)}**")
                    fig = bsv_bar_figure(
                        vals, title=None, y_label="ΔBSV (perturbed − control)",
                        signed=True, height=320, y_range=(-vmax, vmax),
                    )
                    axis_test = testable_map.get(cid, {})
                    shapes = list(fig.layout.shapes) if fig.layout.shapes else []
                    for j, a in enumerate(BSV_COMPONENTS):
                        if not bool(axis_test.get(a, False)):
                            shapes.append(dict(
                                type="rect",
                                x0=j - 0.5, x1=j + 0.5, xref="x",
                                y0=0, y1=1, yref="paper",
                                fillcolor="rgba(148,163,184,0.08)",
                                line=dict(width=0), layer="below",
                            ))
                    fig.update_layout(shapes=shapes)
                    st.plotly_chart(fig, width="stretch")
                st.caption(
                    "Translucent columns mark axes excluded from scoring "
                    "(SAEL direction = `unknown` → not testable)."
                )

            else:  # Radar overlay
                traces = []
                for i, (cid, vals) in enumerate(zip(selected, rows)):
                    traces.append({
                        "name": label_map.get(cid, cid),
                        "values": [abs(v) for v in vals],
                        "color": OVERLAY_COLORS[i % len(OVERLAY_COLORS)],
                        "fill_alpha": 0.22,
                    })
                fig = radar_figure(
                    traces,
                    title="|ΔBSV| magnitude per axis (canonical order)",
                    radial_max=vmax, height=480,
                )
                st.plotly_chart(fig, width="stretch")
                st.caption("Radar shows |ΔBSV| magnitude; signs are visible in bar/heatmap views.")

            st.markdown("---")
            st.markdown("### Interpretation summary")
            summary_cols = [
                "contrast_id", "display_name", "sael_status", "sael_overall_confidence",
                "n_testable_axes", "testable_axes",
                "confidence_weighted_score",
                "n_high_conf_agree", "n_moderate_conf_agree", "n_low_conf_agree",
                "n_disagree", "overall_label",
            ]
            view = conds[conds["contrast_id"].isin(selected)][summary_cols].rename(columns={
                "display_name": "Condition",
                "sael_status": "SAEL status",
                "sael_overall_confidence": "SAEL conf.",
                "n_testable_axes": "# testable",
                "testable_axes": "Testable axes",
                "confidence_weighted_score": "Score",
                "n_high_conf_agree": "Agree (H)",
                "n_moderate_conf_agree": "Agree (M)",
                "n_low_conf_agree": "Agree (L)",
                "n_disagree": "Disagree",
                "overall_label": "Outcome",
            })

            def _pill(label: str) -> str:
                cls = "pill-ok" if label == "pass" else ("pill-bad" if label == "inconsistent" else "pill-warn")
                return f'<span class="pill {cls}">{label}</span>'

            for _, r in view.iterrows():
                st.markdown(
                    f"<div class='box' style='margin-bottom:10px;'>"
                    f"<b style='color:#F8FAFC; font-size:1.02rem;'>{r['Condition']}</b> "
                    f"{_pill(r['Outcome'])} "
                    f"<span style='color:#94A3B8;'> · SAEL <code style='color:#FDE68A;'>"
                    f"{r['SAEL status']}</code> ({r['SAEL conf.']}) · "
                    f"score <code style='color:#FDE68A;'>{r['Score']:+.3f}</code> · "
                    f"testable axes: <b style='color:#E2E8F0;'>{r['# testable']}</b></span>"
                    f"<div style='color:#CBD5E1; font-size:0.85rem; margin-top:6px;'>"
                    f"Testable axes: {r['Testable axes']}"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )


# ──────────────────────────────────────────────────────────────────────
# TAB 4 — REGRESSION / DOSE-RESPONSE
# ──────────────────────────────────────────────────────────────────────

with tab4:
    st.subheader("Dose-response explorer — Ergothioneine titration")

    per_conc = load_erg_per_conc()
    long_df = load_erg_dose_long()

    if per_conc.empty or long_df.empty:
        st.warning("Ergothioneine assets missing.")
    else:
        concs = per_conc["concentration_uM"].tolist()
        c_min, c_max = min(concs), max(concs)

        default_primary = "redox_metabolite" if "redox_metabolite" in long_df["axis"].values \
            else BSV_COMPONENTS[0]
        primary_axis = st.selectbox(
            "Primary axis for dose curve",
            BSV_COMPONENTS,
            index=BSV_COMPONENTS.index(default_primary),
            format_func=lambda a: AXIS_LABELS[a],
            key="reg_primary_axis",
        )

        slider_c = st.select_slider(
            "Concentration",
            options=concs, value=concs[0],
            format_func=format_conc, key="reg_slider",
        )

        baseline_row = per_conc[per_conc["concentration_uM"] == 0.0].iloc[0]
        chosen_row = per_conc[per_conc["concentration_uM"] == slider_c].iloc[0]

        left, right = st.columns([1.08, 1.0])

        # Morphing radar + baseline overlay
        with left:
            base_vals = canonical_bsv_from_cols(baseline_row, prefix="bsv_")
            cur_vals = canonical_bsv_from_cols(chosen_row, prefix="bsv_")

            all_bsv_cols = [f"bsv_{c}" for c in BSV_COMPONENTS]
            r_max = float(per_conc[all_bsv_cols].values.max()) * 1.18

            traces = [
                {"name": "baseline (0.0 µM)", "values": base_vals,
                 "color": "#94A3B8", "fill_alpha": 0.14},
                {"name": format_conc(slider_c), "values": cur_vals,
                 "color": "#F87171", "fill_alpha": 0.34},
            ]
            fig = radar_figure(
                traces,
                title=f"BSV radar — {format_conc(slider_c)} vs baseline",
                radial_max=r_max, height=460,
            )
            st.plotly_chart(fig, width="stretch")

        with right:
            delta_vals = [float(chosen_row[f"delta_bsv_{c}"]) for c in BSV_COMPONENTS]
            abs_max = max(
                0.005,
                per_conc[[f"delta_bsv_{c}" for c in BSV_COMPONENTS]].abs().values.max() * 1.08,
            )
            fig = bsv_bar_figure(
                delta_vals,
                title=f"ΔBSV — {format_conc(slider_c)} − 0.0 µM",
                y_label="ΔBSV",
                signed=True, height=460,
                y_range=(-abs_max, abs_max),
            )
            st.plotly_chart(fig, width="stretch")

        st.markdown("---")

        # Dose-response line chart + slider marker
        sub_primary = long_df[long_df["axis"] == primary_axis].sort_values("concentration_uM")
        axis_color = AXIS_COLORS[primary_axis]
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=sub_primary["concentration_uM"], y=sub_primary["delta_bsv"],
            mode="lines+markers",
            name=f"Δ {AXIS_LABELS[primary_axis]}",
            line=dict(color=axis_color, width=2.8),
            marker=dict(size=10, color=axis_color,
                        line=dict(color="#0B1220", width=1)),
            hovertemplate="c = %{x:.1f} µM<br>ΔBSV = %{y:+.4f}<extra></extra>",
        ))
        fig_curve.add_trace(go.Scatter(
            x=per_conc["concentration_uM"],
            y=per_conc["commit_axes"].astype(float) / len(BSV_COMPONENTS),
            mode="lines+markers", name="Commit fraction (|Δ|>0.005)",
            line=dict(color="#F1F5F9", dash="dot", width=1.8),
            marker=dict(size=6, color="#F1F5F9"),
            yaxis="y2",
            hovertemplate="c = %{x:.1f} µM<br>commit = %{y:.2f}<extra></extra>",
        ))
        fig_curve.add_vline(
            x=slider_c,
            line=dict(color="#60A5FA", width=1.6, dash="dash"),
        )
        fig_curve.add_annotation(
            x=slider_c, y=1.05, yref="paper",
            showarrow=False, text=format_conc(slider_c),
            font=dict(size=12, color="#60A5FA"),
        )
        apply_dark_theme(
            fig_curve,
            title=f"Dose response — Δ {AXIS_LABELS[primary_axis]} and commit fraction",
            height=380, margin=dict(l=62, r=70, t=50, b=60),
        )
        fig_curve.update_xaxes(
            title="Ergothioneine concentration (µM)",
            range=[c_min - 0.1, c_max + 0.1],
        )
        fig_curve.update_yaxes(
            title=f"Δ BSV ({AXIS_LABELS[primary_axis]})",
            zeroline=True, zerolinecolor="rgba(148,163,184,0.55)",
        )
        # secondary axis (y2) — style manually (update_yaxes doesn't touch it)
        fig_curve.update_layout(
            yaxis2=dict(
                title=dict(text="Commit fraction",
                           font=dict(color=TEXT_PRIMARY, size=12)),
                overlaying="y", side="right",
                range=[0, 1], showgrid=False,
                tickfont=dict(color=TEXT_SECONDARY, size=11),
                linecolor="#64748B",
            ),
            hovermode="x unified",
        )
        st.plotly_chart(fig_curve, width="stretch")

        # Interpretation
        primary_delta = float(chosen_row[f"delta_bsv_{primary_axis}"])
        commit_n = int(chosen_row["commit_axes"])
        lod_note = (
            "near-baseline (below commit threshold)" if abs(primary_delta) < 0.005
            else ("below routine LOD" if abs(primary_delta) < 0.01
                  else "above routine LOD")
        )
        saturation_note = ""
        max_primary = float(sub_primary["delta_bsv"].abs().max())
        if abs(primary_delta) > 0.85 * max_primary and slider_c >= 0.5 * c_max:
            saturation_note = "; approaching saturation"

        kc1, kc2, kc3, kc4 = st.columns(4)
        kc1.metric("Concentration", format_conc(slider_c))
        kc2.metric(f"Δ {AXIS_LABELS[primary_axis]}", f"{primary_delta:+.4f}")
        kc3.metric("Commit fraction", f"{commit_n}/{len(BSV_COMPONENTS)}")
        kc4.metric("LOD status", lod_note.split(" (")[0])

        st.markdown(
            f'<div class="box">'
            f"<b style='color:#F8FAFC;'>Interpretation.</b> "
            f"At {format_conc(slider_c)}, "
            f"Δ{AXIS_LABELS[primary_axis]} = <code>{primary_delta:+.4f}</code>, "
            f"with <b style='color:#F8FAFC;'>{commit_n} of {len(BSV_COMPONENTS)}</b> BSV axes committing "
            f"(|Δ| &gt; 0.005). Response is {lod_note}{saturation_note}.  "
            "This view plots ΔBSV (perturbed minus 0 µM baseline) in representation "
            "space only; spectra are not shown in the calibration/regression tabs."
            "</div>",
            unsafe_allow_html=True,
        )

st.markdown("")
st.caption(
    "GAIRA demo v2 · Plotly-based · dark-theme readability pass. "
    "v1 app preserved at streamlit_apps/gaira_demo/."
)
