"""GAIRA — polished demo (4 tabs).

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/streamlit run streamlit_apps/gaira_demo/gaira_demo.py

Tabs:
    1. Methods / Pipeline       — hero figure, pipeline, corpus + atlas explorer
    2. Grounding Results        — pure-molecule spectra + BSV (not ΔBSV)
    3. Calibration Results      — ΔBSV only, canonical order, no spectra
    4. Regression / Dose-response Explorer — slider-driven radar + Δ-axis bars

All charts are Plotly (graph_objects). All axes follow the GAIRA canonical order.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from helpers import (
    AXIS_COLORS, AXIS_LABELS, BSV_COMPONENTS,
    add_atlas_band_shading, bsv_bar_figure, canonical_bsv, canonical_bsv_from_cols,
    canonical_delta, delta_heatmap_figure, format_conc,
    load_atlas_explorer, load_axis_coverage, load_calibration_conditions,
    load_calibration_delta_bsv, load_corpus_summary, load_erg_dose_long,
    load_erg_mean_spectra, load_erg_per_conc, load_family_counts,
    load_molecule_bsv, load_molecule_index, load_molecule_spectra,
    radar_figure, spectra_overlay_figure,
)


st.set_page_config(
    page_title="GAIRA — demo",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1300px; }
    h1, h2, h3 { color: #0F172A; }
    .caption-muted { color: #475569; font-size: 0.95rem; line-height: 1.5; }
    .pill {
        display: inline-block;
        padding: 2px 10px; margin-right: 6px;
        border-radius: 999px;
        background: #EEF2FF; color: #3730A3;
        font-size: 0.78rem; font-weight: 600;
    }
    .pill-ok { background: #ECFDF5; color: #047857; }
    .pill-warn { background: #FEF3C7; color: #92400E; }
    .pill-bad { background: #FEE2E2; color: #B91C1C; }
    .box {
        background: #F8FAFC; border: 1px solid #E2E8F0;
        border-radius: 10px; padding: 14px 18px; margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("GAIRA — biochemical reasoning from Raman / SERS")
st.markdown(
    '<div class="caption-muted">'
    "A domain-aware, uncertainty-aware evidence engine. "
    "Spectra are treated as mixtures, not fingerprints. "
    "Interpretation is band-region aware, multi-axis, and ambiguity-tracked."
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

    hero_path = Path(__file__).resolve().parents[2] / "figures" / "gaira_three_phase_master_figure.png"
    pipe_path = Path(__file__).resolve().parents[2] / "figures" / "phase2_bsv_pipeline_figure.png"
    bsv_vs_peak_path = Path(__file__).resolve().parents[2] / "figures" / "phase2_bsv_vs_peak_matching_explainer.png"

    if hero_path.exists():
        st.image(str(hero_path), width="stretch",
                 caption="GAIRA three-phase system: grounding → representation (BSV) → structured evidence.")
    else:
        st.info("Hero figure not found (figures/gaira_three_phase_master_figure.png).")

    st.markdown(
        '<div class="caption-muted">'
        "<b>One-line description.</b> GAIRA projects a measured spectrum into an "
        "8-axis Biochemical State Vector (BSV), then interprets changes (ΔBSV) "
        "against a curated Raman physics atlas and a grounding corpus of pure molecules."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Stages")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            "<b>① Grounding</b><br>"
            '<span class="caption-muted">Pure molecules (RamanBioLib) and axis-linked literature fix the axis meanings.</span>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            "<b>② Raman physics atlas</b><br>"
            '<span class="caption-muted">Bands with ranges, primary axis, companions, ambiguity, locality.</span>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            "<b>③ Preprocess → window panel</b><br>"
            '<span class="caption-muted">AsLS + SG + L2 → 22 windows → 8-axis BSV.</span>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            "<b>④ Scorer · BSV · ΔBSV</b><br>"
            '<span class="caption-muted">Calibration contrasts are scored on testable axes only (R7c, SAEL).</span>',
            unsafe_allow_html=True,
        )

    if pipe_path.exists():
        with st.expander("Show BSV pipeline schematic", expanded=False):
            st.image(str(pipe_path), width="stretch")
    if bsv_vs_peak_path.exists():
        with st.expander("Show BSV vs peak-matching explainer", expanded=False):
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
            import plotly.graph_objects as go
            fig = go.Figure(go.Bar(
                x=family_counts["n_molecules"],
                y=family_counts["family"],
                orientation="h",
                marker=dict(color="#4C78A8"),
                hovertemplate="<b>%{y}</b><br>%{x} molecules<extra></extra>",
            ))
            fig.update_layout(
                template="simple_white",
                height=340,
                margin=dict(l=120, r=30, t=30, b=30),
                xaxis=dict(title="Pure molecules"),
                yaxis=dict(title="", autorange="reversed", automargin=True),
                title=dict(text="Grounding corpus by family",
                            x=0.02, xanchor="left", font=dict(size=14)),
            )
            st.plotly_chart(fig, width="stretch")

    with col_b:
        axis_cov = load_axis_coverage()
        if not axis_cov.empty:
            import plotly.graph_objects as go
            cols = [c for c in ["anchor", "secondary", "ambiguous"] if c in axis_cov.columns]
            color_for = {"anchor": "#2B8A3E", "secondary": "#4C78A8", "ambiguous": "#C0392B"}
            fig = go.Figure()
            for c in cols:
                fig.add_trace(go.Bar(
                    name=c.capitalize(),
                    x=axis_cov["primary_axis"].map(lambda a: AXIS_LABELS.get(a, a)),
                    y=axis_cov[c],
                    marker=dict(color=color_for.get(c, "#888")),
                    hovertemplate="<b>%{x}</b><br>" + c + ": %{y}<extra></extra>",
                ))
            fig.update_layout(
                template="simple_white", barmode="stack",
                height=340,
                margin=dict(l=50, r=30, t=30, b=70),
                title=dict(text="Atlas band coverage by axis",
                            x=0.02, xanchor="left", font=dict(size=14)),
                xaxis=dict(title="", tickangle=-30),
                yaxis=dict(title="Number of bands"),
                legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
            )
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

        # Band ruler: horizontal segments along wavenumber, colored by axis
        import plotly.graph_objects as go
        ruler_fig = go.Figure()
        # axis position bands stacked by primary axis row
        axes_unique = [a for a in BSV_COMPONENTS if a in view["primary_axis"].unique()]
        axis_to_row = {a: i for i, a in enumerate(axes_unique)}
        for _, row in view.iterrows():
            a = row["primary_axis"]
            if a not in axis_to_row:
                continue
            y = axis_to_row[a]
            color = AXIS_COLORS.get(a, "#888")
            ruler_fig.add_shape(
                type="rect",
                x0=row["start_cm1"], x1=row["end_cm1"],
                y0=y - 0.35, y1=y + 0.35,
                line=dict(width=0),
                fillcolor=color, opacity=0.78 if row["classification"] == "anchor"
                else (0.45 if row["classification"] == "secondary" else 0.22),
            )
            ruler_fig.add_trace(go.Scatter(
                x=[row["central_cm1"]], y=[y],
                mode="markers",
                marker=dict(size=6, color="#0F172A"),
                showlegend=False,
                hovertemplate=(
                    f"<b>{row['display_label']}</b><br>"
                    f"Range: {row['range_label']} cm⁻¹  (width {row['width_cm1']})<br>"
                    f"Classification: {row['classification']}<br>"
                    f"Candidate axes: {row['candidate_axes']}<br>"
                    f"Ambiguity: {row['ambiguity_score']:.2f} | Locality: {row['locality_score']:.2f}<br>"
                    f"Sources: {row['source_count']}<extra></extra>"
                ),
            ))
        ruler_fig.update_layout(
            template="simple_white",
            height=max(220, 56 * max(1, len(axes_unique)) + 60),
            margin=dict(l=130, r=30, t=30, b=50),
            xaxis=dict(
                title="Raman shift (cm⁻¹)",
                range=[440, 3100], showgrid=True, gridcolor="#F1F5F9",
            ),
            yaxis=dict(
                tickmode="array",
                tickvals=list(axis_to_row.values()),
                ticktext=[AXIS_LABELS[a] for a in axes_unique],
                range=[-0.7, len(axes_unique) - 0.3],
                automargin=True,
            ),
            title=dict(text="Band ruler (hover for details)",
                        x=0.02, xanchor="left", font=dict(size=14)),
        )
        st.plotly_chart(ruler_fig, width="stretch")

        # Band table preview
        with st.expander("Band table", expanded=False):
            preview = view[[
                "display_label", "range_label", "classification",
                "candidate_axes", "ambiguity_score", "locality_score",
                "source_count", "priority_tags",
            ]].rename(columns={
                "display_label": "Band",
                "range_label": "Range (cm⁻¹)",
                "classification": "Class",
                "candidate_axes": "Candidate axes",
                "ambiguity_score": "Ambiguity",
                "locality_score": "Locality",
                "source_count": "Sources",
                "priority_tags": "Tags",
            })
            st.dataframe(preview, width="stretch", hide_index=True)


# ──────────────────────────────────────────────────────────────────────
# TAB 2 — GROUNDING RESULTS (pure molecules)
# ──────────────────────────────────────────────────────────────────────

with tab2:
    st.subheader("Pure molecule grounding")

    mol_index = load_molecule_index()
    mol_bsv = load_molecule_bsv()
    mol_spec = load_molecule_spectra()

    if mol_index.empty or mol_bsv.empty or mol_spec.empty:
        st.warning("Grounding assets missing. Run build_demo_assets.py first.")
    else:
        # Default selections bias toward interpretable metabolites
        default_names = [
            n for n in ["L-ergothioneine", "Hypoxanthine", "Uric acid"]
            if n in mol_index["component"].values
        ]
        if not default_names:
            default_names = mol_index["component"].head(2).tolist()

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

        # Build selected spectra / BSV
        if not selected:
            st.info("Pick at least one molecule above.")
        else:
            sel_rows = mol_index[mol_index["component"].isin(selected)]
            sel_ids = sel_rows["id"].tolist()

            # Spectra
            spectra_plot = []
            for mol_id in sel_ids:
                sub = mol_spec[mol_spec["id"] == mol_id].sort_values("wavenumber")
                if sub.empty:
                    continue
                name = mol_index.set_index("id").loc[mol_id, "component"]
                spectra_plot.append((str(name), sub["wavenumber"].to_numpy(),
                                     sub["intensity_norm"].to_numpy()))

            atlas = load_atlas_explorer()
            fig_spec = spectra_overlay_figure(
                spectra_plot,
                title="Processed spectra (min-max normalized; display only)",
                height=430,
            )
            fig_spec = add_atlas_band_shading(
                fig_spec,
                atlas[atlas["classification"] == "anchor"],
                alpha=0.07,
            )
            with right:
                st.caption(
                    "Gentle shaded stripes mark atlas **anchor** bands, "
                    "colored by primary axis."
                )

            st.plotly_chart(fig_spec, width="stretch")

            st.markdown("")
            colL, colR = st.columns([1.05, 1.0])

            # BSV bar (first selected if single, else mean)
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
                    bsv_vals,
                    title=bar_title,
                    y_label="BSV (pure-molecule, no baseline)",
                    signed=False, height=380,
                )
                st.plotly_chart(fig_bar, width="stretch")
                st.caption(
                    "**Pure molecules have no biological baseline → ΔBSV is not meaningful here.** "
                    "We show BSV contribution instead."
                )

            # Radar overlay across selected
            with colR:
                traces = []
                for i, mol_id in enumerate(sel_ids):
                    brow = sel_bsv_rows[sel_bsv_rows["id"] == mol_id]
                    if brow.empty:
                        continue
                    vals = canonical_bsv(brow.iloc[0])
                    from helpers import DEFAULT_LINE_COLORS
                    color = DEFAULT_LINE_COLORS[i % len(DEFAULT_LINE_COLORS)]
                    traces.append({
                        "name": str(brow.iloc[0]["component"]),
                        "values": vals,
                        "color": color,
                        "fill_alpha": 0.18 if len(sel_ids) > 1 else 0.30,
                    })
                radial_max = max(
                    0.15,
                    float(sel_bsv_rows[BSV_COMPONENTS].values.max()) * 1.15,
                )
                fig_radar = radar_figure(
                    traces,
                    title="BSV radar (canonical axis order)",
                    radial_max=radial_max,
                    height=440,
                    show_legend=True,
                )
                st.plotly_chart(fig_radar, width="stretch")

            # Context box
            with st.container():
                st.markdown("### Context")
                ctx = sel_bsv_rows.merge(
                    mol_index[["id", "type"]], on="id", how="left"
                )
                ctx = ctx[[
                    "component", "family", "type",
                    "dominant_axis", "dominant_weight",
                ]].rename(columns={
                    "component": "Molecule",
                    "family": "Family",
                    "type": "Type",
                    "dominant_axis": "Dominant axis",
                    "dominant_weight": "Weight",
                })
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
# TAB 3 — CALIBRATION RESULTS
# ──────────────────────────────────────────────────────────────────────

with tab3:
    st.subheader("Calibration — representation-space response")

    conds = load_calibration_conditions()
    deltas = load_calibration_delta_bsv()

    if conds.empty or deltas.empty:
        st.warning("Calibration assets missing. Check gaira_calibration_eval_v3 outputs.")
    else:
        label_map = dict(zip(conds["contrast_id"], conds["display_name"]))

        c1, c2 = st.columns([1.3, 1.0])
        with c1:
            selected = st.multiselect(
                "Calibration contrasts (compare up to 4)",
                conds["contrast_id"].tolist(),
                default=[conds["contrast_id"].iloc[0], conds["contrast_id"].iloc[1]],
                format_func=lambda cid: label_map.get(cid, cid),
                max_selections=4,
                key="cal_contrasts",
            )
        with c2:
            view_mode = st.radio(
                "ΔBSV view",
                ["Bar per contrast", "Heatmap across contrasts", "Radar overlay"],
                horizontal=False,
                index=0,
                key="cal_view_mode",
            )

        if not selected:
            st.info("Pick at least one calibration contrast.")
        else:
            # Build ordered matrix: rows = contrasts, cols = canonical axes
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
                    height=max(240, 60 * len(rows) + 120),
                    vmax=vmax,
                )
                st.plotly_chart(fig, width="stretch")

            elif view_mode == "Bar per contrast":
                for i, (cid, vals) in enumerate(zip(selected, rows)):
                    st.markdown(f"**{label_map.get(cid, cid)}**")
                    fig = bsv_bar_figure(
                        vals,
                        title=None,
                        y_label="ΔBSV (perturbed − control)",
                        signed=True,
                        height=320,
                    )
                    fig.update_yaxes(range=[-vmax, vmax])
                    # annotate testability: grey-out non-testable bars
                    axis_test = testable_map.get(cid, {})
                    shapes = []
                    for j, a in enumerate(BSV_COMPONENTS):
                        if not bool(axis_test.get(a, False)):
                            shapes.append(dict(
                                type="rect",
                                x0=j - 0.5, x1=j + 0.5,
                                xref="x",
                                y0=0, y1=1, yref="paper",
                                fillcolor="rgba(148,163,184,0.12)",
                                line=dict(width=0), layer="below",
                            ))
                    fig.update_layout(shapes=shapes)
                    st.plotly_chart(fig, width="stretch")
                st.caption(
                    "Light-grey columns mark axes excluded from scoring "
                    "(SAEL direction = `unknown` → not testable)."
                )

            else:  # Radar overlay
                from helpers import DEFAULT_LINE_COLORS
                traces = []
                # Need positive scale for radar → scale by |Δ|
                for i, (cid, vals) in enumerate(zip(selected, rows)):
                    traces.append({
                        "name": label_map.get(cid, cid),
                        "values": [abs(v) for v in vals],
                        "color": DEFAULT_LINE_COLORS[i % len(DEFAULT_LINE_COLORS)],
                        "fill_alpha": 0.18,
                    })
                fig = radar_figure(
                    traces,
                    title="|ΔBSV| magnitude per axis (canonical order)",
                    radial_max=vmax,
                    height=480,
                    show_legend=True,
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
                    f"**{r['Condition']}** {_pill(r['Outcome'])} · "
                    f"SAEL: `{r['SAEL status']}` ({r['SAEL conf.']}) · "
                    f"score `{r['Score']:+.3f}` · testable axes: {r['# testable']}",
                    unsafe_allow_html=True,
                )
                st.caption(f"Testable axes: {r['Testable axes']}")


# ──────────────────────────────────────────────────────────────────────
# TAB 4 — REGRESSION / DOSE-RESPONSE EXPLORER
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

        # Pick primary axis for dose-response line
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
            options=concs,
            value=concs[0],
            format_func=format_conc,
            key="reg_slider",
        )

        baseline_row = per_conc[per_conc["concentration_uM"] == 0.0].iloc[0]
        chosen_row = per_conc[per_conc["concentration_uM"] == slider_c].iloc[0]

        left, right = st.columns([1.08, 1.0])

        # Morphing radar + overlay of baseline
        with left:
            from helpers import DEFAULT_LINE_COLORS
            base_vals = canonical_bsv_from_cols(baseline_row, prefix="bsv_")
            cur_vals = canonical_bsv_from_cols(chosen_row, prefix="bsv_")

            # fixed radial_max across all concentrations
            all_bsv_cols = [f"bsv_{c}" for c in BSV_COMPONENTS]
            r_max = float(per_conc[all_bsv_cols].values.max()) * 1.15

            traces = [
                {"name": "baseline (0.0 µM)", "values": base_vals,
                 "color": "#94A3B8", "fill_alpha": 0.12},
                {"name": format_conc(slider_c), "values": cur_vals,
                 "color": "#E45756", "fill_alpha": 0.32},
            ]
            fig = radar_figure(
                traces,
                title=f"BSV radar — {format_conc(slider_c)} vs baseline",
                radial_max=r_max,
                height=450,
            )
            st.plotly_chart(fig, width="stretch")

        # Δ-axis bar plot for slider point
        with right:
            delta_vals = [float(chosen_row[f"delta_bsv_{c}"]) for c in BSV_COMPONENTS]
            abs_max = max(
                0.005,
                per_conc[[f"delta_bsv_{c}" for c in BSV_COMPONENTS]].abs().values.max() * 1.05,
            )
            fig = bsv_bar_figure(
                delta_vals,
                title=f"ΔBSV — {format_conc(slider_c)} − 0.0 µM",
                y_label="ΔBSV",
                signed=True,
                height=450,
            )
            fig.update_yaxes(range=[-abs_max, abs_max])
            st.plotly_chart(fig, width="stretch")

        st.markdown("---")

        # Dose-response line chart + slider marker
        import plotly.graph_objects as go

        sub_primary = long_df[long_df["axis"] == primary_axis].sort_values("concentration_uM")
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=sub_primary["concentration_uM"], y=sub_primary["delta_bsv"],
            mode="lines+markers",
            name=f"Δ {AXIS_LABELS[primary_axis]}",
            line=dict(color=AXIS_COLORS[primary_axis], width=2.5),
            marker=dict(size=9, color=AXIS_COLORS[primary_axis]),
            hovertemplate="c = %{x:.1f} µM<br>ΔBSV = %{y:+.4f}<extra></extra>",
        ))
        fig_curve.add_trace(go.Scatter(
            x=per_conc["concentration_uM"], y=per_conc["commit_axes"].astype(float) / len(BSV_COMPONENTS),
            mode="lines+markers", name="Commit fraction (|Δ|>0.005)",
            line=dict(color="#475569", dash="dot", width=1.8),
            marker=dict(size=6, color="#475569"),
            yaxis="y2",
            hovertemplate="c = %{x:.1f} µM<br>commit = %{y:.2f}<extra></extra>",
        ))
        fig_curve.add_vline(
            x=slider_c,
            line=dict(color="#0F172A", width=1.5, dash="dash"),
        )
        fig_curve.add_annotation(
            x=slider_c, y=1.05, yref="paper",
            showarrow=False, text=format_conc(slider_c),
            font=dict(size=11, color="#0F172A"),
        )
        fig_curve.update_layout(
            template="simple_white",
            height=360,
            margin=dict(l=60, r=60, t=40, b=50),
            title=dict(text=f"Dose response — Δ {AXIS_LABELS[primary_axis]} and commit fraction",
                        x=0.02, xanchor="left", font=dict(size=14)),
            xaxis=dict(title="Ergothioneine concentration (µM)",
                         range=[c_min - 0.1, c_max + 0.1], gridcolor="#F1F5F9"),
            yaxis=dict(title=f"Δ BSV ({AXIS_LABELS[primary_axis]})", gridcolor="#F1F5F9",
                         zeroline=True, zerolinecolor="#9CA3AF"),
            yaxis2=dict(title="Commit fraction", overlaying="y", side="right",
                          range=[0, 1], showgrid=False),
            legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_curve, width="stretch")

        # Interpretation box
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
            f"<b>Interpretation.</b> At {format_conc(slider_c)}, "
            f"Δ{AXIS_LABELS[primary_axis]} = <code>{primary_delta:+.4f}</code>, "
            f"with {commit_n} of {len(BSV_COMPONENTS)} BSV axes committing "
            f"(|Δ| &gt; 0.005). Response is {lod_note}{saturation_note}.  "
            "This view plots ΔBSV (perturbed minus 0 µM baseline) in representation "
            "space only; spectra are not shown in the calibration/regression tabs."
            "</div>",
            unsafe_allow_html=True,
        )

st.markdown("")
st.caption(
    "GAIRA demo · Plotly-based · real grounding, atlas, calibration v3, and "
    "ergothioneine titration data. See docs/streamlit_demo_data_audit.md for provenance."
)
