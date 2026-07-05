"""GAIRA — demo v3 (semantic cleanup · evidence-layer clarity · selector hygiene).

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/streamlit run streamlit_apps/gaira_demo_v3/gaira_demo_v3.py \\
        --theme.base dark --theme.backgroundColor "#0B1220" \\
        --theme.secondaryBackgroundColor "#111827" \\
        --theme.primaryColor "#60A5FA" --theme.textColor "#F1F5F9"

v1 and v2 apps are preserved untouched under their own folders.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from helpers import (
    AXIS_COLORS, AXIS_LABELS, BSV_COMPONENTS,
    OVERLAY_COLORS, TEXT_PRIMARY, TEXT_SECONDARY,
    add_atlas_band_shading_for_axes, apply_dark_theme, atlas_ruler_figure,
    bsv_bar_figure, canonical_bsv, canonical_bsv_from_cols,
    delta_heatmap_figure, dominant_axes_for_family, dominant_axes_for_molecules,
    format_conc,
    load_atlas_explorer, load_axis_coverage, load_calibration_conditions,
    load_calibration_delta_bsv, load_calibration_metadata,
    load_corpus_summary, load_erg_dose_long, load_erg_per_conc,
    load_family_counts, load_grounding_layer_summary, load_literature_evidence,
    load_molecule_bsv, load_molecule_index, load_molecule_spectra,
    load_regression_registry, pipeline_diagram_figure, radar_figure,
    spectra_overlay_figure,
)


st.set_page_config(
    page_title="GAIRA v3 — demo",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────
# Page CSS (dark)
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

    [data-testid="stMetric"] {
        background: rgba(17, 24, 39, 0.6);
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 10px; padding: 10px 14px;
    }
    [data-testid="stMetricValue"] { color: #F8FAFC; }
    [data-testid="stMetricLabel"] { color: #94A3B8; }

    .stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid rgba(148,163,184,0.18); }
    .stTabs [data-baseweb="tab"] {
        background: rgba(17, 24, 39, 0.6); color: #CBD5E1;
        border-radius: 8px 8px 0 0; padding: 10px 20px;
        border: 1px solid rgba(148,163,184,0.12); border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background: #1E293B; color: #F8FAFC;
        border-color: rgba(96,165,250,0.55);
        border-bottom: 2px solid #60A5FA;
    }

    div[data-baseweb="select"] > div, input, textarea {
        background: #111827 !important; color: #F1F5F9 !important;
        border: 1px solid rgba(148,163,184,0.25) !important;
    }
    div[data-baseweb="popover"] { background: #111827 !important; color: #F1F5F9 !important; }
    div[data-testid="stExpander"] {
        background: rgba(17, 24, 39, 0.55);
        border: 1px solid rgba(148,163,184,0.18); border-radius: 8px;
    }

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
    .box code { background: rgba(148,163,184,0.12); color: #FDE68A; padding: 1px 6px; border-radius: 4px; }

    .layer-card {
        background: rgba(17, 24, 39, 0.55);
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 12px; padding: 14px 16px 16px 16px;
        min-height: 180px;
    }
    .layer-head {
        color: #60A5FA; font-weight: 700; letter-spacing: 0.02em;
        font-size: 0.78rem; text-transform: uppercase;
    }
    .layer-title { color: #F8FAFC; font-weight: 700; font-size: 1.05rem; margin: 4px 0 10px 0; }
    .layer-row { display: flex; justify-content: space-between; margin: 4px 0;
                 color: #CBD5E1; font-size: 0.9rem; }
    .layer-row b { color: #F1F5F9; }
    .layer-note { color: #94A3B8; font-size: 0.82rem; margin-top: 10px; line-height: 1.45; }
    .layer-datasets { color: #E2E8F0; font-size: 0.82rem; margin-top: 6px; }

    hr { border-color: rgba(148,163,184,0.22); }
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
    "Domain-aware evidence engine. Spectra are mixtures, not fingerprints; "
    "interpretation is region-based, multi-axis, and ambiguity-tracked."
    '<span style="color:#60A5FA; margin-left:10px;">· v3 · evidence-layer clarity</span>'
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
    # A. concise intro
    st.markdown(
        '<div class="box" style="margin-bottom: 14px;">'
        "<b style='color:#F8FAFC;'>What GAIRA does.</b> "
        "Given a Raman / SERS spectrum, GAIRA projects it onto an 8-axis "
        "Biochemical State Vector (BSV). When a baseline exists, it reports "
        "ΔBSV against that baseline. Assignments are region-based, axis-level, "
        "and kept multi-candidate where the atlas is ambiguous — not exact "
        "molecule calls."
        "</div>",
        unsafe_allow_html=True,
    )

    # B. current pipeline diagram
    fig_pipe = pipeline_diagram_figure()
    st.plotly_chart(fig_pipe, width="stretch")

    st.markdown("---")

    # C. Grounding corpus overview — evidence layers
    st.subheader("Grounding corpus overview")
    st.markdown(
        '<div class="caption-muted" style="margin-bottom: 8px;">'
        "GAIRA's grounding is made up of three distinct evidence layers. "
        "Each layer has its own provenance and limitations."
        "</div>",
        unsafe_allow_html=True,
    )

    layers = load_grounding_layer_summary()
    if not layers.empty:
        cols = st.columns(3)
        layer_order = ["pure_molecule", "literature_linked", "atlas"]
        layer_badge = {"pure_molecule": "LAYER 1", "literature_linked": "LAYER 2", "atlas": "LAYER 3"}
        for col, key in zip(cols, layer_order):
            row = layers[layers["layer"] == key].iloc[0]
            with col:
                st.markdown(
                    f"<div class='layer-card'>"
                    f"<div class='layer-head'>{layer_badge[key]}</div>"
                    f"<div class='layer-title'>{row['title']}</div>"
                    f"<div class='layer-row'><span>{row['metric_a_label']}</span><b>{row['metric_a_value']}</b></div>"
                    f"<div class='layer-row'><span>{row['metric_b_label']}</span><b>{row['metric_b_value']}</b></div>"
                    f"<div class='layer-row'><span>{row['metric_c_label']}</span><b>{row['metric_c_value']}</b></div>"
                    f"<div class='layer-datasets'><b style='color:#94A3B8;'>Sources:</b> {row['datasets']}</div>"
                    f"<div class='layer-note'>{row['note']}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # Pure-molecule family breakdown + literature source kinds
    st.markdown("")
    bcol1, bcol2 = st.columns([1.1, 1.0])
    with bcol1:
        fams = load_family_counts()
        if not fams.empty:
            fig = go.Figure(go.Bar(
                x=fams["n_molecules"], y=fams["family"],
                orientation="h",
                marker=dict(color="#60A5FA",
                             line=dict(color="rgba(255,255,255,0.2)", width=0.4)),
                hovertemplate="<b>%{y}</b><br>%{x} molecules<extra></extra>",
            ))
            apply_dark_theme(
                fig, title="Pure-molecule layer · family distribution",
                height=330, show_legend=False,
                margin=dict(l=130, r=30, t=44, b=38),
            )
            fig.update_xaxes(title="Pure molecules")
            fig.update_yaxes(title="", autorange="reversed", automargin=True)
            st.plotly_chart(fig, width="stretch")

    with bcol2:
        lit = load_literature_evidence()
        if not lit.empty:
            kinds = (
                lit.groupby("kind")
                .agg(n_sources=("source_id", "nunique"),
                     total_band_support=("n_bands_supported", "sum"))
                .reset_index()
            )
            kinds["kind_label"] = kinds["kind"].map({
                "literature_paper": "Literature papers",
                "core_reference": "Core references (src_001–005)",
            }).fillna(kinds["kind"])
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=kinds["kind_label"], y=kinds["n_sources"],
                name="Unique sources",
                marker=dict(color="#F87171",
                             line=dict(color="rgba(255,255,255,0.2)", width=0.4)),
                hovertemplate="<b>%{x}</b><br>Unique sources: %{y}<extra></extra>",
            ))
            fig.add_trace(go.Bar(
                x=kinds["kind_label"], y=kinds["total_band_support"],
                name="Band-support count",
                marker=dict(color="#C084FC",
                             line=dict(color="rgba(255,255,255,0.2)", width=0.4)),
                hovertemplate="<b>%{x}</b><br>Band-support: %{y}<extra></extra>",
            ))
            apply_dark_theme(
                fig, title="Literature-linked layer · source kinds",
                height=330,
                margin=dict(l=50, r=30, t=44, b=72),
            )
            fig.update_layout(barmode="group")
            fig.update_xaxes(title="", tickangle=-15)
            fig.update_yaxes(title="Count")
            st.plotly_chart(fig, width="stretch")

    st.markdown(
        '<div class="box" style="margin-top: 2px;">'
        "<b style='color:#FBBF24;'>Atlas coverage — honest note.</b> "
        "Atlas coverage is current and uneven across axes. "
        "Purine, Aromatic-AA and Membrane-lipid axes are best supported; "
        "Redox and Pyrimidine bands lean thin and ambiguous. "
        "This will be refined in future work."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # E. Atlas explorer
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
            help="Bands where more than one primary axis is a candidate (multi-assignment).",
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
    atlas = load_atlas_explorer()

    if mol_index.empty or mol_bsv.empty or mol_spec.empty:
        st.warning("Grounding assets missing. Run build_demo_assets.py in v1.")
    else:
        # Dedupe components (same component can appear under multiple ids).
        # Pick the first id per component for display.
        first_by_component = (
            mol_index.sort_values("id").drop_duplicates("component")
            [["id", "component", "family", "type"]]
        )

        # ── Family filter with session-state hygiene ──────────────
        fam_opts = ["(all families)"] + sorted(first_by_component["family"].dropna().unique().tolist())
        fam_pick = st.selectbox("Family filter", fam_opts, index=0, key="grd_family_v3")

        if fam_pick == "(all families)":
            pool = first_by_component
        else:
            pool = first_by_component[first_by_component["family"] == fam_pick]
        pool_names = pool["component"].tolist()

        # Family-scoped key: switching family creates a fresh multiselect instance
        # so selection from a previous family cannot leak as "invalid residual".
        mol_key = f"grd_molecules_v3__{fam_pick}"
        # Only seed defaults on the first entry for this family.
        if mol_key not in st.session_state:
            seed = [n for n in ["L-ergothioneine", "Hypoxanthine", "Uric acid"] if n in pool_names]
            if not seed and pool_names:
                seed = pool_names[: min(2, len(pool_names))]
            st.session_state[mol_key] = seed

        selected = st.multiselect(
            "Molecules (up to 5)",
            pool_names,
            max_selections=5,
            key=mol_key,
        )
        selected_unique = list(dict.fromkeys(selected))  # stable dedupe

        if not selected_unique:
            st.info("Pick at least one molecule above.")
        else:
            sel_rows = first_by_component[first_by_component["component"].isin(selected_unique)]
            name_to_id = dict(zip(sel_rows["component"], sel_rows["id"]))
            sel_ids = [name_to_id[n] for n in selected_unique if n in name_to_id]

            # Molecule-aware band highlights (top-2 dominant axes per molecule)
            highlight_axes = dominant_axes_for_molecules(
                mol_bsv, selected_unique, top_k=2,
            )
            # If only a family is meaningful (e.g. all selected share family), enrich
            if fam_pick != "(all families)":
                highlight_axes = list(dict.fromkeys(
                    highlight_axes + dominant_axes_for_family(mol_bsv, fam_pick, top_k=2)
                ))

            # Spectra — one clean trace per selected molecule
            spectra_plot = []
            for name, mol_id in zip(selected_unique, sel_ids):
                sub = mol_spec[mol_spec["id"] == mol_id].sort_values("wavenumber")
                if not sub.empty:
                    spectra_plot.append(
                        (name, sub["wavenumber"].to_numpy(),
                         sub["intensity_norm"].to_numpy())
                    )

            axes_tag = ", ".join(AXIS_LABELS[a] for a in highlight_axes) or "—"
            fig_spec = spectra_overlay_figure(
                spectra_plot,
                title=f"Processed spectra · atlas bands highlighted for: {axes_tag}",
                height=430,
            )
            fig_spec = add_atlas_band_shading_for_axes(
                fig_spec, atlas, axes=highlight_axes, alpha=0.16, anchor_only=True,
            )
            st.plotly_chart(fig_spec, width="stretch")

            st.caption(
                "Shaded stripes mark the atlas **anchor** bands whose primary axis "
                "matches the selected molecules' dominant BSV axes."
            )

            st.markdown("")

            # Per-molecule BSV bars — stacked vertically, no averaging
            st.markdown("### Per-molecule BSV contribution")
            for name, mol_id in zip(selected_unique, sel_ids):
                brow = mol_bsv[mol_bsv["id"] == mol_id]
                if brow.empty:
                    continue
                vals = canonical_bsv(brow.iloc[0])
                bar_max = max(0.05, max(vals) * 1.18)
                fig_bar = bsv_bar_figure(
                    vals,
                    title=f"{name}",
                    y_label="BSV (pure-molecule · no baseline)",
                    signed=False, height=300,
                    y_range=(0, bar_max),
                )
                st.plotly_chart(fig_bar, width="stretch")

            # BSV radar — one trace per molecule
            st.markdown("### BSV radar (canonical axis order)")
            traces = []
            sel_bsv_rows = mol_bsv[mol_bsv["id"].isin(sel_ids)]
            radial_max = max(
                0.15, float(sel_bsv_rows[BSV_COMPONENTS].values.max()) * 1.18,
            )
            for i, (name, mol_id) in enumerate(zip(selected_unique, sel_ids)):
                brow = sel_bsv_rows[sel_bsv_rows["id"] == mol_id]
                if brow.empty:
                    continue
                vals = canonical_bsv(brow.iloc[0])
                color = OVERLAY_COLORS[i % len(OVERLAY_COLORS)]
                traces.append({
                    "name": name, "values": vals, "color": color,
                    "fill_alpha": 0.22 if len(selected_unique) > 1 else 0.34,
                })
            fig_radar = radar_figure(
                traces, title=None, radial_max=radial_max, height=460,
            )
            st.plotly_chart(fig_radar, width="stretch")

            st.markdown("")
            st.markdown("### Context")
            ctx = (
                sel_bsv_rows[["id", "component", "family", "dominant_axis", "dominant_weight"]]
                .drop_duplicates("component")
                .merge(
                    first_by_component[["id", "type"]], on="id", how="left"
                )
            )
            ctx = ctx[[
                "component", "family", "type", "dominant_axis", "dominant_weight",
            ]].rename(columns={
                "component": "Molecule", "family": "Family", "type": "Type",
                "dominant_axis": "Dominant axis", "dominant_weight": "Weight",
            })
            ctx["Weight"] = ctx["Weight"].map(lambda v: f"{v:.3f}")
            ctx["Dominant axis"] = ctx["Dominant axis"].map(lambda a: AXIS_LABELS.get(a, a))
            st.dataframe(ctx, width="stretch", hide_index=True)
            st.caption(
                "**Caveat.** GAIRA represents mixtures, not fingerprints. "
                "Dominant-axis labels describe spectral projection only, "
                "not exact molecule identity."
            )


# ──────────────────────────────────────────────────────────────────────
# TAB 3 — CALIBRATION
# ──────────────────────────────────────────────────────────────────────

with tab3:
    st.subheader("Calibration — representation-space response")

    conds = load_calibration_conditions()
    deltas = load_calibration_delta_bsv()
    meta = load_calibration_metadata()

    if conds.empty or deltas.empty or meta.empty:
        st.warning("Calibration assets missing. Run build_v3_assets.py.")
    else:
        label_map = dict(zip(meta["contrast_id"], meta["rich_label"]))
        meta_by_id = meta.set_index("contrast_id")

        c1, c2 = st.columns([1.3, 1.0])
        with c1:
            selected = st.multiselect(
                "Calibration contrasts (compare up to 4)",
                meta["contrast_id"].tolist(),
                default=[meta["contrast_id"].iloc[0], meta["contrast_id"].iloc[1]],
                format_func=lambda cid: label_map.get(cid, cid),
                max_selections=4, key="cal_contrasts_v3",
            )
        with c2:
            view_mode = st.radio(
                "ΔBSV view",
                ["Bar per contrast", "Heatmap across contrasts", "Radar overlay"],
                horizontal=False, index=0, key="cal_view_mode_v3",
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
                    matrix, row_labels=labels,
                    col_labels=[AXIS_LABELS[c] for c in BSV_COMPONENTS],
                    title="ΔBSV · observed (perturbed − baseline)",
                    height=max(260, 60 * len(rows) + 120), vmax=vmax,
                )
                st.plotly_chart(fig, width="stretch")

            elif view_mode == "Bar per contrast":
                for cid, vals in zip(selected, rows):
                    m_row = meta_by_id.loc[cid]
                    st.markdown(
                        f"**<span style='color:#F8FAFC;'>{m_row['rich_label']}</span>**  "
                        f"<span style='color:#94A3B8;'>"
                        f"({m_row['baseline_label']} → {m_row['perturbed_label']})"
                        f"</span>",
                        unsafe_allow_html=True,
                    )
                    fig = bsv_bar_figure(
                        vals, title=None,
                        y_label=f"ΔBSV  ({m_row['perturbed_label']} − {m_row['baseline_label']})",
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
                    title="|ΔBSV| magnitude · perturbed vs baseline",
                    radial_max=vmax, height=480,
                )
                st.plotly_chart(fig, width="stretch")
                st.caption(
                    "Radar shows |ΔBSV| magnitude (direction-free). "
                    "Signs are visible in bar or heatmap views."
                )

            st.markdown("---")
            st.markdown("### Condition metadata")

            def _pill(label: str) -> str:
                cls = ("pill-ok" if label == "pass"
                       else ("pill-bad" if label == "inconsistent" else "pill-warn"))
                return f'<span class="pill {cls}">{label}</span>'

            cond_outcome = dict(zip(conds["contrast_id"], conds["overall_label"]))
            cond_score = dict(zip(conds["contrast_id"], conds["confidence_weighted_score"]))

            for cid in selected:
                m_row = meta_by_id.loc[cid]
                outcome = cond_outcome.get(cid, "—")
                score = cond_score.get(cid, float("nan"))
                caveat_html = (
                    f"<div style='color:#FBBF24; margin-top:8px; font-size:0.86rem;'>"
                    f"⚑ {m_row['caveat']}</div>"
                ) if m_row["caveat"] else ""
                st.markdown(
                    f"<div class='box' style='margin-bottom:10px;'>"
                    f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                    f"<b style='color:#F8FAFC; font-size:1.04rem;'>{m_row['rich_label']}</b>"
                    f"<span>{_pill(outcome)}"
                    f"<span style='color:#94A3B8; margin-left:6px;'>"
                    f"score <code>{score:+.3f}</code></span></span>"
                    f"</div>"
                    f"<div style='color:#CBD5E1; font-size:0.88rem; margin-top:10px;'>"
                    f"<b style='color:#94A3B8;'>Baseline:</b> {m_row['baseline_label']} · "
                    f"<b style='color:#94A3B8;'>Perturbed:</b> {m_row['perturbed_label']}"
                    f"</div>"
                    f"<div style='display:grid; grid-template-columns: repeat(3, 1fr); "
                    f"gap:6px 20px; margin-top:8px; color:#CBD5E1; font-size:0.86rem;'>"
                    f"<div><b style='color:#94A3B8;'>Analyte:</b> {m_row['analyte']}</div>"
                    f"<div><b style='color:#94A3B8;'>Matrix:</b> {m_row['matrix']}</div>"
                    f"<div><b style='color:#94A3B8;'>Substrate:</b> {m_row['substrate']}</div>"
                    f"<div><b style='color:#94A3B8;'>Perturbation:</b> {m_row['perturbation_type']}</div>"
                    f"<div><b style='color:#94A3B8;'>Concentration:</b> {m_row['concentration_info']}</div>"
                    f"<div><b style='color:#94A3B8;'>Behavior class:</b> {m_row['behavior_class']}</div>"
                    f"</div>"
                    f"{caveat_html}"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ──────────────────────────────────────────────────────────────────────
# TAB 4 — REGRESSION (narrow to supported ordered series only)
# ──────────────────────────────────────────────────────────────────────

with tab4:
    st.subheader("Regression · ordered dose-response")

    reg_reg = load_regression_registry()
    supported = reg_reg[reg_reg["supported"] == True]

    st.markdown(
        '<div class="caption-muted" style="margin-bottom: 10px;">'
        "This tab only exposes <b style='color:#F1F5F9;'>true ordered series</b> "
        "wired through the GAIRA BSV pipeline. Two-point comparisons "
        "(e.g. uricase untreated vs treated, single-level spikes) stay in the "
        "Calibration tab where they belong."
        "</div>",
        unsafe_allow_html=True,
    )

    if supported.empty:
        st.warning("No regression-ready ordered series.")
    else:
        dataset_id = st.selectbox(
            "Dataset",
            supported["dataset_id"].tolist(),
            format_func=lambda did: dict(zip(supported["dataset_id"], supported["display_name"]))[did],
            key="reg_dataset_v3",
        )

        if dataset_id == "ergothioneine_titration":
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
                    key="reg_primary_axis_v3",
                )

                slider_c = st.select_slider(
                    "Concentration",
                    options=concs, value=concs[0],
                    format_func=format_conc, key="reg_slider_v3",
                )

                baseline_row = per_conc[per_conc["concentration_uM"] == 0.0].iloc[0]
                chosen_row = per_conc[per_conc["concentration_uM"] == slider_c].iloc[0]

                left, right = st.columns([1.08, 1.0])

                with left:
                    base_vals = canonical_bsv_from_cols(baseline_row, prefix="bsv_")
                    cur_vals = canonical_bsv_from_cols(chosen_row, prefix="bsv_")
                    all_bsv_cols = [f"bsv_{c}" for c in BSV_COMPONENTS]
                    r_max = float(per_conc[all_bsv_cols].values.max()) * 1.18
                    traces = [
                        {"name": "baseline · 0.0 µM", "values": base_vals,
                         "color": "#94A3B8", "fill_alpha": 0.14},
                        {"name": f"spiked · {format_conc(slider_c)}", "values": cur_vals,
                         "color": "#F87171", "fill_alpha": 0.34},
                    ]
                    fig = radar_figure(
                        traces,
                        title=f"BSV radar · baseline vs {format_conc(slider_c)}",
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
                        title=f"ΔBSV · {format_conc(slider_c)} − 0.0 µM baseline",
                        y_label="ΔBSV", signed=True, height=460,
                        y_range=(-abs_max, abs_max),
                    )
                    st.plotly_chart(fig, width="stretch")

                st.markdown("---")

                # Dose-response curve
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
                    title=f"Dose response · Δ{AXIS_LABELS[primary_axis]} + commit fraction",
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
                fig_curve.update_layout(
                    yaxis2=dict(
                        title=dict(text="Commit fraction",
                                   font=dict(color=TEXT_PRIMARY, size=12)),
                        overlaying="y", side="right", range=[0, 1],
                        showgrid=False, linecolor="#64748B",
                        tickfont=dict(color=TEXT_SECONDARY, size=11),
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
                    f"At <code>{format_conc(slider_c)}</code>, "
                    f"Δ{AXIS_LABELS[primary_axis]} = <code>{primary_delta:+.4f}</code>, "
                    f"with <b style='color:#F8FAFC;'>{commit_n} of {len(BSV_COMPONENTS)}</b> BSV axes committing "
                    f"(|Δ| &gt; 0.005). Response is {lod_note}{saturation_note}. "
                    "Representation-space view only; spectra are not shown in this tab."
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # Always-visible honest audit block
    with st.expander("Why other datasets are not in this tab", expanded=False):
        unsupported = reg_reg[reg_reg["supported"] == False][
            ["display_name", "n_levels", "reason_if_unsupported", "notes"]
        ].rename(columns={
            "display_name": "Dataset",
            "n_levels": "# levels",
            "reason_if_unsupported": "Why excluded",
            "notes": "Notes",
        })
        st.dataframe(unsupported, width="stretch", hide_index=True)
        st.caption(
            "The regression tab is intentionally narrow. "
            "Endpoint comparisons and single-level spikes live in the Calibration tab."
        )


st.markdown("")
st.caption(
    "GAIRA demo v3 · evidence-layer clarity + selector hygiene. "
    "v1 and v2 are preserved under their own folders."
)
