"""GAIRA Scientific Reasoning Demo v2 (migration-hardened).

Run (from this folder, any machine):
    streamlit run app.py
or use the portable launcher:
    ./run_demo.sh

Data locations are resolved at runtime by gaira_core/paths.py (env overrides
GAIRA_DATA_ROOT / GAIRA_LEGACY_DEMO_DATA, then candidate mounts, then a bundled
copy). The demo never hardcodes a username or a single mount name, and it shows
an explicit real-vs-placeholder banner instead of degrading silently.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from gaira_core import config as cfg
from gaira_core import data_loader as dl
from gaira_core import paths as gpaths
from gaira_core import v3_views as v3v
from gaira_core import plotting as gp
from gaira_core.motif_scoring import MOTIFS, get_motif
from gaira_core.mss_scoring import molecule_axis_contributions, score_one
from gaira_core.report_builder import build_report
from gaira_core.substrate_physics import RULES as SUBSTRATE_RULES


# ──────────────────────────────────────────────────────────────────────
# Page setup
# ──────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GAIRA — Scientific Reasoning Demo v3",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    f"""
    <style>
    .stApp {{
        background: radial-gradient(ellipse at top, #101B31 0%, {cfg.BG_PAGE} 55%);
        color: {cfg.TEXT_PRIMARY};
    }}
    .block-container {{ padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1380px; }}
    h1, h2, h3, h4 {{ color: {cfg.TITLE_COLOR}; letter-spacing: -0.01em; }}
    p, li, label {{ color: {cfg.TEXT_SECONDARY}; }}
    .caption-muted {{ color: {cfg.TEXT_MUTED}; font-size: 0.92rem; line-height: 1.55; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid rgba(148,163,184,0.18); }}
    .stTabs [data-baseweb="tab"] {{
        background: rgba(17, 24, 39, 0.6); color: {cfg.TEXT_SECONDARY};
        border-radius: 8px 8px 0 0; padding: 9px 18px;
        border: 1px solid rgba(148,163,184,0.12); border-bottom: none;
        font-size: 0.92rem;
    }}
    .stTabs [aria-selected="true"] {{
        background: #1E293B; color: {cfg.TITLE_COLOR};
        border-color: rgba(96,165,250,0.55);
        border-bottom: 2px solid #60A5FA;
    }}

    [data-testid="stMetric"] {{
        background: rgba(17, 24, 39, 0.6);
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 10px; padding: 10px 14px;
    }}
    [data-testid="stMetricValue"] {{ color: {cfg.TITLE_COLOR}; }}
    [data-testid="stMetricLabel"] {{ color: {cfg.TEXT_MUTED}; }}

    .card {{
        background: rgba(17, 24, 39, 0.62);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }}
    .card h4 {{ margin-top: 0; margin-bottom: 6px; font-size: 1.02rem; }}
    .card .muted {{ color: {cfg.TEXT_MUTED}; font-size: 0.88rem; }}

    .pill {{
        display: inline-block; padding: 2px 10px; margin-right: 6px;
        border-radius: 999px; font-size: 0.78rem; font-weight: 600;
        background: rgba(99,102,241,0.2); color: #A5B4FC;
        border: 1px solid rgba(99,102,241,0.35);
    }}
    .pill-ok   {{ background: rgba(16,185,129,0.18); color: #6EE7B7; border-color: rgba(52,211,153,0.4); }}
    .pill-warn {{ background: rgba(245,158,11,0.18); color: #FBBF24; border-color: rgba(251,191,36,0.4); }}
    .pill-bad  {{ background: rgba(239,68,68,0.2);   color: #FCA5A5; border-color: rgba(248,113,113,0.4); }}
    .pill-demo {{ background: rgba(168,85,247,0.18); color: #C4B5FD; border-color: rgba(168,85,247,0.4); }}

    div[data-baseweb="select"] > div, input, textarea {{
        background: {cfg.BG_PANEL} !important; color: {cfg.TEXT_PRIMARY} !important;
        border: 1px solid rgba(148,163,184,0.25) !important;
    }}
    div[data-testid="stExpander"] {{
        background: rgba(17, 24, 39, 0.55);
        border: 1px solid rgba(148,163,184,0.18); border-radius: 8px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div style="display: flex; align-items: baseline; gap: 16px; margin-bottom: 0.4rem;">
      <h1 style="margin: 0;">GAIRA — Scientific Reasoning Demo v3</h1>
      <span class="pill pill-ok">Global Coordinate Prototype</span>
      <span class="pill">Frozen calibration</span>
      <span class="pill">Ontology v1</span>
      <span class="pill">Cohort-invariant</span>
    </div>
    <p class="caption-muted" style="margin-bottom:0.2rem;">
      <b>V3 introduces a frozen global biochemical coordinate calibration built on GAIRA's
      transparent heuristic spectral evidence engine. It is a prototype universal coordinate
      system, not yet a trained Raman foundation model.</b>
    </p>
    <p class="caption-muted">
      GAIRA = GenAI Raman Analysis. This demo shows how Raman/SERS spectra are
      converted into interpretable biochemical state vectors using molecular
      spectral signatures, class-level motifs, substrate-aware physics, and
      explicit evidence + caveat tracking. Outputs are class-level by default;
      molecule-level claims require corroborating co-bands.
    </p>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────
# Data-source status banner (v2 migration-hardening)
#
# Makes real-vs-placeholder mode explicit at the top of the app, so the demo
# can never be mistaken for "real" when the external volume is not mounted.
# ──────────────────────────────────────────────────────────────────────

def _render_data_status_banner() -> None:
    status = gpaths.get_data_status()
    mode = status.mode
    n_real = status.real_section_count
    n_total = len(status.checks)

    if mode == "real":
        pill = "<span class='pill pill-ok'>Data mode: REAL</span>"
        root_txt = f"external volume resolved → <code>{status.data_root}</code>"
    elif mode == "degraded":
        pill = "<span class='pill pill-warn'>Data mode: DEGRADED (calibration only)</span>"
        root_txt = ("external GAIRA_DATA volume <b>not resolved</b> — calibration / "
                    "biochemical-space tabs use bundled data; adenine + biological "
                    "pilots fall back to labelled placeholders")
    else:
        pill = "<span class='pill pill-bad'>Data mode: PLACEHOLDER</span>"
        root_txt = "no data roots resolved — every section is a labelled placeholder"

    legacy_txt = {
        "bundled": "legacy CSVs: <b>bundled inside v2</b> (self-contained)",
        "repo":    "legacy CSVs: repo <code>streamlit_apps/gaira_demo/data</code>",
        "env":     "legacy CSVs: <code>$GAIRA_LEGACY_DEMO_DATA</code>",
    }.get(status.legacy_kind, "legacy CSVs: unknown")

    st.markdown(
        f"<div class='card' style='margin-top:2px;'>{pill} "
        f"<span class='pill'>{n_real}/{n_total} sections on real data</span><br>"
        f"<span class='caption-muted'>{root_txt}<br>{legacy_txt}</span></div>",
        unsafe_allow_html=True,
    )

    if mode != "real":
        st.markdown(
            "<div class='caption-muted' style='margin:-4px 0 8px 2px;'>"
            "To point the demo at your data on this machine, set "
            "<code>GAIRA_DATA_ROOT=/path/to/GAIRA_DATA</code> (must contain "
            "<code>raw/</code> and <code>processed/</code>) and relaunch. "
            "See <code>MIGRATION_HARDENING.md</code>.</div>",
            unsafe_allow_html=True,
        )

    with st.expander("Data-source detail (per section)", expanded=(mode != "real")):
        rows = [{"section": k, "source": "real" if v else "placeholder"}
                for k, v in status.checks.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(
            f"Resolved data root: {status.data_root or '—'} · "
            f"autoresearch: {'yes' if status.autoresearch_root else 'no'} · "
            f"adenine raw: {'yes' if status.adenine_dir else 'no'} · "
            f"legacy dir: {status.legacy_dir}"
        )


_render_data_status_banner()


# ──────────────────────────────────────────────────────────────────────
# Mode selector
# ──────────────────────────────────────────────────────────────────────

MODES = ("How GAIRA Works", "Calibration Evidence",
         "Global Biological Projection", "Coordinate Validation")
mode = st.radio("Mode", MODES, horizontal=True, label_visibility="collapsed")

st.markdown("<div style='margin-bottom: 6px;'></div>", unsafe_allow_html=True)


def _placeholder_badge(is_placeholder: bool) -> None:
    if is_placeholder:
        st.markdown(
            "<span class='pill pill-demo'>Demo placeholder — replace with cached GAIRA output</span>",
            unsafe_allow_html=True,
        )


def _evidence_pills(strength: str, domain_fit: str, sub_sens: str, specificity: str) -> str:
    def cls(s: str) -> str:
        return {"high": "pill-ok", "strong": "pill-ok",
                 "moderate": "pill", "moderate-high": "pill-ok",
                 "low": "pill-bad", "caution": "pill-warn",
                 "medium": "pill", "weak": "pill-bad",
                 "class-level": "pill", "candidate-level": "pill-ok",
                 "not supported": "pill-bad"}.get(s.lower(), "pill")
    return (
        f"<span class='pill {cls(strength)}'>Evidence: {strength}</span>"
        f"<span class='pill {cls(domain_fit)}'>Domain: {domain_fit}</span>"
        f"<span class='pill {cls(sub_sens)}'>Substrate: {sub_sens}</span>"
        f"<span class='pill {cls(specificity)}'>Specificity: {specificity}</span>"
    )


# ════════════════════════════════════════════════════════════════════
# MODE 1 — HOW GAIRA WORKS
# ════════════════════════════════════════════════════════════════════

if mode == "How GAIRA Works":
    tabs = st.tabs([
        "Construction Overview",
        "Grounding Corpus Map",
        "11-Axis Biochemical Space",
        "MSS / Motif Explorer",
        "Collision Viewer",
        "Physics-Aware Atlas",
        "End-to-End Workflow",
        "Biochemical Ontology v1",
        "Raw BSV → Global Coordinates",
    ])

    # ── A. Construction Overview ────────────────────────────────────
    with tabs[0]:
        st.subheader("How GAIRA was constructed")
        st.markdown(
            "<p class='caption-muted'>GAIRA converts Raman/SERS spectra into "
            "biochemical state vectors using direct spectral evidence, molecular "
            "spectral signatures, motif patterns, substrate-aware corrections, and "
            "domain-specific caveats. Every output is evidence-grounded.</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(gp.pipeline_figure(), use_container_width=True,
                          config={"displayModeBar": False})

    # ── B. Grounding Corpus Map ─────────────────────────────────────
    with tabs[1]:
        st.subheader("Grounding corpus map")
        corpus, ph_c = dl.load_grounding_corpus()
        family, ph_f = dl.load_family_counts()
        _placeholder_badge(ph_c or ph_f)
        if not ph_c:
            st.caption(
                "Source: `gaira_evidence_warehouse_grounding_backbone_v1/tables/warehouse_source_registry.csv` "
                "(43 source rows) joined with `grounding_peak_support_summary.csv` for per-source spectrum + "
                "peak counts. Tier 1 = reference / serum grounding; Tier 2 = literature / disease-stress papers."
            )
        if ph_f:
            st.caption(
                "**Per-axis analyte counts are curated placeholders** — the generated evidence "
                "table `data/generated/per_axis_grounding_evidence.csv` was not found. "
                "Run `python tools/build_grounding_evidence.py` to regenerate it."
            )

        # ── Tier 1 sources ─────────────────────────────────────────
        t1 = corpus[corpus["tier"] == 1].copy()
        t2 = corpus[corpus["tier"] == 2].copy()

        st.markdown("**Tier 1 — direct spectral grounding**")
        import plotly.graph_objects as go
        if not t1.empty and t1["n_spectra"].sum() > 0:
            t1_chart = t1[t1["n_spectra"] > 0].sort_values("n_spectra", ascending=True)
            fig = go.Figure(go.Bar(
                x=t1_chart["n_spectra"], y=t1_chart["source"], orientation="h",
                marker_color=["#60A5FA" if r == "Raman" else "#F87171" if r == "SERS" else "#A78BFA"
                                for r in t1_chart["regime"]],
                customdata=t1_chart[["source_family", "regime", "n_peaks", "n_classes"]].values,
                hovertemplate=("<b>%{y}</b><br>family: %{customdata[0]}<br>"
                                "regime: %{customdata[1]}<br>%{x} spectra · "
                                "%{customdata[2]} peaks · %{customdata[3]} classes<extra></extra>"),
                name="spectra",
            ))
            gp.apply_dark(fig, title="Tier 1 sources with measured spectra/peak counts",
                            height=300, show_legend=False)
            fig.update_xaxes(title="Spectra count")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("No tier-1 sources have spectrum counts in the join — see table below.")

        with st.expander(f"Tier 1 source table ({len(t1)} rows)", expanded=False):
            st.dataframe(
                t1[["source", "source_family", "regime", "biosample_type",
                     "n_spectra", "n_peaks", "n_classes"]].reset_index(drop=True),
                use_container_width=True, hide_index=True,
            )

        # ── Tier 2 sources ─────────────────────────────────────────
        st.markdown(f"**Tier 2 — literature & disease/stress papers ({len(t2)} sources)**")
        # Aggregate the disease/stress sources by source_family + regime for a clean chart
        if not t2.empty:
            agg = (t2.groupby(["source_family", "regime"]).size()
                       .reset_index(name="count"))
            fig = go.Figure()
            for regime, color in [("Raman", "#60A5FA"), ("SERS", "#F87171"), ("both", "#A78BFA")]:
                sub = agg[agg["regime"] == regime]
                if sub.empty: continue
                fig.add_trace(go.Bar(x=sub["source_family"], y=sub["count"],
                                          name=regime, marker_color=color))
            fig.update_layout(barmode="stack")
            gp.apply_dark(fig, title="Tier 2 source distribution by family × regime",
                            height=260)
            fig.update_yaxes(title="Source rows")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            with st.expander(f"Tier 2 source table ({len(t2)} rows)", expanded=False):
                st.dataframe(
                    t2[["source", "source_family", "regime", "biosample_type", "notes"]]
                        .reset_index(drop=True),
                    use_container_width=True, hide_index=True,
                )

        # ── Per-axis grounding evidence (v2: real, NA-aware) ──────────
        st.markdown("**Per-axis grounding evidence**")
        fam = family.copy()
        fam["label"] = fam["axis"].map(cfg.axis_short)
        fam["color"] = fam["axis"].map(cfg.axis_color)

        # Numeric analyte value for plotting; NA/placeholder-safe.
        def _num(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
        fam["analyte_num"] = fam["n_analytes"].map(_num)

        # Bar shows only axes with a defensible resolved count; NA axes are
        # listed explicitly below rather than drawn as zero.
        drawn = fam[fam["analyte_num"].notna()].copy()
        fig = go.Figure(go.Bar(
            x=drawn["analyte_num"], y=drawn["label"], orientation="h",
            marker_color=drawn["color"],
            hovertemplate="<b>%{y}</b><br>%{x:.0f} unique reference analytes<extra></extra>",
        ))
        gp.apply_dark(fig, title="", height=340, show_legend=False)
        fig.update_xaxes(title="Unique reference analytes"
                         + (" (curated placeholder)" if ph_f else " (resolved axes only)"))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        if not ph_f:
            na_axes = fam[fam["analyte_num"].isna()]["label"].tolist()
            if na_axes:
                st.caption(
                    "**NA (not shown as bars):** " + ", ".join(na_axes) + ". These axes descend "
                    "from legacy 8-axis grounding groups that split into two v11 children "
                    "(purine → Purine-nuc/Purine-met; lipid → Lipid/Sterol; redox → Redox/Metabolite). "
                    "The 8-axis reference table cannot defensibly assign an analyte to one child, "
                    "so the per-axis count is reported as NA rather than 0."
                )
            with st.expander("Per-axis grounding evidence table (methodology-explicit)", expanded=False):
                show_cols = [c for c in ["axis_short", "unique_reference_analytes",
                                         "measured_reference_spectra", "direct_spectral_sources",
                                         "supporting_literature_sources", "unmapped_records",
                                         "mapping_status", "mapping_notes"] if c in fam.columns]
                st.dataframe(fam[show_cols].reset_index(drop=True),
                             hide_index=True, use_container_width=True)
                st.caption(
                    "**Column meanings** — `unique_reference_analytes`: distinct reference molecules "
                    "(RamanBioLib 202-molecule table) whose dominant 8-axis maps to this v11 axis "
                    "(unique compounds, not files or augmented spectra). "
                    "`measured_reference_spectra` / `direct_spectral_sources` / "
                    "`supporting_literature_sources`: NA per-axis because the grounding registry records "
                    "these at the source level, not per axis — they are reported corpus-wide below. "
                    "`unmapped_records`: analytes in a shared legacy pool that cannot be resolved to this "
                    "single child axis. `mapping_status`: resolved (1:1 legacy→v11) / ambiguous_8axis_split / "
                    "not_axis_mapped."
                )

            summ = dl.load_grounding_corpus_summary()
            if summ:
                msr = summ.get("measured_reference_spectra")
                by = summ.get("measured_reference_spectra_by_source", {})
                by_txt = ", ".join(f"{k} {v}" for k, v in by.items()) if by else "—"
                st.caption(
                    "**Corpus-level grounding (unique entities, kept separate by type):** "
                    f"{summ.get('unique_reference_analytes_total','?')} unique reference analytes "
                    "(RamanBioLib); "
                    f"{msr if msr is not None else 'NA'} measured reference spectra "
                    f"across {len(by)} counted reference/serum-grounding sources ({by_txt}); "
                    f"{summ.get('direct_spectral_source_unique','?')} unique direct-spectral sources "
                    f"({summ.get('direct_spectral_source_rows','?')} registry rows); "
                    f"{summ.get('supporting_literature_sources','?')} unique supporting-literature sources. "
                    "Measured spectra are never combined with literature papers. "
                    + (" · ".join(summ.get("registry_notes", [])))
                )

        st.markdown("---")
        st.markdown("**Evidence tier definitions**")
        st.markdown(
            "- **Tier 1 — direct spectral grounding.** Reference molecular spectra "
            "(adenine_sers_control, amino_acid_raman_grounding, metabolite_sers63_support, "
            "ramanbiolib_reference_bridge, serum_ag_colloids_grounding). These are the ground "
            "truth that motifs and MSS are built against."
        )
        st.markdown(
            "- **Tier 2 — literature & disease/stress evidence.** Manuscripts and curated "
            "literature notes (CCA 2024, Chen 2020, exosome-SERS 2023, interlab-SERS, Krafft 2018, etc.) "
            "providing assignment context and domain-specific caveats. These shape interpretation "
            "but do not by themselves create molecule-level calls."
        )

    # ── C. 11-Axis Biochemical Space ────────────────────────────────
    with tabs[2]:
        st.subheader("11-axis biochemical space")
        st.markdown(
            "<p class='caption-muted'>Reference spectra organized into GAIRA's "
            "biochemical family space. Each point is a curated molecule projected "
            "into 2D via UMAP (preferred) or PCA fallback over its 11-axis BSV profile. "
            "This is <b>not</b> a trained foundation model — it is a visualization "
            "of how GAIRA's family ontology separates known chemistry. Translucent "
            "envelopes are 1.8σ ellipses, drawn only for families with ≥5 grounded "
            "molecules and non-degenerate covariance — sparse families are shown "
            "as points only, so the visual does not falsely imply perfect separability.</p>",
            unsafe_allow_html=True,
        )
        ref_df, ph = dl.load_reference_points()
        _placeholder_badge(ph)

        # Filters
        c1, c2, c3 = st.columns(3)
        cats = sorted(ref_df["category"].dropna().unique().tolist())
        regimes = sorted(ref_df["regime"].dropna().unique().tolist())
        axes_filter = ["All"] + [cfg.axis_short(a) for a in cfg.BSV_AXES]
        with c1:
            pick_cat = st.multiselect("Category", cats, default=cats)
        with c2:
            pick_reg = st.multiselect("Regime", regimes, default=regimes)
        with c3:
            pick_axis = st.selectbox("Highlight family", axes_filter, index=0)

        df = ref_df[ref_df["category"].isin(pick_cat)
                    & ref_df["regime"].isin(pick_reg)].copy()
        if df.empty:
            st.warning("No reference points match the filter.")
        else:
            X = df[list(cfg.BSV_AXES)].values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            # Compute PCA (UMAP fallback if available)
            coords = None
            try:
                import umap  # type: ignore
                reducer = umap.UMAP(n_neighbors=min(15, max(2, len(X) - 1)),
                                      min_dist=0.25, random_state=42)
                coords = reducer.fit_transform(X)
                method = "UMAP"
            except Exception:
                try:
                    from sklearn.decomposition import PCA
                    reducer = PCA(n_components=2, random_state=42)
                    coords = reducer.fit_transform(X)
                    method = "PCA"
                except Exception:
                    coords = X[:, :2]
                    method = "raw"

            df["x"], df["y"] = coords[:, 0], coords[:, 1]
            df["dominant_axis"] = df[list(cfg.BSV_AXES)].idxmax(axis=1)

            if pick_axis != "All":
                # Highlight that axis with full opacity, dim others
                target_axis = next(a for a in cfg.BSV_AXES if cfg.axis_short(a) == pick_axis)
                df["dominant_axis"] = np.where(df["dominant_axis"] == target_axis,
                                                  df["dominant_axis"], "_other")

            fig = gp.biochemical_space_figure(
                df,
                title=f"Biochemical space — {method} projection over 11-axis BSV",
                height=560,
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})
            st.caption(
                f"Projection: {method} · {len(df)} reference spectra · "
                f"colored by dominant 11-axis family · translucent envelopes are "
                f"1.8σ confidence ellipses (only drawn when ≥5 grounded points "
                f"and non-degenerate covariance)"
            )

    # ── D. MSS / Motif Explorer ──────────────────────────────────────
    with tabs[3]:
        st.subheader("MSS / motif explorer")
        st.markdown(
            "<p class='caption-muted'>Curated molecular references with their "
            "anchor and supporting bands, anti-evidence regions, and the 11-axis "
            "contribution profile that GAIRA uses to route evidence.</p>",
            unsafe_allow_html=True,
        )

        mol_id = st.selectbox(
            "Molecule",
            options=dl.molecule_list(),
            format_func=lambda m: dl.MOLECULES[m].name,
        )
        ref = dl.MOLECULES[mol_id]
        wn, y = dl.synth_reference_spectrum(mol_id)

        # Spectrum with anchors/supports
        ant_regions = [(c - 8, c + 8, "#34D399") for c in ref.anchors]
        sup_regions = [(c - 10, c + 10, "#A5B4FC") for c in ref.supports]
        anti_regions = [(c - 8, c + 8, "#F87171") for c in ref.anti_evidence]
        col_l, col_r = st.columns([1.6, 1])
        with col_l:
            fig = gp.spectrum_figure(
                [{"x": wn, "y": y, "name": ref.name, "color": "#60A5FA"}],
                title=f"{ref.name} — reference spectrum",
                highlight_anchors=list(ref.anchors),
                highlight_supports=list(ref.supports),
                highlight_regions=ant_regions + sup_regions + anti_regions,
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})
            st.caption(
                "Green: anchor bands · Light-blue dotted: supporting bands · "
                "Red shaded: anti-evidence regions"
            )

        with col_r:
            st.markdown(
                f"""
                <div class="card">
                  <h4>MSS card — {ref.name}</h4>
                  <div class="muted" style="margin-bottom: 8px;">{ref.summary}</div>
                  <b>Primary family:</b> {cfg.axis_long(ref.primary_axis)}<br>
                  <b>Anchor bands:</b> {', '.join(f'{c:.0f}' for c in ref.anchors)} cm⁻¹<br>
                  <b>Supports:</b> {', '.join(f'{c:.0f}' for c in ref.supports) or '—'} cm⁻¹<br>
                  <b>Anti-evidence:</b> {', '.join(f'{c:.0f}' for c in ref.anti_evidence) or '—'} cm⁻¹<br>
                  <b>Interpretation rule:</b>
                  <span class="muted">Class-level evidence unless co-features support
                  molecule-level specificity.</span><br><br>
                  <b>Substrate / domain notes:</b>
                  <div class="muted">{ref.domain_notes}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 11-axis contribution profile (static curated profile)
        contrib = molecule_axis_contributions(mol_id)
        st.markdown("**11-axis contribution profile**")
        st.plotly_chart(
            gp.axis_bar_figure(contrib, title=None, height=300),
            use_container_width=True, config={"displayModeBar": False},
        )

    # ── E. Collision Viewer ──────────────────────────────────────────
    with tabs[4]:
        st.subheader("Collision viewer — admitting ambiguity")
        st.markdown(
            "<p class='caption-muted'>GAIRA does not force a molecule-level call "
            "when spectral evidence collides. Shared bands are routed to "
            "biochemical families, and molecular specificity is downgraded.</p>",
            unsafe_allow_html=True,
        )

        COLL_PAIRS = [
            ("adenine", "uric_acid", "Purine collision — adenine vs uric acid"),
            ("phenylalanine", "tyrosine", "Aromatic AA collision — Phe vs Tyr"),
            ("lactate", "glucose", "Lactate vs glucose — C–C–O overlap near 845 cm⁻¹"),
            ("oleic_acid", "cholesterol", "Lipid vs sterol — 1440 CH₂ shared, 548/1655 distinguishable"),
            ("ergothioneine", "adenine", "Thione imidazole 720 cm⁻¹ vs adenine purine 725 cm⁻¹"),
        ]
        labels = [t for _, _, t in COLL_PAIRS]
        pick = st.selectbox("Collision pair", labels, index=0)
        a_id, b_id, _ = COLL_PAIRS[labels.index(pick)]
        ra, rb = dl.MOLECULES[a_id], dl.MOLECULES[b_id]
        wn, ya = dl.synth_reference_spectrum(a_id)
        _, yb = dl.synth_reference_spectrum(b_id)

        # Shared vs unique anchors (within ±10 cm-1)
        shared = []
        unique_a, unique_b = list(ra.anchors), list(rb.anchors)
        for ca in ra.anchors:
            for cb in rb.anchors:
                if abs(ca - cb) <= 10.0:
                    shared.append((ca, cb))
                    if ca in unique_a: unique_a.remove(ca)
                    if cb in unique_b: unique_b.remove(cb)

        fig = gp.spectrum_figure(
            [
                {"x": wn, "y": ya, "name": ra.name, "color": "#60A5FA"},
                {"x": wn, "y": yb, "name": rb.name, "color": "#F87171"},
            ],
            title=pick,
            highlight_regions=[(c - 8, c + 8, "#FBBF24") for c, _ in shared],
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})
        st.caption("Yellow shaded regions: shared bands between the two molecules.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"<div class='card'><h4>Shared bands</h4>"
                f"<div class='muted'>{', '.join(f'{a:.0f}/{b:.0f}' for a, b in shared) or '—'} cm⁻¹</div></div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"<div class='card'><h4>Unique to {ra.name}</h4>"
                f"<div class='muted'>{', '.join(f'{c:.0f}' for c in unique_a) or '—'} cm⁻¹</div></div>",
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"<div class='card'><h4>Unique to {rb.name}</h4>"
                f"<div class='muted'>{', '.join(f'{c:.0f}' for c in unique_b) or '—'} cm⁻¹</div></div>",
                unsafe_allow_html=True,
            )

        # Collision score (Jaccard over anchor sets ±10 cm-1 tolerance)
        score = len(shared) / max(1, len(set(ra.anchors) | set(rb.anchors)) - len(shared))
        score = min(1.0, score)
        st.metric("Collision score (anchor overlap)", f"{score:.2f}")
        st.markdown(
            f"<div class='card'><h4>GAIRA interpretation</h4>"
            f"<div class='muted'>With {len(shared)} shared anchor band(s), "
            f"molecule-level resolution requires unique co-bands. "
            f"GAIRA routes shared evidence to the family level "
            f"({cfg.axis_long(ra.primary_axis)} / {cfg.axis_long(rb.primary_axis)}) "
            f"and downgrades molecular specificity to <b>class-level</b> unless "
            f"unique anchors fire above prominence threshold.</div></div>",
            unsafe_allow_html=True,
        )

    # ── F. Physics-Aware Atlas ───────────────────────────────────────
    with tabs[5]:
        st.subheader("Physics-aware spectral atlas")
        st.markdown(
            "<p class='caption-muted'>GAIRA's spectral interpretation is anchored "
            "to a curated atlas of biochemically-meaningful regions. Each region "
            "carries plausible assignments, ambiguity notes, and substrate "
            "sensitivity — encoded as a modular scientific layer that informs "
            "every BSV call.</p>",
            unsafe_allow_html=True,
        )

        st.plotly_chart(gp.atlas_ruler_figure(), use_container_width=True,
                        config={"displayModeBar": False})

        # Detailed region cards
        atlas_details = {
            "Skeletal / ring / metal-ligand": {
                "assignments": "Skeletal modes, ring deformations (sterol 548), low-freq lattice, "
                                 "metal-ligand stretches, thione/thiol C–S near 490–510.",
                "ambiguity": "Many overlapping small modes; class-level only unless a sharp anchor (e.g. 548, 495) fires.",
                "substrate": "SERS substrates often dominated by metal-ligand stretches; "
                              "ambiguity high in Ag-colloid serum.",
                "confounders": "Substrate background, glass/CaF₂ artifacts in some Raman setups.",
                "treatment": "GAIRA uses 548 cm⁻¹ as sterol-specific anchor; 490–510 cm⁻¹ as thione/thiol anchor; "
                              "all other low-frequency bands routed to family-level only.",
            },
            "Purine / nucleobase-sensitive": {
                "assignments": "Purine ring breathing (adenine 725, guanine 670 shifted), shared by hypoxanthine/uric-acid context.",
                "ambiguity": "Adenine, hypoxanthine, and other purines all contribute near 720–740. Ag-SERS amplifies this region.",
                "substrate": "Ag colloid SERS: ×3–10 amplification — molecule-level call requires corroborating 1335 (adenine) "
                              "or 640+891 (uric acid).",
                "confounders": "Imidazole (e.g. ergothioneine 720); ergothioneine-purine collision.",
                "treatment": "Boost purine-family evidence; downgrade molecule-level specificity until co-bands agree.",
            },
            "Ring breathing / AA / carbohydrate": {
                "assignments": "Tyrosine doublet 830/850, tryptophan 760, ring breathing modes; sugar anomeric C–H 845; lactate C–C–O 845.",
                "ambiguity": "Lactate 845 collides with sugar anomeric; tyrosine doublet ratio is conformation-sensitive.",
                "substrate": "Moderate; aromatic AA bands relatively stable across regimes.",
                "confounders": "Lipid/sugar overlap near 870.",
                "treatment": "Tyrosine doublet retained as conformation context; lactate requires 845+925 co-fire.",
            },
            "C–C / C–O / phosphate / glycan": {
                "assignments": "Glycan C–O/C–C 1020–1150, nucleic-acid PO₂⁻ 1080, glucose 1125.",
                "ambiguity": "Phosphate 1080 and glycan 1075–1085 overlap; both routed to G04 and G05 with co-band weighting.",
                "substrate": "Phosphate signal weakens on Ag-SERS unless adsorbed.",
                "confounders": "Lipid C–C stretches near 1130 mix in.",
                "treatment": "Co-band logic — 1080 + nucleobase ⇒ nucleic acid; 1080 + 1125 ⇒ glycan.",
            },
            "Nucleobase / protein / lipid mixed": {
                "assignments": "Adenine 1335, amide-III 1230–1300, lipid C–C 1300, pyrimidine 1230, lactate 1280.",
                "ambiguity": "Highly mixed; rarely resolves to one chemistry alone.",
                "substrate": "Amide-III is reliable in Raman; SERS rebalances toward adsorbed species.",
                "confounders": "Multi-chemistry overlap; competitor scoring required.",
                "treatment": "GAIRA scores each motif independently then uses anti-evidence rules to prevent false amplification.",
            },
            "CH deformation / nucleobase / lipid": {
                "assignments": "CH₂ deformation 1440 (lipid + sterol), nucleobase 1485 (adenine), 1410 amino-acid carboxyl.",
                "ambiguity": "CH₂ 1440 is shared across all lipids and many small molecules.",
                "substrate": "Reliable across regimes; minor shift on SERS.",
                "confounders": "Sterol C=C 1665 vs amide-I overlap.",
                "treatment": "1440 routed to G08 family; sterol-specific call requires 548 cm⁻¹.",
            },
            "Aromatic / amide / C=C / unsat.": {
                "assignments": "Amide-I 1640–1670, aromatic 1605 (Phe), C=C 1655 (lipid unsat.), carotenoid 1517 (UA collision), tryptophan 1550.",
                "ambiguity": "Amide-I and lipid C=C overlap; carotenoid 1517 collides with uric-acid 1517 in serum SERS.",
                "substrate": "Carotenoid background is matrix-dependent; SERS rebalances aromatic intensities.",
                "confounders": "Carotenoid masking of UA in serum SERS.",
                "treatment": "G07 (aromatic) and G08 (lipid) co-firing triggers ambiguity routing; carotenoid caveat applied automatically.",
            },
            "Carbonyl / lipid ester / oxidation": {
                "assignments": "Ester carbonyl 1745, oxidation products 1700–1800, peroxide signatures.",
                "ambiguity": "Atmospheric/oxidation artifacts can produce 1700–1800 bands not present in fresh sample.",
                "substrate": "Sensitive to laser-induced photoproducts on SERS.",
                "confounders": "Photoproduct artifacts flagged as anti-evidence in some MSS.",
                "treatment": "1745 supports lipid ester evidence; 1770–1800 flagged as artifact-suspect for aromatic and other axes.",
            },
        }

        for region in cfg.ATLAS_REGIONS:
            label = region["label"]
            details = atlas_details.get(label, {})
            axes_str = " · ".join(cfg.axis_short(a) for a in region["axes"])
            with st.expander(f"{region['start']}–{region['end']} cm⁻¹ — {label}", expanded=False):
                st.markdown(f"**Implicated GAIRA axes:** {axes_str}")
                if details:
                    st.markdown(f"**Possible evidence:** {details['assignments']}")
                    st.markdown(f"**Ambiguity:** {details['ambiguity']}")
                    st.markdown(f"**Substrate sensitivity:** {details['substrate']}")
                    st.markdown(f"**Known confounders:** {details['confounders']}")
                    st.markdown(f"**GAIRA treatment:** {details['treatment']}")

        st.caption(
            "The atlas is built as a modular scientific context layer; future "
            "architecture could expose these modules as MCP tools."
        )

    # ── G. End-to-End Workflow ───────────────────────────────────────
    with tabs[6]:
        st.subheader("End-to-end workflow")
        st.markdown(
            "<p class='caption-muted'>Pick a scenario to see the full GAIRA pipeline "
            "on a single spectrum: input → preprocessing → primitives → MSS / motif "
            "scoring → substrate adjustment → 11-axis BSV → interpretation.</p>",
            unsafe_allow_html=True,
        )

        scenarios = {
            "Ergothioneine calibration (Raman)": ("ergothioneine", "Raman", "calibration"),
            "Adenine calibration (Ag-SERS)":     ("adenine",       "Ag colloid SERS", "calibration"),
            "Uric acid serum reference (Raman)": ("uric_acid",     "Raman", "serum"),
            "Healthy serum proxy (Raman)":       ("phenylalanine", "Raman", "serum"),
            "Lipid-rich EV proxy (Raman)":       ("oleic_acid",    "Raman", "extracellular_vesicle"),
        }
        scenario = st.selectbox("Scenario", list(scenarios.keys()))
        mol_id, substrate, domain = scenarios[scenario]
        wn, y = dl.synth_reference_spectrum(mol_id)

        report = build_report(
            sample_id=f"demo_{mol_id}_{substrate.replace(' ', '_')}",
            title=scenario,
            domain=domain, substrate=substrate,
            wavenumber=wn, intensity=y,
        )
        _placeholder_badge(True)

        # Step cards
        st.markdown("##### Step 1 — Input spectrum")
        st.plotly_chart(
            gp.spectrum_figure(
                [{"x": wn, "y": y, "name": "raw (synthesised reference)", "color": "#60A5FA"}],
                height=260,
            ),
            use_container_width=True, config={"displayModeBar": False},
        )

        st.markdown("##### Step 2 — Preprocessing (ASLS + Savitzky–Golay + L2)")
        pwn = np.array(report["spectrum"]["wavenumber"])
        proc = np.array(report["spectrum"]["processed_intensity"])
        st.plotly_chart(
            gp.spectrum_figure(
                [{"x": pwn, "y": proc, "name": "processed", "color": "#34D399"}],
                height=260,
            ),
            use_container_width=True, config={"displayModeBar": False},
        )
        st.caption(report["preprocessing"])

        st.markdown("##### Step 3 — Peak / motif detection")
        col1, col2 = st.columns([1, 1])
        with col1:
            anchors_df = pd.DataFrame(report["features"]["anchors"])
            if anchors_df.empty:
                st.info("No motifs fired above floor for this scenario.")
            else:
                st.dataframe(anchors_df, hide_index=True, use_container_width=True)
        with col2:
            st.metric("Peaks detected", report["features"]["n_peaks"])
            st.metric("Motifs firing", sum(1 for v in report["motif_scores_adjusted"].values() if v >= 0.05))

        st.markdown("##### Step 4 — MSS scoring")
        mss_df = pd.DataFrame([
            {"molecule": mol_id, "fire": v["fire"],
              "anchor": v["anchor"], "support": v["support"], "anti": v["anti"]}
            for mol_id, v in sorted(report["mss_fires"].items(),
                                       key=lambda kv: -kv[1]["fire"])
            if v["fire"] > 0.02
        ])
        if not mss_df.empty:
            st.dataframe(mss_df, hide_index=True, use_container_width=True)

        st.markdown("##### Step 5 — Substrate adjustment")
        if report["substrate_events"]:
            ev_df = pd.DataFrame(report["substrate_events"])
            st.dataframe(ev_df, hide_index=True, use_container_width=True)
        else:
            st.caption(f"No substrate-specific motif adjustment rules fired for substrate = "
                          f"<b>{substrate}</b>.")

        st.markdown("##### Step 6 — BSV projection (11-axis radar)")
        st.plotly_chart(
            gp.radar_figure(
                [{"name": scenario, "values": report["bsv"], "color": "#60A5FA"}],
                height=460, radial_max=1.0,
            ),
            use_container_width=True, config={"displayModeBar": False},
        )

        st.markdown("##### Step 7 — Interpretation report")
        conf = report["confidence"]
        st.markdown(
            _evidence_pills(
                strength="high" if conf["overall"] == "moderate-high" else conf["overall"],
                domain_fit="strong" if domain != "extracellular_vesicle" else "caution",
                sub_sens=conf["substrate_sensitivity"],
                specificity=conf["molecular_specificity"],
            ),
            unsafe_allow_html=True,
        )
        st.markdown("**Top axes**")
        for entry in report["top_axes"]:
            st.markdown(
                f"- {cfg.axis_long(entry['axis'])} — value {entry['value']:.3f} "
                f"({entry['direction']}, evidence {entry['evidence_strength']})"
            )
        if report["evidence"]:
            st.markdown("**Evidence**")
            for ev in report["evidence"][:5]:
                bands = ", ".join(ev["bands"][:4]) or "—"
                st.markdown(
                    f"- *{cfg.axis_long(ev['axis'])}* "
                    f"<span class='pill'>{ev['evidence_type']}</span> "
                    f"<span class='pill pill-ok' style='font-size:0.74rem;'>{ev['confidence']}</span><br>"
                    f"<span class='caption-muted'>bands: {bands} — {ev['summary']}</span>",
                    unsafe_allow_html=True,
                )
        if report["caveats"]:
            st.markdown("**Caveats**")
            for c in report["caveats"]:
                st.markdown(
                    f"- <span class='pill pill-warn'>{c['type']}</span> "
                    f"<span class='caption-muted'>{c['summary']}</span>",
                    unsafe_allow_html=True,
                )

    # ── H. Biochemical Ontology v1 (V3) ─────────────────────────────
    with tabs[7]:
        v3v.render_ontology_panel()

    # ── I. Raw BSV → Global Coordinates (V3) ────────────────────────
    with tabs[8]:
        v3v.render_coordinate_construction()


# ════════════════════════════════════════════════════════════════════
# MODE 2 — CALIBRATION EVIDENCE
# ════════════════════════════════════════════════════════════════════

elif mode == "Calibration Evidence":
    tabs = st.tabs([
        "Ergothioneine Dose Slider",
        "Adenine Detection",
        "Uric Acid / Isotope Validation",
        "Axis Coverage Summary",
    ])

    # ── A. Ergothioneine Dose Slider ────────────────────────────────
    with tabs[0]:
        st.subheader("Ergothioneine — dose response")
        st.markdown(
            "<p class='caption-muted'>Increasing ergothioneine concentration should "
            "produce a monotonic rise in the sulfur–thiol/redox biochemical state axis. "
            "GAIRA's calibration data confirms this expected behavior.</p>",
            unsafe_allow_html=True,
        )

        erg, ph = dl.load_ergothioneine_dose()
        _placeholder_badge(ph)
        if not ph:
            st.caption(
                "Source: `streamlit_apps/gaira_demo/data/ergothioneine_dose_response.csv` "
                "(real cached BSV) — 8-axis legacy values remapped to the 11-axis "
                "system; `redox_metabolite` is split 50/50 between G10 (sulfur/thiol/redox) "
                "and G11 (metabolic small molecule). Ergothioneine's biology lives "
                "primarily in G10 once remapped."
            )

        concs = sorted(erg["concentration_uM"].unique().tolist())
        conc = st.select_slider("Ergothioneine concentration (µM)", options=concs,
                                  value=concs[len(concs) // 2])
        row = erg[erg["concentration_uM"] == conc].iloc[0]
        bsv = {a: float(row[a]) for a in cfg.BSV_AXES}
        baseline = erg[erg["concentration_uM"] == concs[0]].iloc[0]
        baseline_bsv = {a: float(baseline[a]) for a in cfg.BSV_AXES}

        # Dynamic radial axis — real ergothioneine BSV values are O(0.01–0.06)
        # after the 8→11 remap, so a fixed radial_max=1.0 would render slider
        # changes invisible. Use the per-dataset max with headroom.
        radial_max = max(
            max(baseline_bsv.values()),
            max(bsv.values()),
            max(float(erg[a].max()) for a in cfg.BSV_AXES),
        )
        radial_max = max(0.05, radial_max * 1.15)

        col1, col2 = st.columns([1.1, 1])
        with col1:
            # Synthesised spectrum scaled by concentration in the G10 anchor regions
            wn, y0 = dl.synth_reference_spectrum("ergothioneine", seed=1)
            # Scale anchors by concentration
            scale = float(conc) / max(concs)
            y_scaled = y0 * (0.3 + 0.7 * scale)
            fig = gp.spectrum_figure(
                [{"x": wn, "y": y_scaled, "name": f"{conc:.2f} µM (synth.)", "color": "#FDE68A"}],
                title="Ergothioneine reference spectrum (synthesised, scaled by concentration)",
                highlight_anchors=list(dl.MOLECULES["ergothioneine"].anchors),
                height=320,
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})
            st.caption(
                "Spectrum is a synthesised reference (Gaussian-bump model) — "
                "the BSV values below are real cached values, not regenerated from this synthetic spectrum."
            )

        with col2:
            st.plotly_chart(
                gp.radar_figure(
                    [
                        {"name": f"0 µM (baseline)", "values": baseline_bsv, "color": "#64748B"},
                        {"name": f"{conc:.2f} µM", "values": bsv, "color": "#FDE68A"},
                    ],
                    title=f"BSV — baseline vs current dose (radial scale 0–{radial_max:.2f})",
                    height=380, radial_max=radial_max,
                ),
                use_container_width=True, config={"displayModeBar": False},
            )

        c1, c2, c3 = st.columns(3)
        c1.metric("G10 — Sulfur/thiol/redox", f"{bsv['G10_sulfur_thiol_redox']:.3f}",
                    delta=f"{bsv['G10_sulfur_thiol_redox'] - baseline_bsv['G10_sulfur_thiol_redox']:+.3f}")
        c2.metric("G11 — Metabolic small mol.", f"{bsv['G11_metabolic_small_molecule']:.3f}",
                    delta=f"{bsv['G11_metabolic_small_molecule'] - baseline_bsv['G11_metabolic_small_molecule']:+.3f}")
        c3.metric("G07 — Aromatic (imidazole leak)", f"{bsv['G07_aromatic_residue']:.3f}",
                    delta=f"{bsv['G07_aromatic_residue'] - baseline_bsv['G07_aromatic_residue']:+.3f}")

        st.plotly_chart(
            gp.dose_response_figure(erg, "G10_sulfur_thiol_redox",
                                      title="Dose-response — G10 sulfur/thiol/redox",
                                      height=340),
            use_container_width=True, config={"displayModeBar": False},
        )

        st.markdown(
            f"""
            <div class='card'>
              <h4>Interpretation</h4>
              <div class='muted'>Increasing ergothioneine produces a monotonic rise in the
              <b>G10 sulfur/thiol/redox</b> axis, driven primarily by the C–S thione stretch
              near 490–500 cm⁻¹. A secondary, smaller rise in <b>G11 metabolic small molecule</b>
              reflects ergothioneine's metabolite character. The G07 aromatic leak corresponds to
              the imidazole 720 cm⁻¹ band which GAIRA flags as a known purine-collision feature.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── B. Adenine Detection ────────────────────────────────────────
    with tabs[1]:
        st.subheader("Adenine — real Ag-SERS dose response")
        ad, ph = dl.load_adenine_calibration()
        _placeholder_badge(ph)
        _adenine_dir = gpaths.adenine_raw_dir()
        if ph:
            st.caption(
                "**Placeholder.** Adenine raw spectra were not found. Expected the "
                "`raw/adenine_sers_control/` folder under the resolved GAIRA_DATA root "
                f"(currently: `{cfg.GAIRA_DATA_VOLUME}`). Mount the drive or set "
                "`GAIRA_DATA_ROOT` — see the data-source banner at the top of the app."
            )
        else:
            st.caption(
                f"Source: `{_adenine_dir}` "
                "(6-point bAgNPs concentration series, 10 pg/mL → 10 µg/mL). Each "
                "raw `.CSV` (semicolon-separated, Windows-1252, Latin comma decimals) "
                "is parsed, cropped to 400–1800 cm⁻¹, interpolated to a 1 cm⁻¹ grid, "
                "and run through `gaira_core/report_builder.py:build_report` with "
                "substrate=`Ag colloid SERS` so the demo's Ag-SERS purine-dampening "
                "rule applies. BSV values below are computed live, not cached."
            )

        conditions = ad["condition"].tolist()
        cond = st.select_slider("Adenine concentration", options=conditions, value=conditions[len(conditions)//2])
        row = ad[ad["condition"] == cond].iloc[0]
        bsv = {a: float(row[a]) for a in cfg.BSV_AXES}
        baseline_bsv = {a: float(ad.iloc[0][a]) for a in cfg.BSV_AXES}

        # Dynamic radial — real adenine BSV peaks at ~0.17 even at 10 µg/mL
        # (Ag-SERS dampening keeps the call class-level). Fixed scale would hide motion.
        radial_max = max(
            max(bsv.values()), max(baseline_bsv.values()),
            max(float(ad[a].max()) for a in cfg.BSV_AXES),
        )
        radial_max = max(0.05, radial_max * 1.15)

        col1, col2 = st.columns([1.1, 1])
        with col1:
            wn_ref, y_ref = dl.synth_reference_spectrum("adenine", seed=2)
            fig = gp.spectrum_figure(
                [{"x": wn_ref, "y": y_ref, "name": "adenine reference (synth.)", "color": "#F87171"}],
                title="Adenine reference spectrum (synthesised, illustrative only)",
                highlight_anchors=list(dl.MOLECULES["adenine"].anchors),
                height=320,
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})
            st.caption(
                "Spectrum panel above is a synthesised reference for visual anchor placement only. "
                "The radar and dose curve are computed from real Ag-SERS spectra."
            )
        with col2:
            st.plotly_chart(
                gp.radar_figure(
                    [
                        {"name": f"{conditions[0]} (lowest)", "values": baseline_bsv, "color": "#64748B"},
                        {"name": cond, "values": bsv, "color": "#F87171"},
                    ],
                    title=f"BSV — lowest vs current (radial 0–{radial_max:.2f})",
                    height=380, radial_max=radial_max,
                ),
                use_container_width=True, config={"displayModeBar": False},
            )

        # Dose-response curve for G01 across all concentrations
        dose_df = ad.copy()
        dose_df = dose_df.rename(columns={"concentration_ng_mL": "concentration_uM"})
        # we keep the x-label honest below
        import plotly.graph_objects as _go
        fig_d = _go.Figure(_go.Scatter(
            x=ad["concentration_ng_mL"], y=ad["G01_purine_nucleotide"], mode="lines+markers",
            line=dict(color=cfg.axis_color("G01_purine_nucleotide"), width=2.5),
            marker=dict(color=cfg.axis_color("G01_purine_nucleotide"), size=10),
            hovertemplate="%{x} ng/mL<br>G01 = %{y:.3f}<extra></extra>",
        ))
        gp.apply_dark(fig_d, title="Dose response — G01 purine nucleotide (real Ag-SERS)",
                       height=320, show_legend=False)
        fig_d.update_xaxes(title="Concentration (ng/mL, log)", type="log")
        fig_d.update_yaxes(title="BSV G01 (Ag-SERS-dampened)")
        st.plotly_chart(fig_d, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            f"""
            <div class='card'>
              <h4>Evidence — adenine at {cond}</h4>
              <ul style='margin-top: 4px;'>
                <li><b>G01 Purine nucleotide:</b> {bsv['G01_purine_nucleotide']:.3f}
                    — driven by 720–740 cm⁻¹ ring-breathing motif (Ag-SERS amplified, demo dampened ×0.65 by substrate rule).</li>
                <li><b>G02 Purine metabolite:</b> {bsv['G02_purine_metabolite']:.3f}
                    — collision caveat: hypoxanthine and uric acid share the 720 cm⁻¹ ring breathing band.</li>
                <li><b>G04 Nucleic acid phosphate:</b> {bsv['G04_nucleic_acid_phosphate']:.3f}
                    — small supporting contribution.</li>
              </ul>
              <div class='muted'>
              Specificity here remains <b>class-level</b> — only the purine ring-breathing motif fires
              consistently in these single-substrate single-laser SERS measurements; the demo does NOT
              promote this to a molecule-level adenine call without corroborating co-bands at 1335 + 1485 cm⁻¹.
              Real spectra: see <code>{_adenine_dir}/{row['source_file']}</code>.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── C. Uric Acid / Isotope Validation ──────────────────────────
    with tabs[2]:
        st.subheader("Uric acid — purine-metabolite validation (real contrasts)")
        ua, ph = dl.load_uric_acid_validation()
        _placeholder_badge(ph)
        if not ph:
            st.caption(
                "Source: `streamlit_apps/gaira_demo/data/calibration_conditions.csv` "
                "+ `calibration_delta_bsv.csv` — real per-axis Δ BSV from the SAEL "
                "calibration layer, remapped from legacy 8-axis to 11-axis. **No isotope "
                "(¹⁵N) data exists in the corpus** — the previous demo isotope condition "
                "was fabricated and has been removed."
            )

        cond_id = st.selectbox(
            "Condition",
            options=ua["condition_id"].tolist(),
            format_func=lambda c: ua[ua["condition_id"] == c]["display_name"].iloc[0],
            index=0,
        )
        row = ua[ua["condition_id"] == cond_id].iloc[0]
        delta = {a: float(row[f"delta_{a}"]) for a in cfg.BSV_AXES}

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("n control",      int(row["n_control"]))
        c2.metric("n perturbed",   int(row["n_perturbed"]))
        c3.metric("SAEL confidence", str(row["confidence"]))
        c4.metric("Verdict label",   str(row["label"]))

        col1, col2 = st.columns([1, 1])
        with col1:
            # Synthesised reference spectrum stays as illustration, clearly tagged
            wn, y = dl.synth_reference_spectrum("uric_acid", seed=3)
            fig = gp.spectrum_figure(
                [{"x": wn, "y": y, "name": "uric acid (synth. reference)", "color": "#FB923C"}],
                title="Uric acid reference spectrum (synthesised, illustrative)",
                highlight_anchors=list(dl.MOLECULES["uric_acid"].anchors),
                height=320,
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})
            st.caption(
                "Spectrum is illustrative only. The Δ BSV bar below is real, "
                "computed from the SAEL contrast for the selected condition."
            )
        with col2:
            # Δ BSV bar — dynamic axis range to show small but real shifts
            st.plotly_chart(
                gp.delta_bar_figure(
                    delta,
                    title=f"ΔBSV — {row['display_name']}",
                    height=380,
                ),
                use_container_width=True, config={"displayModeBar": False},
            )

        # Per-axis verdict table (real SAEL output)
        st.markdown("**Per-axis SAEL verdict (real values)**")
        verdict_rows = []
        for a in cfg.BSV_AXES:
            v = str(row[f"verdict_{a}"])
            d = float(row[f"delta_{a}"])
            exp = str(row[f"expected_direction_{a}"])
            if v == "not_testable" and abs(d) < 1e-6:
                continue
            verdict_rows.append({
                "axis":               cfg.axis_long(a),
                "Δ BSV (remapped)":   round(d, 5),
                "expected direction": exp,
                "verdict":            v,
            })
        if verdict_rows:
            st.dataframe(pd.DataFrame(verdict_rows), hide_index=True,
                          use_container_width=True)

        st.markdown(
            f"""
            <div class='card'>
              <h4>Honest interpretation</h4>
              <div class='muted'>
              <b>Hypoxanthine spike (serum):</b> Δ G01/G02 (remapped from legacy purine_nucleotide)
              rises by the expected sign — SAEL records this as <b>agree</b>. Effect size is small
              (Δ ≈ 0.007 across remap), consistent with class-level (not molecule-level) call.<br>
              <b>Uricase depletion (Sigma serum):</b> SAEL flags this as <b>inconsistent</b> —
              several axes moved opposite to the literature-expected direction, including
              purine_nucleotide. Honest evidence is that this contrast did <i>not</i> behave as
              expected; possible explanations include serum-matrix variability and substrate effects.
              GAIRA surfaces both behaviors rather than hiding the disagreement.<br>
              <b>Hypoxanthine spike + uricase:</b> purine_nucleotide rises as expected
              (Δ ≈ 0.006), agree.<br>
              <b>Substrate caveat:</b> Carotenoid 1517 cm⁻¹ overlaps uric-acid 1517 in serum
              SERS — GAIRA's substrate rule soft-dampens G02 specificity in carotenoid-rich matrices.<br>
              <b>Not in this dataset:</b> isotope (¹⁵N / ¹³C) validation — no such spectra exist in
              the GAIRA corpus. A future calibration run would need to add isotope-spiked spectra
              before that claim can be made.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


    # ── D. Axis Coverage Summary (V3) ───────────────────────────────
    with tabs[3]:
        st.subheader("Axis reference-space coverage")
        v3v.render_axis_coverage()


# ════════════════════════════════════════════════════════════════════
# MODE 3 — GLOBAL BIOLOGICAL PROJECTION (V3)
# ════════════════════════════════════════════════════════════════════

elif mode == "Global Biological Projection":
    v3v.render_global_projection()


# ════════════════════════════════════════════════════════════════════
# MODE 4 — COORDINATE VALIDATION (V3)
# ════════════════════════════════════════════════════════════════════

elif mode == "Coordinate Validation":
    v3v.render_validation()


# ════════════════════════════════════════════════════════════════════
# (legacy) V2 biological pilot body — retained for reference; not selectable
# in V3 (superseded by Mode 3 Global Biological Projection). Guard never matches.
# ════════════════════════════════════════════════════════════════════

elif mode == "__legacy_v2_biological_pilot__":
    tabs = st.tabs(["Serum Liver Disease", "EV Diabetes", "SHINE Liver Injury / Hepatotoxicity"])

    PILOTS = {
        "Serum Liver Disease":             ("serum_liver",         "Healthy"),
        "EV Diabetes":                     ("ev_diabetes",         "Normal-Weight Control"),
        "SHINE Liver Injury / Hepatotoxicity": ("shine_liver_injury", "Day 0 · C0"),
    }
    DESCRIPTIONS = {
        "Serum Liver Disease": (
            "Serum biochemical state across healthy, HCC, CCA, and liver-metastases cohorts. "
            "GAIRA reports cohort-level biochemical shifts; not a disease classifier."
        ),
        "EV Diabetes": (
            "Extracellular vesicle biochemical state across normal-weight control, "
            "overweight diabetic, and normal-weight diabetic cohorts."
        ),
        "SHINE Liver Injury / Hepatotoxicity": (
            "Real APAP-dose × day matrix from the SHINE EV-SERS hepatotoxicity pilot. "
            "Only Day 0 and Day 2 timepoints are available in the corpus (× C0/C10/C20/C40 dose). "
            "GAIRA reports state shifts consistent with hepatotoxicity-associated biochemical "
            "progression; this is not a diagnostic classifier."
        ),
    }

    for tab, (pilot_label, (pilot_key, ref_cohort_default)) in zip(tabs, PILOTS.items()):
        with tab:
            st.subheader(pilot_label)
            st.markdown(f"<p class='caption-muted'>{DESCRIPTIONS[pilot_label]}</p>",
                            unsafe_allow_html=True)
            df, ph = dl.load_pilot_cohorts(pilot_key)
            _placeholder_badge(ph)
            if pilot_key == "serum_liver" and not ph:
                # Detect whether the spectra-based loader fired (n_sampled column present)
                if "n_sampled" in df.columns:
                    n_sampled = int(df["n_sampled"].iloc[0])
                    n_axes_nz = int(df["n_nonzero_axes"].iloc[0])
                    st.caption(
                        f"Source: `pilot4_1_cca_hcc_lm_serum_patient_level/tables/patient_level_mean_spectra.csv` "
                        f"(real preprocessed serum mean spectra, 400–1800 cm⁻¹ at 1 cm⁻¹). "
                        f"**{n_sampled} patients × 4 cohorts** stratified-sampled and run through the demo's own "
                        f"motif/MSS/substrate pipeline (`build_report`) to produce real 11-axis BSV. "
                        f"All {n_axes_nz}/11 axes light up because the demo pipeline is designed for "
                        f"multi-axis biology — unlike the autoresearch v1 BSV exports which collapse "
                        f"onto 3 axes by construction (a known upstream limitation, documented in "
                        f"[BIOLOGICAL_PILOT_BSV_AUDIT.md](./BIOLOGICAL_PILOT_BSV_AUDIT.md))."
                    )
                else:
                    st.caption(
                        "Source: `pilot4_1_cca_hcc_lm_serum_patient_level/tables/patient_level_bsv.csv` "
                        "(autoresearch BSV — sparse 3-axis upstream)."
                    )
            elif pilot_key == "ev_diabetes" and not ph:
                if "n_sampled" in df.columns:
                    n_sampled = int(df["n_sampled"].iloc[0])
                    n_axes_nz = int(df["n_nonzero_axes"].iloc[0])
                    st.caption(
                        f"Source: `pilot2_target_validation_v1/tables/sample_query_spectra.csv` "
                        f"(real EV mean spectra with wavenumbers_json + intensity_json, 500–1600 cm⁻¹ "
                        f"interpolated to the demo's 400–1800 grid). "
                        f"**{n_sampled} samples × 2 cohorts** stratified-sampled and run through the demo's "
                        f"own pipeline → real 11-axis BSV ({n_axes_nz}/11 axes lit). "
                        f"Cohorts are autoresearch project-specific labels — **Impact** (n=39) and "
                        f"**Strong-D** (n=24); not a generic Normal vs Diabetic split."
                    )
                else:
                    st.caption(
                        "Source: `pilot2_target_validation_v1/tables/class_mean_bsv.csv` "
                        "(autoresearch BSV — sparse 3-axis upstream)."
                    )
            if pilot_key == "shine_liver_injury" and not ph:
                st.caption(
                    "Sources: `pilot3_shine_single_set_day0_day2/tables/class_mean_bsv_day0_day2.csv` "
                    "(real per-class means) + `pilot3_shine_ev_sers/tables/per_sample_bsv.csv` (real "
                    "per-sample BSVs covering Day 0 + Day 1 + Day 2). **Known upstream limitation:** "
                    "the autoresearch v1 BSV is a structural 3-axis projection (100% of 15,027 SHINE "
                    "spectra fire on exactly G04 + G11 + a substrate-bias axis — verified in the audit doc). "
                    "Unlike serum_liver and ev_diabetes, SHINE has no preprocessed mean-spectra file in the "
                    "autoresearch outputs (only deeply-nested raw `s_N` per-scan files in Set9/Set10), so "
                    "the demo cannot rerun SHINE through its own 11-axis pipeline without a preprocessing "
                    "step. The radar therefore shows the autoresearch 3-axis projection verbatim. The "
                    "richer autoresearch family fingerprint (purine_core / methylated_purine / guanidine) "
                    "is shown in the expander below for a complementary view."
                )

            cohorts = df["cohort"].tolist()
            col_a, col_b = st.columns(2)
            with col_a:
                ref_cohort = st.selectbox(
                    f"Reference cohort ({pilot_label})", cohorts,
                    index=cohorts.index(ref_cohort_default) if ref_cohort_default in cohorts else 0,
                    key=f"ref_{pilot_key}",
                )
            with col_b:
                others = [c for c in cohorts if c != ref_cohort]
                cond_cohort = st.selectbox(
                    f"Condition cohort ({pilot_label})", others, index=0,
                    key=f"cond_{pilot_key}",
                )

            ref_row = df[df["cohort"] == ref_cohort].iloc[0]
            cond_row = df[df["cohort"] == cond_cohort].iloc[0]
            ref_bsv = {a: float(ref_row[a]) for a in cfg.BSV_AXES}
            cond_bsv = {a: float(cond_row[a]) for a in cfg.BSV_AXES}
            delta = {a: cond_bsv[a] - ref_bsv[a] for a in cfg.BSV_AXES}

            # Dynamic radial max — real SHINE values are small (~0.0–0.5 on the
            # populated axes) and even smaller on the others; fixed scale would
            # hide cohort differences.
            radial_max = max(
                max(ref_bsv.values()),
                max(cond_bsv.values()),
                max(float(df[a].max()) for a in cfg.BSV_AXES),
            )
            radial_max = max(0.10, radial_max * 1.15)

            # Top-3 axes per cohort summary above the radar
            def _top3(bsv: dict, k: int = 3) -> str:
                items = [(cfg.axis_short(a), v) for a, v in bsv.items() if v > 1e-4]
                items.sort(key=lambda kv: -kv[1])
                return " · ".join(f"{a} {v:.2f}" for a, v in items[:k]) or "—"

            ref_n_label = (f"n_sample={int(ref_row.get('n_sampled', ref_row.get('n', 0)))}"
                              f" / cohort={int(ref_row.get('n_cohort_total', ref_row.get('n', 0)))}")
            cond_n_label = (f"n_sample={int(cond_row.get('n_sampled', cond_row.get('n', 0)))}"
                              f" / cohort={int(cond_row.get('n_cohort_total', cond_row.get('n', 0)))}")
            tcol1, tcol2 = st.columns(2)
            with tcol1:
                st.markdown(f"**{ref_cohort}** ({ref_n_label})<br>"
                              f"<span class='caption-muted'>top axes: {_top3(ref_bsv)}</span>",
                              unsafe_allow_html=True)
            with tcol2:
                st.markdown(f"**{cond_cohort}** ({cond_n_label})<br>"
                              f"<span class='caption-muted'>top axes: {_top3(cond_bsv)}</span>",
                              unsafe_allow_html=True)

            col1, col2 = st.columns([1, 1])
            with col1:
                st.plotly_chart(
                    gp.radar_figure(
                        [
                            {"name": f"{ref_cohort} (n={int(ref_row['n'])})",
                              "values": ref_bsv, "color": "#60A5FA"},
                            {"name": f"{cond_cohort} (n={int(cond_row['n'])})",
                              "values": cond_bsv, "color": "#F87171"},
                        ],
                        title=f"Cohort BSV overlay (radial scale 0–{radial_max:.2f})",
                        height=460, radial_max=radial_max,
                    ),
                    use_container_width=True, config={"displayModeBar": False},
                )
            with col2:
                st.plotly_chart(
                    gp.delta_bar_figure(
                        delta, title=f"ΔBSV — {cond_cohort} vs {ref_cohort}", height=460,
                    ),
                    use_container_width=True, config={"displayModeBar": False},
                )

            # Show autoresearch native axes for any pilot loaded from autoresearch outputs
            if not ph and pilot_key in ("shine_liver_injury", "serum_liver", "ev_diabetes"):
                with st.expander("Autoresearch 8-axis raw values (pre-remap)", expanded=False):
                    auto_cols = [c for c in df.columns if c.startswith("autoresearch_")]
                    if auto_cols:
                        meta_cols = [c for c in ("cohort", "day", "dose", "n", "n_scans", "raw_label") if c in df.columns]
                        st.dataframe(
                            df[meta_cols + auto_cols].reset_index(drop=True),
                            hide_index=True, use_container_width=True,
                        )
                    else:
                        st.caption(
                            "Autoresearch raw axes are not present in this load — for serum_liver / "
                            "ev_diabetes the demo bypassed the sparse 3-axis autoresearch BSV and "
                            "recomputed real 11-axis BSV from preprocessed spectra via the demo's "
                            "own motif/MSS/substrate pipeline."
                        )

                # Family-fingerprint complementary view (richer biological signal
                # the autoresearch pipeline does produce — orthogonal to the v11 ontology,
                # NOT remapped, shown verbatim so user can compare).
                ff_loader = {
                    "shine_liver_injury": dl.load_shine_family_fingerprint,
                    "serum_liver":         dl.load_liver_family_fingerprint,
                    "ev_diabetes":         dl.load_ev_diabetes_family_fingerprint,
                }[pilot_key]
                ff_df, ff_ph = ff_loader()
                if not ff_ph and not ff_df.empty:
                    with st.expander("Autoresearch family fingerprint (complementary biological view)",
                                       expanded=False):
                        st.caption(
                            "These are the **autoresearch v1 family fractions** for each cohort — a "
                            "different ontology than GAIRA's 11 BSV axes. Note all biological pilots "
                            "collapse onto the same 3 families (purine_core / methylated_purine / "
                            "guanidine); this is an autoresearch-pipeline characteristic, not GAIRA's."
                        )
                        st.dataframe(ff_df.round(3), hide_index=True, use_container_width=True)

            # Top shifted axes
            top = sorted(delta.items(), key=lambda kv: -abs(kv[1]))[:4]
            st.markdown("**Top shifted axes**")
            cols = st.columns(len(top))
            for col, (axis, val) in zip(cols, top):
                col.metric(cfg.axis_short(axis), f"{val:+.3f}",
                            delta=f"{val:+.3f}")

            # Evidence + caveats panels
            colA, colB = st.columns(2)
            with colA:
                st.markdown("**Evidence**")
                for axis, val in top:
                    if abs(val) < 0.04:
                        continue
                    direction = "increased" if val > 0 else "decreased"
                    strength = "high" if abs(val) >= 0.20 else ("moderate" if abs(val) >= 0.08 else "low")
                    st.markdown(
                        f"- *{cfg.axis_long(axis)}* — {direction} (Δ {val:+.3f}) "
                        f"<span class='pill pill-ok' style='font-size:0.74rem;'>evidence {strength}</span>",
                        unsafe_allow_html=True,
                    )
            with colB:
                st.markdown("**Caveats**")
                # Generic caveats for biological pilots
                if pilot_key == "serum_liver":
                    st.markdown(
                        "- Canonical cohort sizes: <b>212 unique patients</b> — HA 48 / CCA 66 / "
                        "HCC 49 / LM 49 (reconciled 2026-07-15). The radar is computed from "
                        "<code>patient_level_mean_spectra.csv</code> (one mean spectrum per patient, "
                        "212 rows). The sibling <code>patient_level_bsv.csv</code> has 213 rows only "
                        "because CCA patient <code>SER-CCA-58</code> carries a duplicate measurement "
                        "row; no table contains 214. See "
                        "<code>reports/SERUM_LIVER_PROVENANCE_RECONCILIATION_2026-07-15.md</code>.<br>"
                        "- Serum is a mixture matrix — class-level interpretation only.<br>"
                        "- Autoresearch ontology fires on only 6 of 8 axes for this dataset "
                        "(protein_peptide and lipid_membrane are 0 in all patient rows); the demo "
                        "11-axis radar inherits that sparsity.<br>"
                        "- Cohort means are very close to each other on the aggregated axes; per-patient "
                        "variance and motif-level structure (not visible here) carry the discriminative signal.<br>"
                        "- GAIRA does not classify disease from BSV alone; the demo shows biochemical "
                        "state shifts <i>consistent with</i> the cohort, not a diagnosis.",
                        unsafe_allow_html=True,
                    )
                elif pilot_key == "ev_diabetes":
                    # v2 fix: name the loader that ACTUALLY ran. The spectra→build_report
                    # path (n_sampled present) recomputes real 11-axis BSV from
                    # sample_query_spectra.csv; only the fallback reads class_mean_bsv.csv.
                    if "n_sampled" in df.columns:
                        ev_source_line = (
                            "- Real 11-axis BSV recomputed from "
                            "<code>pilot2_target_validation_v1/tables/sample_query_spectra.csv</code> "
                            "(per-sample EV spectra) via the demo's own motif/MSS/substrate pipeline. "
                            "Two cohorts: <b>Impact</b> (n=39) and <b>Strong-D</b> (n=24) — project-specific labels.<br>"
                        )
                    else:
                        ev_source_line = (
                            "- Real per-class mean BSV from "
                            "<code>pilot2_target_validation_v1/tables/class_mean_bsv.csv</code> "
                            "(autoresearch 3-axis fallback). "
                            "Two cohorts: <b>Impact</b> (n=39) and <b>Strong-D</b> (n=24) — project-specific labels.<br>"
                        )
                    st.markdown(
                        ev_source_line +
                        "- EV pellet composition is heterogeneous — interpret BSV at cohort level.<br>"
                        "- The 11-axis BSV is a transparent band-evidence heuristic (11 curated motifs), "
                        "not a calibrated model — cohort deltas are exploratory and composition-relative.<br>"
                        "- This is NOT a Normal vs Diabetic split; relabel only with project documentation in hand.",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "- Only **Day 0 and Day 2** are available in the SHINE per-class means file — "
                        "no Day 3 or Day 7 data exists; do not extrapolate.<br>"
                        "- Sample n per cohort is small (2–7 patients per (day, dose) cell) but real — "
                        "per-cohort n_scans (882–3,999 spectra) shown in the autoresearch expander below.<br>"
                        "- Δ across doses C0→C40 is small in absolute terms; biology lives in the "
                        "ratio between dose levels, not absolute magnitudes.<br>"
                        "- Autoresearch ontology splits SERS-substrate effects into separate axes "
                        "(matrix_background, substrate_adsorption_bias, protocol_sensitive_signal) — "
                        "these are not biology and are reported as caveats, not BSV.<br>"
                        "- A separate Day-1 cohort (n=4 patients) also exists in the per-sample file "
                        "but is not surfaced here because the per-class means file does not include it; "
                        "rerun `pilot3_shine_single_set_day0_day2` with Day 1 inclusion to add it.",
                        unsafe_allow_html=True,
                    )

            st.markdown(
                f"<div class='card'><h4>Interpretation language</h4>"
                f"<div class='muted'>GAIRA identifies biochemical state shifts <b>consistent with</b> "
                f"<i>{cond_cohort}</i> relative to <i>{ref_cohort}</i>. "
                f"This is not a diagnostic call; it is a class-level biochemical state assessment "
                f"based on cohort averages of n = {int(cond_row['n'])} vs n = {int(ref_row['n'])} samples.</div></div>",
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<p class='caption-muted'>GAIRA Scientific Reasoning Demo v2 (migration-hardened) — "
    "evidence-grounded biochemical state estimation from Raman/SERS spectra. "
    "Class-level interpretation by default · substrate-aware physics · "
    "explicit caveats and ambiguity routing.</p>",
    unsafe_allow_html=True,
)
