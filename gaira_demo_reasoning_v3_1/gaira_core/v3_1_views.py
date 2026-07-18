"""GAIRA Demo v3.1 — redesigned Mode-3 (Global Biological Projection).

Fixes carried in this module:
  * Signed coordinates are shown on a DIVERGING horizontal plot with a zero
    reference line and symmetric scale — never on a zero-origin radar that
    hides/clips negatives.
  * EV-diabetes: default = cohort-relative biochemical effect profile
    (historical pooled z-score, reproduced exactly from the audited 1322
    analysis); secondary = frozen Ag-SERS reference coordinates; raw kept as an
    expandable technical view; explicit provenance.
  * SHINE: reduced-dimensional legacy presentation (no 11-axis radar).
  * Naming: frozen coordinates renamed to reflect the Ag-SERS fit population.

The frozen calibration values are unchanged from V3; this is a visualization,
naming, and provenance correction only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import config as cfg
from . import plotting as gp
from . import global_coordinates as gc
from . import coordinate_validation as cv
from . import v3_views  # reuse cross-dataset map + nuisance

REFERENCE_COORD_NAME = "GAIRA Frozen Biological Ag-SERS Reference Coordinates v1"
REFERENCE_COORD_SHORT = "GAIRA Ag-SERS Reference Coordinates v1"

EQUIV_DIR = cfg.GENERATED_DIR / "diabetes_equivalence"
AXES = list(cfg.BSV_AXES)


def _short(a):  # cautious "-associated" axis names
    return cfg.axis_short(a) + "-assoc."


# ─────────────────────────────────────────────────────────────────────
# Signed diverging visualization (replaces nonnegative radar for signed coords)
# ─────────────────────────────────────────────────────────────────────

def diverging_figure(cohort_values: dict[str, dict[str, float]], *, title: str,
                     xlabel: str, height: int = 460) -> go.Figure:
    """cohort_values: {cohort_name: {axis: signed_value}}. One row per axis;
    grouped horizontal bars per cohort; zero reference line; symmetric x-range."""
    fig = go.Figure()
    labels = [_short(a) for a in AXES]
    allv = [v for m in cohort_values.values() for v in m.values() if np.isfinite(v)]
    xmax = max(0.1, max(abs(x) for x in allv) * 1.15) if allv else 1.0
    for i, (coh, m) in enumerate(cohort_values.items()):
        fig.add_trace(go.Bar(
            y=labels, x=[float(m.get(a, 0.0)) for a in AXES], orientation="h",
            name=str(coh), marker_color=cfg.OVERLAY_COLORS[i % len(cfg.OVERLAY_COLORS)],
            opacity=0.85))
    fig.add_vline(x=0, line=dict(color=cfg.AXIS_LINE_COLOR, width=1.5))
    fig.update_layout(barmode="group")
    gp.apply_dark(fig, title=title, height=height)
    fig.update_xaxes(title=xlabel, range=[-xmax, xmax], zeroline=True)
    fig.update_yaxes(autorange="reversed")
    return fig


# ─────────────────────────────────────────────────────────────────────
# Equivalence-artifact loaders (bundled)
# ─────────────────────────────────────────────────────────────────────

def _load_norm_variant(variant: str) -> dict[str, dict[str, float]] | None:
    p = EQUIV_DIR / "normalization_variants.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    sub = df[df["variant"] == variant]
    if sub.empty:
        return None
    out: dict[str, dict[str, float]] = {}
    for coh, g in sub.groupby("cohort"):
        out[str(coh)] = {r["axis"]: float(r["value"]) for _, r in g.iterrows()}
    return out


def _load_stats() -> pd.DataFrame | None:
    """Prefer the EXACT historical 2-group stats (from the audited 1322 folder);
    fall back to the V3.1-recomputed Path A stats."""
    exact = EQUIV_DIR / "historical_2group_stats_exact.csv"
    if exact.exists():
        df = pd.read_csv(exact)
        # normalize column names to the shared schema
        ren = {"axis_label": "axis_short"}
        df = df.rename(columns=ren)
        if "axis_short" not in df.columns and "axis" in df.columns:
            df["axis_short"] = df["axis"].map(cfg.axis_short)
        return df
    p = EQUIV_DIR / "diabetes_2group_stats_pathA_v31.csv"
    return pd.read_csv(p) if p.exists() else None


# ─────────────────────────────────────────────────────────────────────
# Mode 3
# ─────────────────────────────────────────────────────────────────────

def render_global_projection():
    st.header("Mode 3 — Global Biological Projection")
    df_ref = cv.load_reference_samples()
    calib = gc.load_calibration()
    if df_ref is None or calib is None:
        st.error("GLOBAL COORDINATE UNAVAILABLE — projection artifacts missing. "
                 "Coordinates are never refit at runtime.")
        return
    tabs = st.tabs(["EV Diabetes", "Serum Liver", "SHINE", "Cross-dataset map", "Nuisance diagnostics"])
    with tabs[0]:
        _render_ev_diabetes(df_ref, calib)
    with tabs[1]:
        _render_serum(df_ref, calib)
    with tabs[2]:
        _render_shine_reduced()
    with tabs[3]:
        v3_views._render_cross_dataset_map(df_ref, calib)
    with tabs[4]:
        v3_views._render_nuisance(df_ref)


def _render_ev_diabetes(df_ref, calib):
    st.subheader("EV diabetes — biochemical effect profile")
    st.caption("Two distinct questions are separated below: **within-cohort effect** "
               "(default) and **frozen absolute position** (secondary). Cohorts: "
               "**OWD** (Impact, n=39) vs **NWD** (Strong-D, n=24). Labels are the study-"
               "design cohorts (Impact→OWD, Strong-D→NWD); see provenance.")

    # ── PRIMARY: cohort-relative biochemical effect profile (historical z-score) ──
    st.markdown("### Cohort-relative biochemical effect profile  ·  *default*")
    st.caption("Each biochemical axis is standardized within the pooled diabetes EV cohort "
               "(z = (cohort_mean − pooled_mean) / pooled_SD). This view highlights multiaxis "
               "group differences but is **not** directly comparable across datasets. "
               "Reproduced exactly (≤1e-9) from the audited 20260701_1322 analysis, which uses "
               "a diabetes-audit engine override on the redox axis (tightened thione window + "
               "co-band-gated thiol boost). All other axes match the demo engine.")
    hz = _load_norm_variant("historical_cohort_z_exact") or _load_norm_variant("historical_cohort_z")
    if hz:
        st.plotly_chart(diverging_figure(hz, title="Cohort-relative biochemical effect profile (pooled z-score)",
                                         xlabel="z (deviation from pooled diabetes cohort)"),
                        use_container_width=True, config={"displayModeBar": False})
    stats = _load_stats()
    if stats is not None:
        st.markdown("**Per-axis group effect (OWD vs NWD) — Path A engine, reproduced**")
        show = stats.copy()
        show["axis"] = show["axis_short"] + "-assoc."
        show = show[["axis", "mean_OWD", "mean_NWD", "delta_OWD_minus_NWD", "cohens_d",
                     "mannwhitney_U", "p_value", "q_value_fdr_bh"]].round(4)
        st.dataframe(show, hide_index=True, use_container_width=True)
        st.caption("Cohen's d, Mann–Whitney U, Benjamini–Hochberg q. Strongest effects are "
                   "sterol/neutral-lipid- and metabolic-small-molecule-associated, not redox — "
                   "the pooled z-score reveals multiaxis structure the raw redox-dominated "
                   "magnitude radar hides. Statistical separation is not biochemical validation.")

    # ── SECONDARY: frozen Ag-SERS reference coordinates ──
    st.markdown(f"### {REFERENCE_COORD_SHORT}  ·  *secondary (absolute position)*")
    st.caption(f"These coordinates are fixed across cohorts and show position relative to the "
               f"frozen V3 Ag-SERS biological reference population (275 serum+EV Ag-SERS spectra). "
               f"They are **not** axis-balanced and may retain matrix or dataset effects. "
               f"Signed robust-z, shown on a diverging scale (not a zero-origin radar).")
    ev = df_ref[df_ref.dataset == "ev_diabetes"]
    gmap = {}
    for lbl, disp in [("Impact", "OWD (Impact)"), ("Strong-D", "NWD (Strong-D)")]:
        sub = ev[ev.label == lbl]
        if len(sub):
            gmap[disp] = {a: float(sub[f"global_{a}"].mean()) for a in AXES}
    if gmap:
        st.plotly_chart(diverging_figure(gmap, title=f"{REFERENCE_COORD_SHORT} (signed robust-z)",
                                         xlabel="global coordinate (robust σ from Ag-SERS reference median)"),
                        use_container_width=True, config={"displayModeBar": False})

    # ── raw heuristic BSV (expandable technical view) ──
    with st.expander("Raw heuristic band-evidence BSV (technical view)"):
        raw = _load_norm_variant("raw")
        if raw:
            st.plotly_chart(diverging_figure(raw, title="Raw heuristic BSV cohort means (nonnegative)",
                                             xlabel="raw BSV (band-evidence, nonnegative)", height=420),
                            use_container_width=True, config={"displayModeBar": False})
            st.caption("Raw band-evidence BSV is nonnegative and redox-dominant by magnitude; "
                       "this is why the historical raw radar looked sparse/redox-heavy.")

    # ── provenance / methods ──
    with st.expander("Provenance & methods (source paths, label mapping, formulas, differences)"):
        st.markdown(
            "**Source analysis:** `results/diabetes_gaira_audit_20260701_1322` (later) and "
            "`…_1304` (earlier).\n\n"
            "**Source BSV table:** `diabetes_gaira_scores_per_sample.csv` (historical, engine = "
            "`analysis/_diabetes_overrides.build_report_diabetes`).\n\n"
            "**Current recomputed BSV:** V3.1 `tools/build_diabetes_equivalence.py` "
            "(`data/generated/diabetes_equivalence/`).\n\n"
            "**Label mapping (proven):** Impact → OWD (n=39 with spectra / 40 patients), "
            "Strong-D → NWD (n=24). Direct `group_raw→group_2` map "
            "(`run_diabetes_gaira_audit.py:205`); a `bmi≥25→OWD` rule exists but was NOT used "
            "for `group_2`. **Confirmed equivalent** (relabel of study-design cohorts).\n\n"
            "**Cohort-relative formula:** `z = (cohort_mean − pooled_mean) / pooled_SD` "
            "(ddof=1, pooled over all 63 samples). Reproduced to ≤1e-9.\n\n"
            "**Frozen global formula:** `global = (raw_bsv − center)/scale`, robust median/MAD, "
            "fit label-free on 275 Ag-SERS biological spectra (unchanged from V3).\n\n"
            "**Difference between historical and V3 BSV:** confined to the **G10 redox axis** "
            "(historical tightens the thione window to 490–505 and gates the thiol boost on the "
            "720 cm⁻¹ imidazole co-band). All 10 other axes are identical. The historical "
            "**balanced** radar comes from the **z-score normalization**, not from a different "
            "engine.")


def _render_serum(df_ref, calib):
    st.subheader("Serum liver — frozen reference position")
    serum = df_ref[df_ref.dataset == "serum_liver"]
    st.caption(f"{len(serum)} patients (HA/CCA/HCC/LM). Signed frozen coordinates on a diverging "
               f"scale. Serum Ag-SERS; class-level only.")
    view = st.radio("Coordinate system", [REFERENCE_COORD_SHORT, "Cohort-relative (within serum)"],
                    horizontal=True, key="serum_view")
    if view == REFERENCE_COORD_SHORT:
        gmap = {str(c): {a: float(g[f"global_{a}"].mean()) for a in AXES}
                for c, g in serum.groupby("label")}
        xlabel = "global coordinate (robust σ from Ag-SERS reference median)"
    else:
        raws = [{a: float(r[f"raw_{a}"]) for a in AXES} for _, r in serum.iterrows()]
        z = gc.cohort_relative_zscores(raws)
        zdf = pd.DataFrame(z); zdf["label"] = serum["label"].values
        gmap = {str(c): {a: float(g[a].mean()) for a in AXES} for c, g in zdf.groupby("label")}
        xlabel = "cohort-relative z (within serum — not globally comparable)"
    st.plotly_chart(diverging_figure(gmap, title=f"Serum liver — {view}", xlabel=xlabel, height=480),
                    use_container_width=True, config={"displayModeBar": False})


def _render_shine_reduced():
    st.subheader("SHINE — Legacy reduced-dimensional response")
    from . import data_loader as dl
    st.caption("**SHINE has no reconstructable per-sample/mean spectra** (only ~15,027 deeply-"
               "nested raw scans in a zip). It therefore CANNOT be recomputed through the demo's "
               "11-axis engine. The autoresearch BSV is a collapsed low-dimensional projection "
               "(only ~2–3 axes carry signal). The full 11-axis radar has been **removed** for "
               "SHINE to avoid implying 11 independent measured axes.")
    sh, ph = dl.load_pilot_cohorts("shine_liver_injury")
    if ph or sh is None or sh.empty:
        st.warning("SHINE data unavailable (placeholder mode).")
        return
    # Show only the actually-active autoresearch axes as a dose × time heatmap.
    auto_cols = [c for c in sh.columns if c.startswith("autoresearch_")]
    active = [c for c in auto_cols if sh[c].abs().sum() > 1e-6]
    st.markdown("**Active source dimensions (autoresearch ontology) — dose × time**")
    if active and "day" in sh.columns and "dose" in sh.columns:
        hm = sh.copy()
        hm["cell"] = "D" + hm["day"].astype(str) + "·C" + hm["dose"].astype(str)
        z = hm.set_index("cell")[active]
        fig = go.Figure(go.Heatmap(z=z.to_numpy().T, x=list(z.index),
                                   y=[c.replace("autoresearch_", "") for c in active],
                                   colorscale="Viridis"))
        gp.apply_dark(fig, title="SHINE active source axes (legacy autoresearch), dose × time", height=300)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    # dose × time trajectory on the dominant active axis
    if active:
        dom = active[int(np.argmax([sh[c].abs().sum() for c in active]))]
        fig2 = go.Figure()
        for day, g in sh.groupby("day"):
            g = g.sort_values("dose")
            fig2.add_trace(go.Scatter(x=g["dose"], y=g[dom], mode="lines+markers", name=f"Day {day}"))
        gp.apply_dark(fig2, title=f"SHINE dose × time trajectory — {dom.replace('autoresearch_','')}", height=320)
        fig2.update_xaxes(title="APAP dose (C)"); fig2.update_yaxes(title="autoresearch axis value")
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    nz = [int((sh.iloc[i][[a for a in AXES]].abs() > 1e-4).sum()) for i in range(len(sh))]
    st.caption(f"Raw nonzero 11-axis values per cohort: {nz} (collapsed upstream). This is a "
               "**Legacy reduced-dimensional SHINE response**, not a position in the full 11-axis "
               "reference space.")
