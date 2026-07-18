"""GAIRA Demo v3 — new UI render functions (kept out of app.py for clarity).

Every radar states its coordinate system. Global coordinates come ONLY from the
frozen artifact (never refit at runtime). If the artifact is missing the UI
shows GLOBAL COORDINATE UNAVAILABLE and retains the raw heuristic BSV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from . import config as cfg
from . import plotting as gp
from . import global_coordinates as gc
from . import coordinate_validation as cv
from . import domain_context as dctx
from .ontology import ontology

COORD_LABELS = {
    "raw": "Raw heuristic band-evidence BSV",
    "global": "Position in GAIRA Frozen Biological Ag-SERS Reference Coordinates v1",
    "cohort": "Cohort-relative z-score view — not globally comparable",
}


# ─────────────────────────────────────────────────────────────────────
# Shared: cohort-mean coordinate frames from the frozen reference samples
# ─────────────────────────────────────────────────────────────────────

def _pilot_frame(pilot_key: str, df_ref: pd.DataFrame):
    """Return (per_sample_df, dataset_label, cohort_col) for a pilot, or None."""
    ds = {"serum_liver": "serum_liver", "ev_diabetes": "ev_diabetes"}.get(pilot_key)
    if ds is None:
        return None
    sub = df_ref[df_ref["dataset"] == ds].copy()
    if sub.empty:
        return None
    return sub, ds, "label"


def _cohort_means(sub: pd.DataFrame, prefix: str) -> dict[str, dict]:
    out = {}
    for coh, g in sub.groupby("label"):
        out[str(coh)] = {a: float(g[f"{prefix}{a}"].mean()) for a in cfg.BSV_AXES}
    return out


# ─────────────────────────────────────────────────────────────────────
# Mode 1 additions — ontology + coordinate construction
# ─────────────────────────────────────────────────────────────────────

def render_ontology_panel():
    onto = ontology()
    st.subheader("GAIRA Biochemical Ontology v1")
    st.markdown(
        "<p class='caption-muted'>Disease-label-independent interpretable axes. "
        "Axis meaning comes from spectral motifs + curated MSS analytes + the "
        "202-molecule reference family mapping — never from disease labels. "
        "The global-coordinate <i>scale</i> is a separate frozen calibration and "
        "does not redefine axis meaning.</p>", unsafe_allow_html=True)
    st.info(onto.notes.strip())
    rows = []
    for a in onto.axes:
        rows.append({
            "axis": a.id, "display_name": a.display_name,
            "grounding_status": a.grounding_status,
            "from_split": a.from_legacy_split,
            "legacy_source": a.legacy_source_axis,
            "motifs": ", ".join(a.contributing_motifs),
            "mss_analytes": ", ".join(a.contributing_mss_analytes) or "—",
            "evidence_confidence": a.evidence_confidence,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    with st.expander("Per-axis detail (collisions, substrate sensitivity, limitations)"):
        for a in onto.axes:
            st.markdown(
                f"**{a.display_name}** (`{a.id}`) — *{a.grounding_status}*  \n"
                f"{a.interpretation}  \n"
                f"Collisions: {'; '.join(a.known_collisions) or '—'}  \n"
                f"Substrate: {'; '.join(a.substrate_sensitivities) or '—'}  \n"
                f"Limitations: {a.limitations}")
    st.caption(
        "Grounding status vocabulary: independently_grounded (1:1 legacy + own "
        "support) · partially_grounded (1:1 but thin) · derived_split (proportional "
        "split of a shared legacy axis) · insufficiently_grounded (very small pool). "
        "The 11 axes are the first operational system, not assumed final.")


def render_coordinate_construction():
    st.subheader("Raw BSV → Frozen Ag-SERS Reference Coordinates")
    st.caption("Naming note: the frozen scale is fit on 275 biological **Ag-colloid SERS** "
               "spectra. It is a biological Ag-SERS reference, not a universal scale across "
               "Raman, all SERS substrates, matrices, or instruments.")
    calib = gc.load_calibration()
    st.markdown(
        "<p class='caption-muted'>V3 keeps the unchanged V2 raw heuristic BSV and adds a "
        "<b>frozen, versioned</b> global-coordinate layer. The same spectrum receives the "
        "same global coordinates regardless of the comparison cohort.</p>",
        unsafe_allow_html=True)
    st.code(
        "raw_bsv_j            = noisy-OR band evidence (V2 engine, UNCHANGED)\n"
        "global_j (unbounded) = (raw_bsv_j - center_j) / scale_j     # frozen robust z\n"
        "global_j (display)   = clip(global_j, -clip, +clip)          # unbounded preserved\n"
        "  center_j = median(raw_bsv_j)  over frozen biological reference population\n"
        "  scale_j  = 1.4826 * MAD(raw_bsv_j)  (floored)  ->  robust ~sigma", language="text")
    if calib is None:
        st.error("GLOBAL COORDINATE UNAVAILABLE — frozen calibration artifact not found. "
                 "Raw BSV remains available. Coordinates are never refit at runtime.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ontology", calib.ontology_version)
    c2.metric("Calibration", calib.calibration_version)
    c3.metric("Fit population (biological)", calib.n_biological_spectra)
    c4.metric("Calibration spectra", calib.n_calibration_spectra)
    st.caption(f"Transform: {calib.transform} · clip ±{calib.clip} · scale floor {calib.scale_floor}. "
               f"Labels not used in fit. Limitations: {calib.limitations}")
    cal_tbl = pd.DataFrame([
        {"axis": cfg.axis_short(a), "center": round(calib.center[a], 4),
         "scale": round(calib.scale[a], 4),
         "ref_q2.5": round(calib.q_low.get(a, float('nan')), 4),
         "ref_q97.5": round(calib.q_high.get(a, float('nan')), 4)}
        for a in cfg.BSV_AXES])
    st.dataframe(cal_tbl, hide_index=True, use_container_width=True)


def render_axis_coverage():
    st.markdown("**Per-axis reference-space coverage**")
    p = cfg.GENERATED_DIR / "axis_reference_coverage_v1.csv"
    if not p.exists():
        st.caption("Coverage artifact missing — run tools/build_axis_coverage.py.")
        return
    df = pd.read_csv(p, keep_default_na=False)
    st.dataframe(df[["axis_short", "unique_reference_analytes", "calibration_datasets",
                     "biological_datasets_occupying", "ontology_independence_status",
                     "insufficient_grounding_flag"]], hide_index=True, use_container_width=True)
    st.caption("NA = not defensibly mappable at v11 resolution (never shown as 0). "
               "Measured-spectra / source / literature counts are corpus-level (see grounding map).")


# ─────────────────────────────────────────────────────────────────────
# Mode 3 — Global Biological Projection (3 selectable coordinate systems)
# ─────────────────────────────────────────────────────────────────────

def _coord_system_selector(key: str) -> str:
    choice = st.radio(
        "Coordinate system",
        ["Global GAIRA coordinates (default)", "Raw heuristic BSV",
         "Cohort-relative exploratory"],
        key=f"coord_{key}", horizontal=True)
    return {"Global GAIRA coordinates (default)": "global",
            "Raw heuristic BSV": "raw",
            "Cohort-relative exploratory": "cohort"}[choice]


def render_global_projection():
    st.header("Mode 3 — Global Biological Projection")
    df_ref = cv.load_reference_samples()
    calib = gc.load_calibration()
    if df_ref is None or calib is None:
        st.error("GLOBAL COORDINATE UNAVAILABLE — projection artifacts missing. "
                 "Ensure GAIRA_DATA is mounted and run tools/build_global_coordinate_reference.py. "
                 "Coordinates are never refit at runtime.")
        return

    tabs = st.tabs(["EV Diabetes", "Serum Liver", "SHINE", "Cross-dataset map", "Nuisance diagnostics"])
    with tabs[0]:
        _render_pilot("ev_diabetes", df_ref, calib, "extracellular_vesicle", "Ag colloid SERS")
    with tabs[1]:
        _render_pilot("serum_liver", df_ref, calib, "serum", "Ag colloid SERS")
    with tabs[2]:
        _render_shine(calib)
    with tabs[3]:
        _render_cross_dataset_map(df_ref, calib)
    with tabs[4]:
        _render_nuisance(df_ref)


def _render_pilot(pilot_key, df_ref, calib, domain, substrate):
    frame = _pilot_frame(pilot_key, df_ref)
    if frame is None:
        st.caption("No projected samples for this pilot.")
        return
    sub, ds, cohort_col = frame
    n = len(sub)
    st.caption(f"**{n} per-sample spectra** projected through the frozen calibration "
               f"(recomputed raw-spectrum projection). Global coordinates were frozen "
               f"BEFORE any label comparison; disease labels are used only for the "
               f"comparison below, never to fit the calibration.")

    system = _coord_system_selector(pilot_key)
    st.markdown(f"<span class='pill'>{COORD_LABELS[system]}</span>", unsafe_allow_html=True)

    if system == "global":
        means = _cohort_means(sub, "globaldisp_")
    elif system == "raw":
        means = _cohort_means(sub, "raw_")
    else:  # cohort-relative z within this pilot's samples
        raws = [{a: float(r[f"raw_{a}"]) for a in cfg.BSV_AXES} for _, r in sub.iterrows()]
        z = gc.cohort_relative_zscores(raws)
        sub = sub.reset_index(drop=True)
        zdf = pd.DataFrame(z)
        zdf["label"] = sub["label"].values
        means = {str(c): {a: float(g[a].mean()) for a in cfg.BSV_AXES}
                 for c, g in zdf.groupby("label")}

    cohorts = list(means)
    traces = [{"name": f"{c} (n={int((sub[cohort_col]==c).sum())})", "values": means[c]}
              for c in cohorts]
    rmax = None
    if system in ("global", "cohort"):
        allv = [v for m in means.values() for v in m.values()]
        rmax = max(0.5, max(abs(x) for x in allv) * 1.1) if allv else None
    st.plotly_chart(gp.radar_figure(traces, title=f"{COORD_LABELS[system]}",
                                    radial_max=rmax, height=460),
                    use_container_width=True, config={"displayModeBar": False})

    # effect sizes on GLOBAL coords (labels used only post-fit)
    if len(cohorts) >= 2:
        ref = "HA" if "HA" in cohorts else cohorts[0]
        other = [c for c in cohorts if c != ref]
        pick = st.selectbox("Compare cohort vs reference", other, key=f"cmp_{pilot_key}")
        es = cv.group_effect_sizes(sub, cohort_col, pick, ref, coord_prefix="global_")
        st.markdown(f"**Per-axis effect size (Cohen's d, global coords): {pick} vs {ref}**")
        st.dataframe(es.head(6)[["axis_short", "mean_a", "mean_b", "cohens_d"]].round(3),
                     hide_index=True, use_container_width=True)

    # domain context (interpretation only; never changes coordinates)
    ctx = dctx.get_domain_context(domain, substrate)
    with st.expander("Domain context (interpretation only — does NOT alter coordinates)"):
        for c in ctx.caveats:
            st.markdown(f"- {c}")


def _render_shine(calib):
    from . import data_loader as dl
    st.caption("SHINE Day 0 / Day 2 projected from the **legacy cached autoresearch BSV "
               "remap** (NOT a recomputed raw-spectrum projection — SHINE has no per-sample "
               "mean-spectra file). Its upstream 3-axis collapse is preserved; do not read "
               "this as 11 independent measured axes.")
    sh, ph = dl.load_pilot_cohorts("shine_liver_injury")
    if ph or sh is None or sh.empty:
        st.warning("SHINE data unavailable (placeholder mode).")
        return
    system = st.radio("Coordinate system", ["Global GAIRA coordinates", "Raw (legacy remap)"],
                      key="coord_shine", horizontal=True)
    traces = []
    for _, r in sh.iterrows():
        raw = {a: float(r[a]) for a in cfg.BSV_AXES}
        vals = gc.global_display_dict(raw, calib) if system.startswith("Global") else raw
        traces.append({"name": r["cohort"], "values": vals})
    label = COORD_LABELS["global"] if system.startswith("Global") else "Raw legacy autoresearch remap"
    st.markdown(f"<span class='pill'>{label}</span>", unsafe_allow_html=True)
    st.plotly_chart(gp.radar_figure(traces[:6], title=label, height=460),
                    use_container_width=True, config={"displayModeBar": False})
    nz = [sum(1 for a in cfg.BSV_AXES if abs(float(r[a])) > 1e-4) for _, r in sh.iterrows()]
    st.caption(f"Raw nonzero axes per cohort: {nz} — upstream 3-axis projection limitation preserved.")


def _render_cross_dataset_map(df_ref, calib):
    st.caption("Biological samples in the FROZEN global coordinate space, shown via a "
               "2-D PCA **for display only**. The PCA axes are NOT the global biochemical "
               "coordinates. Reference/calibration anchors are shown separately.")
    from sklearn.decomposition import PCA
    G = cv.global_matrix(df_ref)
    p = PCA(n_components=2, random_state=0).fit(G[df_ref["role"] == "biological_range"])
    coords = p.transform(G)
    d = df_ref.copy()
    d["x"], d["y"] = coords[:, 0], coords[:, 1]
    color_by = st.radio("Color by", ["matrix", "dataset", "label"], horizontal=True, key="map_color")
    import plotly.graph_objects as go
    fig = go.Figure()
    for key, grp in d.groupby(color_by):
        is_anchor = grp["role"].iloc[0] == "calibration_behavior"
        fig.add_trace(go.Scatter(
            x=grp["x"], y=grp["y"], mode="markers", name=f"{key}" + (" (anchor)" if is_anchor else ""),
            marker=dict(size=7, symbol="diamond" if is_anchor else "circle",
                        opacity=0.55 if is_anchor else 0.8)))
    gp.apply_dark(fig, title="Global-coordinate PCA (display only — not the biochemical axes)", height=520)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    eta = cv.nuisance_eta_squared(df_ref, "dataset")["eta_squared"].mean()
    st.caption(f"Mean dataset-identity η² across global axes = {eta:.2f} "
               f"({'dataset identity is a strong separator' if eta>0.5 else 'moderate separator — not full cross-domain invariance'}). "
               "Disease clustering here does NOT validate biochemical meaning.")


def _render_nuisance(df_ref):
    st.caption("Diagnostic only — no batch correction is applied in V3. Association of "
               "global coordinates with nuisance variables (η²; higher = more confounded).")
    eta_ds = cv.nuisance_eta_squared(df_ref, "dataset")
    st.markdown("**Dataset identity η² on global coordinates**")
    st.dataframe(eta_ds.round(3), hide_index=True, use_container_width=True)
    bio = df_ref[df_ref.dataset.isin(["serum_liver", "ev_diabetes"])]
    eta_mx = cv.nuisance_eta_squared(bio, "matrix")
    st.markdown("**Matrix (serum vs EV) η²**")
    st.dataframe(eta_mx.round(3), hide_index=True, use_container_width=True)
    mean_eta = float(eta_ds["eta_squared"].mean())
    verdict = ("DATASET IDENTITY DOMINATES" if mean_eta > 0.5
               else "moderate — global space is a prototype, NOT full cross-domain invariance"
               if mean_eta > 0.2 else "weak")
    st.warning(f"Mean dataset η² = {mean_eta:.2f} → {verdict}. All calibration/biological "
               "spectra here are Ag-SERS; Raman-regime generalization is untested.")


# ─────────────────────────────────────────────────────────────────────
# Mode 4 — Coordinate Validation
# ─────────────────────────────────────────────────────────────────────

def render_validation():
    st.header("Mode 4 — Coordinate Validation")
    df_ref = cv.load_reference_samples()
    calib = gc.load_calibration()
    if df_ref is None or calib is None:
        st.error("GLOBAL COORDINATE UNAVAILABLE — validation artifacts missing.")
        return

    st.subheader("Cohort-invariance test")
    serum = [{a: float(r[f"raw_{a}"]) for a in cfg.BSV_AXES}
             for _, r in df_ref[df_ref.dataset == "serum_liver"].head(10).iterrows()]
    ev = [{a: float(r[f"raw_{a}"]) for a in cfg.BSV_AXES}
          for _, r in df_ref[df_ref.dataset == "ev_diabetes"].head(10).iterrows()]
    res = cv.invariance_check(serum[0], calib,
                              [[], serum[1:], ev, serum[1:5] + ev[:5]], atol=1e-9)
    c1, c2 = st.columns(2)
    c1.metric("Global coord max deviation across cohorts", f"{res['global_max_deviation']:.1e}")
    c2.metric("Cohort-relative max deviation", f"{res['cohort_relative_max_deviation']:.2f}")
    (st.success if res["global_invariant"] else st.error)(
        f"Global coordinates invariant to comparison cohort: {res['global_invariant']} "
        f"(≤1e-9). Cohort-relative coordinates change as expected: {res['cohort_relative_changes']}.")

    st.subheader("Axis variance: raw vs global (dominance)")
    va = cv.variance_before_after(df_ref)
    st.dataframe(va[["axis_short", "raw_var_rank", "global_var_rank",
                     "raw_dyn_range", "global_dyn_range"]].round(3),
                 hide_index=True, use_container_width=True)
    rd = cv.redox_dominance(df_ref)
    st.caption(f"Redox (G10): raw variance rank {rd['raw_variance_rank']} → global rank "
               f"{rd['global_variance_rank']}; global max|z|={rd['global_max_abs']:.1f}. "
               "Calibration removes raw-scale dominance while letting extreme redox exceed "
               "the biological reference range (comparability, not cosmetic equality).")

    st.subheader("Calibration manifest")
    import json
    mp = cfg.GENERATED_DIR / "global_coordinate_build_manifest_v1.json"
    if mp.exists():
        st.json(json.loads(mp.read_text()))
    render_axis_coverage()
    st.subheader("Known limitations")
    st.markdown(
        "- Global coordinates are a **deterministic prototype**, not a trained model or "
        "clinical measurement.\n"
        "- Fit population is 100% Ag-SERS (serum+EV); Raman generalization untested.\n"
        "- Three legacy split families (purine/lipid/redox) are not independently grounded.\n"
        "- Dataset identity remains a moderate separator (see nuisance diagnostics).")
