"""Page 5 — Serum Spike Stress Test. Where Ag-SERS recovery succeeds and where it
fails. Grounded in the committed Spike Validation outputs (phase7_serum_vs_pure,
serum-baseline / spiked-serum projections). Scientifically honest by design.
"""
from __future__ import annotations
import numpy as np
import streamlit as st

from .. import components as C, figures as F, serum as S
from ..engine_bridge import get_bridge

TIER_LABELS = {"strong": "Strongly recoverable", "partial": "Partially / inconsistently recoverable",
               "poor": "Poorly recoverable / matrix-dominated"}


@st.cache_data(show_spinner=False)
def _summary():
    return S.recoverability_summary()


@st.cache_data(show_spinner=False)
def _reco():
    return S.load_recoverability()


@st.cache_data(show_spinner="Scoring serum recoverability…")
def _conf_df():
    return S.confidence_recoverability(get_bridge(), S.load_recoverability())


@st.cache_data(show_spinner=False)
def _heatmap(analytes):
    mat, themes = S.theme_delta_matrix(get_bridge(), list(analytes))
    return mat, themes


# ── Section A ──
def _explainer():
    st.markdown("### A · Why recovery is not guaranteed")
    C.figure(F.recoverability_cascade(),
             cap="What is present is not what is observed. Each ≠ is a place an analyte can be lost.")
    c1, c2 = st.columns([1.0, 1.0], gap="large")
    with c1:
        st.markdown("**The observation, step by step**")
        st.markdown(
            "Serum baseline + one spiked analyte + Ag colloid → competitive adsorption → SERS "
            "spectrum → GAIRA inference. Between *present* and *observed* sit:")
        st.markdown(
            "- analyte abundance · adsorption affinity · surface competition\n"
            "- molecular orientation · hotspot access · aggregation state\n"
            "- matrix background · acquisition variability")
    with c2:
        st.markdown('<div class="gaira-caveat"><b>Failure to recover an analyte is NOT evidence '
                    'of biochemical absence.</b> A poorly-adsorbing molecule can be abundant in '
                    'serum yet invisible on silver. This page maps that boundary honestly — it is '
                    'a property of the measurement, not a bug in the engine.</div>',
                    unsafe_allow_html=True)


# ── Section B ──
def _dataset_summary():
    s = _summary()
    st.markdown("### B · Dataset summary")
    C.stat_row([
        (f"{s['n_analytes']}", "analytes spiked"),
        (f"{s['n_strong']}", "strongly recoverable"),
        (f"{s['n_partial']}", "partial"),
        (f"{s['n_poor']}", "poor / matrix-dominated"),
        (f"{s['n_above_null_p05']}", "above null (p<0.05)"),
    ])
    st.markdown(
        f'<div class="gaira-caption">All from the committed Spike Validation output '
        f'<code>phase7_serum_vs_pure.csv</code> (53 analytes, Ag colloid, 785 nm, 5 replicates '
        f'each; serum baseline = 15 unspiked spectra). Recoverability tier = '
        f'cos(serum-spike direction, that analyte\'s pure-SERS fingerprint): '
        f'strong ≥ {S.STRONG}, partial ≥ {S.PARTIAL}, else poor. Median spike-vs-pure angle is '
        f'{s["median_angle_deg"]:.0f}° (≈orthogonal): most analytes do NOT reproduce their own '
        f'signature in serum. Only {s["n_above_null_p05"]}/{s["n_analytes"]} exceed the '
        f'permutation null.</div>', unsafe_allow_html=True)
    st.markdown('<div class="gaira-caveat">Historical notes sometimes cite different analyte '
                'counts; this page uses the file-derived count (53) from the canonical validation '
                'table, not a remembered number.</div>', unsafe_allow_html=True)


# ── Section C ──
def _tiers(b):
    st.markdown("### C · Recoverability tiers")
    df = _reco()
    C.figure(F.recoverability_scatter(df, annotate=["hypoxanthine", "xanthine", "ergothioneine",
                                                    "phenylalanine", "adenine", "lactate"]),
             cap="All 53 analytes. x = identity recovery (spike direction vs the analyte's own "
                 "pure-SERS fingerprint); y = replicate consistency; size = displacement.",
             interp="The strongly recoverable set is dominated by strong Ag adsorbers — the "
                     "oxopurines (hypoxanthine, xanthine, guanine) and ergothioneine. Note "
                     "phenylalanine: highly consistent yet near-zero identity recovery — it moves "
                     "reproducibly toward the matrix, not toward itself.")

    tier = st.radio("Tier", ["strong", "partial", "poor"], horizontal=True,
                    format_func=lambda t: TIER_LABELS[t])
    opts = list(df[df.tier == tier].analyte)
    default = {"strong": "hypoxanthine", "partial": "acetoacetate", "poor": "phenylalanine"}.get(tier)
    analyte = st.selectbox("Analyte", opts, index=opts.index(default) if default in opts else 0)
    _analyte_detail(b, analyte, df)


def _analyte_detail(b, analyte, df):
    ba = S.before_after(b, analyte)
    if ba is None:
        C.caveats([f"No spiked-serum spectra for {analyte} in the committed projection."]); return
    row = df[df.analyte == analyte].iloc[0]
    C.stat_row([
        (f"{row.cos_spike_vs_pureSERS:+.2f}", "identity recovery (cos)"),
        (f"{row.replicate_direction_cos:.2f}", "replicate consistency"),
        (f"{ba['ood']:.2f}", "OOD score"),
        (f"{ba['confidence']:.2f}", "confidence"),
        (f"{ba['background']:.2f}", "matrix share"),
    ])
    c1, c2 = st.columns(2, gap="large")
    with c1:
        gb, sb = b.reconstruct(ba["before_coord"]); _, sa = b.reconstruct(ba["after_coord"])
        C.figure(F.difference_spectrum(gb, sb, sa, title=f"{analyte}: spiked − baseline (serum)"),
                 cap="Atlas reconstruction of the difference. Red = gained on spiking.")
        C.figure(F.radar(ba["radar_after"], title=f"{analyte} BSV", ref_axes=ba["radar_before"]),
                 cap="Solid = spiked serum; dashed = serum baseline.")
    with c2:
        themes = b.bio_themes
        C.figure(F.difference_bars([F.THEME_SHORT.get(t, t) for t in themes],
                                   [ba["bsv_before"].composition[t] for t in themes],
                                   [ba["bsv_after"].composition[t] for t in themes],
                                   title=f"{analyte} — ΔBSV themes"),
                 cap="Signed theme change on spiking (serum domain).")
        ids = [m.id for m in b.mss.biochemical()]
        names = {m.id: m.name for m in b.mss.motifs}
        C.figure(F.difference_bars([names[i] for i in ids],
                                   [ba["mss_before"][i] for i in ids],
                                   [ba["mss_after"][i] for i in ids],
                                   title=f"{analyte} — ΔMSS motifs"),
                 cap="Which spectral motifs move — the mechanistic read-out.")
    tier = row.tier
    msg = {
        "strong": f"{analyte} is a strong Ag adsorber: the serum spike moves in the direction of "
                  f"its own pure-SERS fingerprint (cos {row.cos_spike_vs_pureSERS:.2f}). The expected "
                  f"motif/theme move in the correct direction — recovery succeeds.",
        "partial": f"{analyte} is partially recovered: some directional agreement with its pure "
                   f"fingerprint (cos {row.cos_spike_vs_pureSERS:.2f}) but weaker/less consistent. "
                   f"Interpret with caution.",
        "poor": f"{analyte} is matrix-dominated: the spike moves nearly orthogonally to its own "
                f"fingerprint (cos {row.cos_spike_vs_pureSERS:.2f}). The serum/Ag background "
                f"overwhelms the analyte-specific signal — recovery fails at the MEASUREMENT "
                f"layer. This is not evidence the analyte is absent.",
    }[tier]
    (C.takeaways if tier == "strong" else C.caveats)([msg])


# ── Section D ──
def _success_vs_failure(b):
    st.markdown("### D · Success vs failure — and the confidence limitation")
    ex = ["hypoxanthine", "xanthine", "guanine", "ergothioneine", "ascorbate",
          "acetoacetate", "adenine", "phenylalanine", "lactate", "glucose"]
    mat, themes = _heatmap(tuple(ex))
    C.figure(F.recoverability_heatmap(ex, mat, themes,
                                      title="ΔBSV (spiked − baseline) · strong (top) → failing (bottom)"),
             cap="Signed theme change per analyte. Strong adsorbers (top) show coherent, "
                 "analyte-appropriate moves; failing analytes (bottom) show diffuse, "
                 "matrix-driven change.")
    st.markdown("#### The current confidence limitation")
    C.figure(F.confidence_limitation(_conf_df()),
             cap="Engine confidence vs identity recovery across all 53 analytes.",
             interp="Confidence is essentially FLAT across the recoverability range (strong-tier "
                     "mean ≈ poor-tier mean). Confidence tracks domain support and spectrum "
                     "quality — it does NOT yet distinguish strong from weak Ag adsorbers within "
                     "serum.",
             limits="A matrix-recoverability prior for confidence is a recommended, NOT-yet-"
                     "implemented upgrade (see BSV Validation).")


def render(bridge):
    s = bridge.platform_stats()
    C.page_header(
        "Stress test · SERS recoverability",
        "Where Ag-SERS succeeds — and where it cannot",
        "Bulk concentration is not surface occupancy, and surface occupancy is not SERS signal. "
        "This page defines the current boundary of GAIRA's SERS interpretation honestly: strong "
        "Ag adsorbers are recovered, weak ones are not, and the engine flags the difference "
        "through OOD — though, as shown below, not yet through confidence.")
    C.question("When does a known analyte perturbation remain recoverable after serum competition "
               "and Ag adsorption, and when does it fail?")

    _explainer()
    st.markdown("<hr/>", unsafe_allow_html=True)
    _dataset_summary()
    st.markdown("<hr/>", unsafe_allow_html=True)
    _tiers(bridge)
    st.markdown("<hr/>", unsafe_allow_html=True)
    _success_vs_failure(bridge)
    st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown("### E · Scientific conclusion")
    C.takeaways([
        "The serum-spike stress test defines the current boundary of GAIRA's SERS interpretation.",
        "Raman-derived biochemical motifs are useful when surface recovery is adequate (strong "
        "adsorbers — oxopurines, thiones).",
        "Poor adsorption or matrix competition can prevent analyte-specific recovery even when the "
        "analyte is present — a measurement limit, not an engine error.",
        "Future Au-SERS and DART datasets will be used to learn observation-specific "
        "recoverability; no such model is claimed today.",
    ])
    C.caveats([
        "Serum on Ag colloid is out-of-domain for a Raman atlas; high OOD here is correct.",
        "Confidence does not yet distinguish strong from weak Ag adsorbers within serum "
        "(Section D).",
    ])
    C.related(["3 · How GAIRA Reasons", "4 · Calibration", "6 · Biological Studies"])
    C.provenance_footer(s)
