"""Page 5 — Serum Spike Stress Test. Where Ag-SERS recovery succeeds and where it
fails. Grounded in the committed Spike Validation outputs (phase7_serum_vs_pure,
serum-baseline / spiked-serum projections). Scientifically honest by design.
"""
from __future__ import annotations
import numpy as np
import streamlit as st

from .. import components as C, figures as F, serum as S, metrics as MET, biological as B
from ..engine_bridge import get_bridge, cache_key

TIER_LABELS = {"strong": "Strongly recoverable", "partial": "Partially / inconsistently recoverable",
               "poor": "Poorly recoverable / matrix-dominated"}


@st.cache_data(show_spinner=False)
def _summary():
    return S.recoverability_summary()


@st.cache_data(show_spinner=False)
def _reco():
    return S.load_recoverability()


@st.cache_data(show_spinner="Scoring serum recoverability…")
def _conf_df(_ck):
    return S.confidence_recoverability(get_bridge(), S.load_recoverability())


@st.cache_data(show_spinner=False)
def _heatmap(_ck, analytes):
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


def _paper_validation():
    st.markdown("### B+ · Validated against the source paper")
    st.markdown(
        '<div class="gaira-take"><b>GAIRA independently recovered the published finding.</b> '
        'The source Ag-SERS serum study '
        '(<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC12680727/">PMC12680727</a>) spiked '
        'serum with tyrosine (55 µM), lactose (100 µM), uric acid (280 µM), hypoxanthine (10 µM), '
        '<b>adenine (0.3–0.4 µM)</b>, CoA (100 µM) and more, and reported that — despite serum '
        'holding &gt;4000 metabolites — <b>only uric acid and hypoxanthine produce a meaningful '
        '(~2×) SERS signal</b>; tyrosine, lactose, adenine and even millimolar glucose gave none, '
        'because "concentration alone doesn\'t predict SERS visibility" (it is Ag adsorption '
        'affinity + competition + protein steric hindrance). GAIRA\'s tiers, computed only from '
        'the frozen Raman atlas, land in the same place: the strong tier is the oxopurines + '
        'thiones; adenine at 0.4 µM and the high-concentration weak adsorbers are poor. This is '
        'the physics of the measurement, reproduced from an independent representation.</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="gaira-caveat"><b>One honest discrepancy — urate.</b> The paper calls uric '
        'acid the <i>strongest</i> serum signal, yet GAIRA rates <b>urate "poor"</b> (direction '
        'agreement 0.07). Both are correct because they measure different things: urate was spiked '
        'at 280 µM into serum that is <i>already uric-acid-saturated</i> — urate IS the dominant '
        'serum background — so adding more barely changes the <i>direction</i> of the state, even '
        'though the <i>absolute</i> urate band is huge. GAIRA\'s metric is INCREMENTAL directional '
        'recoverability of a spike, not absolute detectability. Read urate\'s "poor" as "a urate '
        'spike is not separable from the urate background", not "urate is invisible".</div>',
        unsafe_allow_html=True)


def _donor_sera(b):
    """B++ — the paper's 81 real healthy-donor sera projected live through the frozen
    V6 engine. A single-group characterization (no disease contrast); the point is that
    GAIRA's BSV of real serum SERS is purine-dominated, exactly as the paper reports."""
    d = B.characterization_summary("gobbato_donor_sera")
    if d is None:
        return
    st.markdown("### B++ · Real donor sera confirm it — 81 spectra through the V6 engine")
    st.markdown(
        f'<div class="gaira-caption">The spike-in tiers above are pure-vs-serum <i>directions</i>. '
        f'Here are the paper\'s <b>{d["n"]} real healthy-donor sera</b> themselves — every one '
        f'projected live through the frozen atlas → BSV (committed sanitized artifact, atlas '
        f'<code>09ed804a…</code>). No spike, no disease label; just what real serum SERS looks like '
        f'in biochemical-state space.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.0, 1.05], gap="large")
    with c1:
        C.figure(F.multi_radar([{"name": "donor sera (mean)", "axes": d["axes"]}],
                               title="Mean BSV of 81 real donor sera"),
                 cap="Absolute biochemical-state composition, averaged over 81 donors. One "
                     "theme dominates.")
    with c2:
        st.markdown("**Mean composition across 81 donors** (± SD)")
        for t, m, sd in d["top_themes"]:
            st.markdown(f"- {F.THEME_SHORT.get(t, t)} — **{m:.3f}** (±{sd:.3f})")
        C.stat_row([
            (f"{d['purine_share']:.2f}", "purine/nucleobase share"),
            (f"{d['ood']:.2f}", "mean OOD (out-of-domain)"),
            (f"{d['conf']:.2f}", "mean confidence"),
            (f"{d['bg']:.2f}", "matrix background"),
        ])
    st.markdown(
        '<div class="gaira-take"><b>GAIRA reproduces the paper\'s central result on the real '
        'sera, not just the spikes.</b> Gobbato et al. show that human serum SERS variance is '
        'dominated by <b>uric acid + hypoxanthine</b> (their PC1 ≈ 70%). Run blind through the '
        'frozen Raman atlas, the mean BSV of all 81 donor sera is led by the '
        '<b>purine/nucleobase</b> theme — the same two oxopurines, from an entirely independent '
        'representation that never saw a SERS spectrum during fitting. The tight spread (SD ≈ '
        '0.01 on the leading theme) says this purine dominance is a property of serum-on-silver, '
        'shared across donors.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="gaira-caveat">This is <b>characterization, not diagnosis</b>: a single '
        'healthy-donor group, strongly out-of-domain (mean OOD '
        f'{d["ood"]:.2f}). The absolute BSV reflects the Ag-adsorbed serum subset (purines '
        'adsorb; most metabolites do not — see the tiers above), not whole-serum composition. '
        'Inter-individual differences here are dominated by the urate/hypoxanthine ratio, exactly '
        'as the paper describes.</div>', unsafe_allow_html=True)


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
    b = get_bridge()
    bsv = b.infer(ba["after_coord"], domain="serum").bsv
    st.markdown("**Recoverability terms** (separate, not one opaque number)")
    C.stat_row([
        (f"{row.cos_spike_vs_pureSERS:+.2f}", "direction agreement*"),
        (f"{row.spike_displacement_norm:.3f}", "detectability"),
        (f"{row.replicate_direction_cos:.2f}", "reproducibility"),
        (f"{ba['background']:.2f}", "matrix dominance"),
        (f"{row.spike_conc_uM:g} µM", "spike concentration"),
    ])
    st.markdown("**Interpretation confidence — separated (Part 5)**")
    C.stat_row([
        (f"{MET.atlas_support(bsv):.2f}", "atlas support (1−OOD)"),
        (f"{ba['ood']:.2f}", "OOD"),
        (f"{MET.overall_theme_specificity(b.eng.builder, bsv):.2f}", "theme specificity"),
        (f"{MET.replicate_reliability(row.replicate_direction_cos):.2f}", "replicate reliability"),
        (f"{row.cos_spike_vs_pureSERS:+.2f}", "matrix recoverability*"),
    ])
    C.caption("*direction agreement / matrix recoverability = cos(serum-spike, pure-SERS "
              "fingerprint), the validated primary metric that defines the tier. Atlas support and "
              "theme specificity measure ATLAS fit, not surface observability — they are shown "
              "separately, never merged into one 'confidence'.")
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
    mat, themes = _heatmap(cache_key(), tuple(ex))
    C.figure(F.recoverability_heatmap(ex, mat, themes,
                                      title="ΔBSV (spiked − baseline) · strong (top) → failing (bottom)"),
             cap="Signed theme change per analyte. Strong adsorbers (top) show coherent, "
                 "analyte-appropriate moves; failing analytes (bottom) show diffuse, "
                 "matrix-driven change.")
    st.markdown("#### The current confidence limitation")
    C.figure(F.confidence_limitation(_conf_df(cache_key())),
             cap="Engine confidence vs identity recovery across all 53 analytes.",
             interp="Confidence is essentially FLAT across the recoverability range (strong-tier "
                     "mean ≈ poor-tier mean). Confidence tracks domain support and spectrum "
                     "quality — it does NOT yet distinguish strong from weak Ag adsorbers within "
                     "serum.",
             limits="A matrix-recoverability prior for confidence is a recommended, NOT-yet-"
                     "implemented upgrade (see BSV Validation).")


def _pure_vs_serum_audit(b):
    st.markdown("### C+ · Pure-vs-serum audit — adenine & ergothioneine")
    st.markdown('<div class="gaira-caption">Pure-analyte dose response and serum-spike are '
                '<b>different regimes</b>. This audit checks the data mapping and cross-compares '
                'them for the two calibrants.</div>', unsafe_allow_html=True)
    cols = st.columns(2, gap="large")
    for col, analyte in zip(cols, ["adenine", "ergothioneine"]):
        pv = S.pure_vs_serum(b, analyte)
        with col:
            st.markdown(f"**{analyte.capitalize()}** → "
                        f"{F.THEME_SHORT.get(pv['target_theme'], pv['target_theme'])}")
            st.markdown(
                f"- PURE: {pv['pure_target_lo']:.3f} → {pv['pure_target_hi']:.3f} over "
                f"{pv['pure_conc_range'][0]:g}–{pv['pure_conc_range'][1]:g} µM (slope "
                f"{pv['pure_slope']:+.4f}) — **strong**\n"
                f"- SERUM: spiked at **{pv['serum_conc_uM']:g} µM**, direction agreement "
                f"{pv['serum_direction_agreement']:+.2f} → tier **{pv['serum_tier']}**")
    st.markdown('<div class="gaira-caveat"><b>Adenine audit result.</b> Adenine is strong in pure '
                'but poor in serum — and its serum spike was only <b>0.4 µM</b>, 12–25× lower than '
                'the recoverable analytes (ergothioneine 5 µM, hypoxanthine 10 µM, xanthine 50 µM). '
                'The mapping is correct (adenine is not confused with hypoxanthine/urate). The poor '
                'serum result is <b>consistent with the low spike concentration plus surface '
                'competition / matrix masking</b> — not proof of poor adsorption. Contrast '
                'phenylalanine (spiked at 78 µM yet still poor): a genuine adsorption/matrix '
                'failure, a different mode.</div>', unsafe_allow_html=True)


def _terms_and_ablation(b):
    st.markdown("### C++ · Recoverability terms & ablation")
    df = _reco()
    terms = S.recoverability_terms(b, df)
    st.markdown('<div class="gaira-caption">The tier is defined by the <b>validated primary</b> '
                'metric (direction agreement = cos vs pure-SERS). The other terms are shown '
                'separately, never merged with invented weights.</div>', unsafe_allow_html=True)
    show = terms.sort_values("direction_agreement", ascending=False)
    st.dataframe({"analyte": show.analyte, "tier": show.tier, "conc_µM": show.spike_conc_uM,
                  "direction_agreement*": show.direction_agreement.round(3),
                  "detectability": show.detectability.round(3),
                  "reproducibility": show.reproducibility.round(3),
                  "matrix_dominance": show.matrix_dominance.round(3)},
                 use_container_width=True, hide_index=True, height=280)
    abl = S.ablation_table(terms)
    st.markdown("**Ablation — top-5 'recoverable' set by each single term:**")
    for k, v in abl.items():
        st.markdown(f"- by **{k}**: {', '.join(v)}")
    st.markdown('<div class="gaira-caption">Ranking by <i>reproducibility</i> alone would rank '
                'phenylalanine highly — it moves consistently, but in the WRONG direction. This is '
                'why direction agreement (not detectability or reproducibility) is the meaningful '
                'recoverability criterion.</div>', unsafe_allow_html=True)


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
    _paper_validation()
    st.markdown("<hr/>", unsafe_allow_html=True)
    _donor_sera(bridge)
    st.markdown("<hr/>", unsafe_allow_html=True)
    _tiers(bridge)
    st.markdown("<hr/>", unsafe_allow_html=True)
    _pure_vs_serum_audit(bridge)
    st.markdown("<hr/>", unsafe_allow_html=True)
    _terms_and_ablation(bridge)
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
