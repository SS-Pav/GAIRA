"""Page 4 — Calibration. The strongest page: controlled perturbations validate the
frozen coordinate system.

Layout: tabs (Adenine · Ergothioneine · Uricase depletion · Compare the three). Each
addition study runs the same reasoning workflow, with the signature reasoning-cascade
driven by a concentration slider. Uricase is a depletion (before/after/difference).
Throughout, the MSS layer is given more prominence than the radar: the radar
summarizes, the MSS explains, the components are the evidence.
"""
from __future__ import annotations
import numpy as np
import streamlit as st

from .. import components as C, figures as F, data as D, calibration as CAL
from ..engine_bridge import get_bridge

MOTIF_NAMES = None  # filled lazily


# ── cached heavy prep (numpy-only outputs; safe for st.cache_data) ──
@st.cache_data(show_spinner="Projecting the dose ladder…")
def _prep_dose(cal_key, method):
    b = get_bridge()
    cal = D.calibration(cal_key)
    s = CAL.build_dose_series(cal, method=method)
    ids = [m.id for m in b.mss.biochemical()]
    evo = CAL.motif_evolution(b, s, ids)
    mean, rl, rs = CAL.theme_series(b, s, cal.target_theme)
    fit = CAL.langmuir_fit(rl, rs)
    rho = CAL.spearman(rl, rs)
    comp = CAL.component_series(b, s)
    vecs = CAL.bsv_theme_vectors(b, s.mean_coord)
    proj, var = CAL.trajectory_2d(vecs)
    return dict(levels=s.levels, mean_coord=s.mean_coord, evo=evo, motif_ids=ids,
                theme_mean=mean, rep_levels=rl, rep_scores=rs, fit=fit, rho=rho,
                comp=comp, proj=proj, var=var, condition=s.condition)


@st.cache_data(show_spinner=False)
def _prep_uricase():
    b = get_bridge()
    cond = CAL.uricase_conditions(b)
    before, after = cond["spiked"], cond["spiked+uricase"]
    ids = [m.id for m in b.mss.biochemical()]
    mss_b = {a.id: a.composition for a in b.bsv_and_mss(before)[1]}
    mss_a = {a.id: a.composition for a in b.bsv_and_mss(after)[1]}
    ob, oa = b.infer(before).bsv, b.infer(after).bsv
    themes = b.bio_themes
    return dict(before=before, after=after, motif_ids=ids,
                mss_before=[mss_b[i] for i in ids], mss_after=[mss_a[i] for i in ids],
                theme_before=[ob.composition[t] for t in themes],
                theme_after=[oa.composition[t] for t in themes], themes=themes,
                radar_before=b.infer(before).radar["axes"], radar_after=b.infer(after).radar["axes"],
                purine_before=ob.composition["nucleic_purine"],
                purine_after=oa.composition["nucleic_purine"])


def _names(b):
    return {m.id: m.name for m in b.mss.motifs}


# ── the reusable addition-study workflow (adenine / ergothioneine) ──
def _addition_study(b, cal, method, mechanism_note):
    P = _prep_dose(cal.key, method)
    levels = P["levels"]; names = _names(b)

    st.markdown("##### 1 · Experimental setup")
    C.figure(F.experimental_schematic(cal.analyte, method.split("@")[0] if method else "Ag",
                                      method.split("@")[1] if method and "@" in method else "785"),
             cap=f"{cal.analyte.capitalize()} adsorbs to the colloid, is probed by SERS, and the "
                 f"spectrum is projected into the frozen atlas. Condition: {P['condition']}.")

    st.markdown("##### 2 · The reasoning cascade — drag the concentration slider")
    st.markdown('<div class="gaira-caption">Every panel updates together: the reconstructed '
                'spectrum, the latent components, the MSS motifs, the BSV and the radar. '
                'This is the whole engine, live, at one concentration.</div>',
                unsafe_allow_html=True)
    dose = st.select_slider("Concentration (µM)", options=[float(x) for x in levels],
                            value=float(levels[-1]), key=f"slider_{cal.key}")
    idx = int(np.argmin(np.abs(levels - dose)))
    C.figure(F.reasoning_cascade(b, P["mean_coord"][idx], dose_label=f"{levels[idx]:.2f} µM"),
             cap="The GAIRA reasoning cascade at the selected concentration. Spectrum → latent "
                 "components → MSS motifs → BSV → radar, computed live by the frozen engine.",
             interp="As you raise the dose, watch the target motif strengthen and the radar move "
                     "toward the expected biochemical system.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("##### 3 · Representative spectra")
        lo, md, hi = 0, len(levels) // 2, len(levels) - 1
        specs = [b.reconstruct(P["mean_coord"][i])[1] for i in (lo, md, hi)]
        grid = b.reconstruct(P["mean_coord"][0])[0]
        C.figure(F.representative_spectra(
            specs, grid, [f"{levels[lo]:.2f} µM", f"{levels[md]:.2f} µM", f"{levels[hi]:.2f} µM"],
            bands=b.motif_by_id(cal.target_motif).bands_cm),
            cap="Atlas reconstructions at low / medium / high dose. Dashed lines: the target "
                "motif's bands.")
    with c2:
        st.markdown("##### 4 · Latent components")
        top_js = list(int(x) for x in np.argsort(-(P["comp"].max(0) - P["comp"].min(0)))[:6])
        C.figure(F.component_evolution(levels, P["comp"], top_js,
                                       title=f"{cal.analyte.capitalize()} — component evolution"),
                 cap=mechanism_note)

    st.markdown("##### 5 · Molecular Spectral Signatures  ·  the central panel")
    C.figure(F.mss_evolution(levels, P["evo"], cal.target_motif, names,
                             title=f"{cal.analyte.capitalize()} — MSS motif evolution with dose"),
             cap="Each biochemical motif's elevation vs dose; the target motif (★) is emphasised.",
             interp="Component changes propagate into motif changes here — the layer where the "
                     "spectroscopy becomes chemistry.")

    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.markdown("##### 6 · Dose-response + Langmuir fit")
        C.figure(F.dose_response_langmuir(levels, P["theme_mean"], P["rep_levels"], P["rep_scores"],
                                          P["fit"], cal.analyte, cal.target_theme, P["rho"]),
                 cap="Target-theme evidence share vs concentration: replicate cloud, per-dose means, "
                     "and a saturating Langmuir overlay.",
                 interp="Monotonic and saturating — the BSV validation found this for every "
                         "calibration relation (all permutation p = 0.002).")
    with c4:
        st.markdown("##### 7 · Dose trajectory in BSV space")
        C.figure(F.trajectory_2d(P["proj"], levels, P["var"],
                                 title=f"{cal.analyte.capitalize()} — dose trajectory"),
                 cap=f"BSV theme-vectors across dose, PCA to 2-D (visualisation only). PC1 = "
                     f"{P['var'][0]:.0%} of the movement.",
                 limits="High PC1 fraction means the perturbation moves the BSV along essentially "
                         "one direction — consistent with the low effective dimensionality found "
                         "in BSV validation.")

    st.markdown("##### 8 · Biochemical State Vector — low vs high dose")
    rc1, rc2 = st.columns([1.1, 1.0], gap="large")
    with rc1:
        radar_lo = b.infer(P["mean_coord"][0]).radar["axes"]
        radar_hi = b.infer(P["mean_coord"][-1]).radar["axes"]
        C.figure(F.radar(radar_hi, title=f"{cal.analyte.capitalize()} radar", ref_axes=radar_lo),
                 cap="Solid = highest dose; dashed = zero dose. The radar summarises the same "
                     "movement the MSS panel explains.")
    with rc2:
        _evidence_panel(b, cal, P)


def _evidence_panel(b, cal, P):
    st.markdown("##### 9 · Evidence panel")
    motif = b.motif_by_id(cal.target_motif)
    st.markdown(f"**Target theme** · {F.THEME_SHORT.get(cal.target_theme, cal.target_theme)} "
                f"(ρ={P['rho']:.2f} vs dose)")
    st.markdown(f"**Target motif** · {motif.name} (confidence {motif.confidence:.2f})")
    st.markdown("**Supporting components** · "
                + ", ".join(f"c{c['component']} ({c['weight']:.2f})" for c in motif.contributors[:5]))
    if motif.reference_analytes:
        st.markdown("**Reference chemistries** · " + ", ".join(motif.reference_analytes[:8]))
    pert = motif.perturbation
    st.markdown(f"**Perturbation evidence** · {len(pert['dose_responsive_components'])} "
                f"dose-responsive components, {len(pert['serum_spike_matches'])} serum-spike matches"
                + (f", {len(pert['depletion_matches'])} depletion matches"
                   if pert["depletion_matches"] else ""))
    if P["fit"] is not None:
        st.markdown(f"**Langmuir** · R²={P['fit'][3]:.2f}, K={P['fit'][4]:.2g} µM")


def _adenine(b):
    st.markdown('<div class="gaira-card"><b>Adenine — component redistribution.</b> A strong Ag '
                'adsorber. The purine theme rises monotonically, but the underlying components '
                '<b>redistribute</b>: different latent motifs dominate at different concentrations '
                '(c3/c13 rise while others fall). This is not simple scaling.</div>',
                unsafe_allow_html=True)
    st.write("")
    _addition_study(b, D.calibration("adenine"), CAL.ADENINE_METHOD,
                    "Colour = direction. c3/c13 (adenine/purine components) RISE while others FALL "
                    "— the signature of redistribution rather than single-component scaling.")
    C.caveats([
        "This validates the biochemical reasoning layer under conditions where adenine is "
        "effectively recovered by Ag-SERS (cAg@785). It does not imply universal SERS modality "
        "invariance.",
        "The sterol motif tracks purine here because component c3 carries both adenine and estrone "
        "reference loadings — a real spectral collision, surfaced not hidden.",
    ])


def _ergothioneine(b):
    st.markdown('<div class="gaira-card"><b>Ergothioneine — single-motif scaling.</b> The cleanest '
                'calibrant: a thione that chemisorbs to silver. In contrast to adenine, the sulfur '
                'motif largely <b>scales one dominant latent pattern</b> monotonically, giving a '
                'near-textbook Langmuir dose-response.</div>', unsafe_allow_html=True)
    st.write("")
    _addition_study(b, D.calibration("ergothioneine"), None,
                    "Contrast with adenine: the component evolution is dominated by scaling rather "
                    "than crossover — fewer components change direction.")
    C.caveats([
        "This validates the reasoning layer for a strong, well-behaved Ag chemisorber. Weak "
        "adsorbers behave very differently (see the Serum Spike Stress Test).",
        "Buffer matrix, single substrate (cAg@785) — the most favourable possible condition.",
    ])


def _uricase(b):
    U = _prep_uricase(); names = _names(b)
    st.markdown('<div class="gaira-card"><b>Uricase depletion — biochemical subtraction.</b> '
                'Uricase enzymatically removes urate from spiked serum. This is a knock-OUT, not an '
                'addition: the read-out is what <i>decreases</i>. The difference isolates '
                'purine-specific evidence.</div>', unsafe_allow_html=True)
    st.write("")

    st.markdown("##### Experimental setup")
    C.figure(F.experimental_schematic("urate-spiked serum", "Ag", "785"),
             cap="Urate-spiked serum, then + uricase enzyme (removes urate). Before = spiked, "
                 "after = spiked + uricase. Matrix: serum (out-of-domain).")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("##### Difference spectrum")
        gb, sb = b.reconstruct(U["before"]); _, sa = b.reconstruct(U["after"])
        C.figure(F.difference_spectrum(gb, sb, sa,
                 bands=b.motif_by_id("oxopurine_carbonyl").bands_cm),
                 cap="Atlas reconstruction, after − before. Dashed: oxopurine bands. Blue = lost "
                     "on depletion.")
    with c2:
        st.markdown("##### Difference radar")
        C.figure(F.radar(U["radar_after"], title="Uricase: after vs before",
                         ref_axes=U["radar_before"]),
                 cap="Solid = after uricase; dashed = before. The purine-associated axis contracts.")

    st.markdown("##### Difference MSS  ·  the key result")
    C.figure(F.difference_bars([names[i] for i in U["motif_ids"]], U["mss_before"], U["mss_after"],
                               title="Uricase — MSS motif difference (after − before)"),
             cap="Signed change per motif on urate removal.",
             interp="The oxopurine-carbonyl motif (urate/xanthine subgroup) falls most, while the "
                     "adenine-type purine-ring motif barely moves. The MSS layer RESOLVES the "
                     "specific depletion that the coarse purine theme (Δ≈"
                     f"{U['purine_after'] - U['purine_before']:+.3f}) almost entirely hides.")

    st.markdown("##### Difference BSV (theme waterfall)")
    C.figure(F.difference_bars([F.THEME_SHORT.get(t, t) for t in U["themes"]],
                               U["theme_before"], U["theme_after"],
                               title="Uricase — biochemical theme difference"),
             cap="Theme-level composition change. Composition is a competing share, so removing "
                 "urate lifts other themes' relative shares.")

    C.takeaways([
        "Enzymatic urate removal DECREASES the oxopurine motif specifically — the correct "
        "direction and the correct chemistry.",
        "The MSS layer localises the change to oxopurines; the theme-level purine signal barely "
        "moves, showing why motif resolution matters.",
    ])
    C.caveats([
        "Serum on Ag colloid is out-of-domain for a Raman atlas; the absolute effect is small and "
        "OOD is high. This validates <i>direction and specificity</i>, not absolute quantitation.",
        "Individual components can move either way on depletion; the motif- and theme-level "
        "aggregates are the interpretable read-out.",
    ])


def _compare(b):
    st.markdown('<div class="gaira-card"><b>Three perturbation classes, one coordinate system.</b> '
                'Addition that redistributes (adenine), addition that scales (ergothioneine), and '
                'depletion (uricase) trace distinct, characteristic paths through the same frozen '
                'BSV space. This is exactly the structure DART will make dynamic.</div>',
                unsafe_allow_html=True)
    st.write("")
    J = CAL.joint_trajectories(b)
    trajs = [
        {"name": "adenine · redistribution", "proj": J["adenine"]["proj"], "color": F.T.PRIMARY,
         "marker": "o"},
        {"name": "ergothioneine · scaling", "proj": J["ergothioneine"]["proj"], "color": F.T.GOOD,
         "marker": "s"},
        {"name": "uricase · depletion", "proj": J["uricase"]["proj"], "color": F.T.UP, "marker": "^"},
    ]
    cc1, cc2 = st.columns([1.15, 1.0], gap="large")
    with cc1:
        C.figure(F.compare_trajectories(trajs),
                 cap="All three studies projected into one shared BSV PCA space (visualisation "
                     "only). Open circles mark each trajectory's start.",
                 interp="Different biochemical operations produce different trajectory shapes in "
                         "the same fixed coordinate system — the foundation for dynamic (DART) "
                         "interpretation.")
    with cc2:
        st.markdown("**Trajectory classes**")
        st.markdown(
            "- **Scaling** (ergothioneine): one motif grows; the state moves along a straight ray.\n"
            "- **Redistribution** (adenine): the dominant components change with dose; the path "
            "curves as evidence shifts between motifs.\n"
            "- **Depletion** (uricase): the state retreats along the purine/oxopurine direction.\n\n"
            "These are the static analogues of the dynamic trajectory classes on the **Future "
            "DART** page (scaling, redistribution, loops, hysteresis, thresholds).")


def render(bridge):
    s = bridge.platform_stats()
    C.page_header(
        "Validation · controlled perturbations",
        "Calibration: predictable movement through the coordinate system",
        "These three datasets are controlled biochemical perturbations used to validate the frozen "
        "GAIRA coordinate system. If the architecture is sound, a known change must move the right "
        "spectral motif in the right direction, monotonically — and it does. This is the evidence "
        "that GAIRA reasons, rather than classifies.")
    C.question("When we deliberately add or remove a known analyte, does the corresponding MSS "
               "motif and biochemical theme respond monotonically, specifically, and reproducibly?")

    t_ade, t_erg, t_uri, t_cmp = st.tabs(
        ["Adenine", "Ergothioneine", "Uricase depletion", "Compare the three"])
    with t_ade:
        _adenine(bridge)
    with t_erg:
        _ergothioneine(bridge)
    with t_uri:
        _uricase(bridge)
    with t_cmp:
        _compare(bridge)

    C.provenance_footer(s)
