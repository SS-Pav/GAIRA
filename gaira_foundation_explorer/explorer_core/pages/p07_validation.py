"""Page 7 — Validation (six datasets, projected through the frozen atlas)."""
from __future__ import annotations
import streamlit as st

from .. import data as D, ui, charts as C, theme as T


def render():
    V = D.validation()
    ui.page_header("The scientific proof", "Validation",
                   "Six datasets, none used to fit the atlas, projected through the frozen "
                   "coordinate system. The model must represent Raman faithfully, flag SERS as "
                   "out-of-domain, dose-respond to the right chemistry, and fail honestly where "
                   "surface physics forbids recovery.")
    ui.question("Does the frozen atlas behave correctly on data it has never seen?")

    tabs = st.tabs(["1 · Pure Raman", "2 · Gobbato SERS", "3 · Adenine", "4 · Ergothioneine",
                    "5 · Serum spike", "6 · Uricase"])

    # ── Tab 1 — pure Raman (in-domain) ──
    with tabs[0]:
        r = V.get("1_gobbato_raman", {})
        ui.question("Does the atlas faithfully represent Raman chemistry it was built on?")
        ui.stat_row([(r.get("n_spectra"), "spectra"), (r.get("n_analytes"), "analytes"),
                     (ui.fmt(r.get("mean_ood"), 3), "mean OOD"),
                     (ui.fmt(r.get("median_ood"), 3), "median OOD")])
        st.markdown(
            "Projecting the pure Gobbato Raman analytes (in-domain) gives a **mean out-of-domain "
            f"score of {ui.fmt(r.get('mean_ood'),3)}** — essentially in-distribution — with "
            "chemically sensible dominant themes (adenine→purine, glucose→saccharide, "
            "albumin→protein).")
        ui.note("info", "This sets the OOD baseline (~0.05) against which every out-of-domain set "
                        "below is measured. Passing is necessary, not sufficient.")
        ex = r.get("example_dominant_themes", {})
        if ex:
            st.markdown("**Example dominant themes:** " +
                        ", ".join(f"{a}→{t.replace('nucleic_','').replace('_',' ')}"
                                  for a, t in list(ex.items())[:10]))
        ui.takehome("The atlas is self-consistent: it represents its own reference chemistry with "
                    "near-zero OOD and correct themes.")

    # ── Tab 2 — SERS transfer ──
    with tabs[1]:
        r = V.get("2_gobbato_sers_transfer", {})
        ui.question("How much of a molecule's Raman signature survives the trip onto silver?")
        ui.stat_row([
            (r.get("n_matched_analytes"), "matched analytes"),
            (ui.fmt(r.get("sers_mean_ood"), 3), "SERS OOD"),
            (ui.fmt(r.get("raman_mean_ood"), 3), "Raman OOD"),
            (ui.fmt(r.get("median_coord_cosine"), 2), "median transfer cosine"),
            (f"{r.get('n_theme_preserved')}/{r.get('n_matched_analytes')}", "theme preserved"),
        ])
        pairs = D.transfer_pairs()
        if len(pairs):
            st.plotly_chart(C.transfer_bar(pairs), width="stretch",
                            config={"displayModeBar": False})
            st.markdown('<div class="small">Per-analyte cosine between the pure-Raman and pure-Ag-'
                        'SERS coordinates. Green = the dominant biochemical theme survives the '
                        'transfer; slate = it lands on a different theme. Hover for both themes.</div>',
                        unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            ui.card("What transferred",
                    "Strong, rigid Ag adsorbers — the **oxopurines** hypoxanthine (0.84), xanthine "
                    "(0.81) — plus albumin and phosphatidylinositol. Their ring/backbone modes "
                    "dominate both spectra.")
        with b:
            ui.card("What failed — and why",
                    "Weak adsorbers — **glucose (0.20), uracil (0.055)**, small amino acids. On "
                    "silver their spectrum is reshaped by adsorption affinity, orientation and "
                    "surface selection rules. This is *measurement physics*, not a modelling error.")
        ui.note("take",
                "SERS is flagged out-of-domain (OOD 3.5× Raman) and the model recovers **nothing "
                "spurious**. This is the boundary between the biochemical <b>representation</b> and "
                "the surface <b>observation</b> — and the empirical seed of a future observation "
                "model.")

    # ── Tab 3 — adenine ──
    with tabs[2]:
        r = V.get("3_adenine_dose", {})
        ui.question("Does a controlled dose move the correct biochemical theme, and how?")
        ui.stat_row([
            (ui.fmt(r.get("monotonicity_rho"), 3), "Spearman ρ (purine)"),
            (r.get("best_dose_model"), "best dose model"),
            (ui.fmt(r.get("saturating_K_uM"), 2) + " µM", "Langmuir K"),
            (ui.fmt(r.get("saturating_r2"), 3), "Langmuir R²"),
        ])
        if r.get("levels_uM"):
            st.plotly_chart(
                C.dose_curve(r["levels_uM"], r["theme_series"], name="adenine", color=T.NAVY,
                             ytitle="nucleic_purine share"),
                width="stretch", config={"displayModeBar": False})
        st.markdown(
            f"As adenine rises 0→1.8 µM the **purine theme climbs monotonically** "
            f"(ρ={ui.fmt(r.get('monotonicity_rho'),3)}), following a near-textbook **saturating "
            f"(Langmuir) adsorption law** (K={ui.fmt(r.get('saturating_K_uM'),2)} µM, "
            f"R²={ui.fmt(r.get('saturating_r2'),3)}). The motion lifts the whole purine subsystem "
            f"(components c0/c3/c15), not a single axis.")
        ui.takehome("The correct theme, the correct direction, a saturating dose law — recovered "
                    "from a frozen Raman basis on a SERS series it never saw. The strongest single "
                    "validation of the theme layer.")

    # ── Tab 4 — ergothioneine ──
    with tabs[3]:
        r = V.get("4_ergothioneine_dose", {})
        ui.question("Does a second, chemically distinct analyte drive its own correct theme?")
        ui.stat_row([
            (ui.fmt(r.get("monotonicity_rho"), 3), "Spearman ρ (sulfur)"),
            (r.get("best_dose_model"), "best dose model"),
            (ui.fmt(r.get("straightness"), 2), "trajectory straightness"),
        ])
        if r.get("levels_uM"):
            st.plotly_chart(
                C.dose_curve(r["levels_uM"], r["theme_series"], name="ergothioneine", color=T.GOOD,
                             ytitle="sulfur_antioxidant share"),
                width="stretch", config={"displayModeBar": False})
        st.markdown(
            f"Ergothioneine drives the **sulfur/antioxidant theme** monotonically "
            f"(ρ={ui.fmt(r.get('monotonicity_rho'),3)}), also saturating. Its swing is smaller than "
            "adenine's because sulfur is a lower-share theme with a single clean exemplar.")
        ui.note("caveat",
                "Ergothioneine is a weaker adsorber against a stronger serum background, so the "
                "<b>absolute</b> composition moves only a little. The informative view of any "
                "perturbation is the <b>Δ</b> (change vs baseline) and the <b>elevation</b> (vs pure "
                "references), not the absolute radar — a point that matters throughout Page 7.")
        ui.takehome("A second chemically-distinct analyte drives a second chemically-correct theme "
                    "— the theme axes mean what they say.")

    # ── Tab 5 — serum ──
    with tabs[4]:
        r = V.get("5_serum_spike", {})
        ui.question("In real serum on silver, which spiked analytes remain recoverable?")
        ui.stat_row([
            (r.get("n_analytes"), "analytes spiked"),
            (r.get("strong_recovery"), "strong"),
            (r.get("moderate_recovery"), "moderate"),
            (r.get("weak_recovery"), "weak / matrix-dominated"),
            (ui.fmt(r.get("median_direction_agreement"), 2), "median direction agreement"),
        ])
        st.markdown(
            "**Recoverability** = cosine between the serum-spike direction and the analyte's own "
            "pure-SERS fingerprint. Only strong Ag adsorbers survive serum competition:")
        ex = r.get("strong_examples", [])
        if ex:
            st.markdown("**Strongly recoverable:** " + ", ".join(ex))
        a, b = st.columns(2)
        with a:
            ui.card("Why most fail",
                    "Competitive adsorption, matrix suppression and steric hindrance by serum "
                    "protein mean a molecule can be **abundant yet invisible** on silver. "
                    "Concentration does not predict SERS visibility.")
        with b:
            ui.card("Why this is the right answer",
                    "A representation failure would recover the *wrong* theme. Instead the atlas "
                    "recovers *nothing spurious* for the weak analytes — the honest failure mode. "
                    "It independently reproduces the source paper: serum SERS ≈ uric acid + "
                    "hypoxanthine.")
        ui.takehome("Recoverability is a property of the measurement, not the atlas. The strong set "
                    "is the same oxopurines that transfer (Tab 2) and dose-respond (Tab 3).")

    # ── Tab 6 — uricase ──
    with tabs[5]:
        r = V.get("6_uricase_depletion", {})
        ui.question("Does enzymatic removal of urate localise to the expected spectral motif?")
        ui.stat_row([
            (ui.fmt(r.get("purine_delta"), 3), "Δ purine theme"),
            (ui.fmt(r.get("delta_oxopurine_motif"), 3), "Δ oxopurine motif"),
            (ui.fmt(r.get("delta_purine_ring_motif"), 3), "Δ purine-ring motif"),
        ])
        dth = r.get("delta_theme", {})
        if dth:
            st.plotly_chart(
                C.delta_bar([k.replace("_", " ") for k in dth], list(dth.values()),
                            xtitle="Δ theme share (spiked+uricase − spiked)"),
                width="stretch", config={"displayModeBar": False})
        st.markdown(
            f"At the coarse **theme** level the purine change is small and diffuse "
            f"(Δ={ui.fmt(r.get('purine_delta'),3)}; compositional closure spreads it across other "
            f"themes). But at the **MSS motif** level the **oxopurine-carbonyl motif drops sharply "
            f"(Δ={ui.fmt(r.get('delta_oxopurine_motif'),3)})** — the single largest motif change — "
            f"while the generic purine-ring motif is unchanged.")
        ui.note("take",
                "This is chemically exact: uricase removes <b>urate</b>, an oxopurine, so the "
                "<b>oxopurine carbonyl</b> motif — not the generic purine motif — is what "
                "disappears. The MSS layer localises a perturbation the coarse radar smears. "
                "\"MSS resolves what themes hide.\"")

    ui.rule()
    ui.report_expander("VALIDATION_SUMMARY.md", "Read the full validation summary (Parts 9–10)")
