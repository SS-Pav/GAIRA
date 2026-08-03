"""The 18 pages of Foundation Explorer V5. Each renders without exception under AppTest."""
from __future__ import annotations
import numpy as np, pandas as pd
import plotly.graph_objects as go
import streamlit as st
from . import data as D
from . import ui
from .ui import OI, INK

CFG = {"displayModeBar": False}


def _lay(fig, h=430, **k):
    fig.update_layout(height=h, margin=dict(l=50, r=20, t=40, b=50), plot_bgcolor="white",
                      paper_bgcolor="white", font=dict(color=INK, size=13), **k)
    fig.update_xaxes(gridcolor="#eef1f4", zeroline=False); fig.update_yaxes(gridcolor="#eef1f4", zeroline=False)
    return fig


def p01_overview():
    s = D.summary()
    ui.header("GAIRA Foundation Model Explorer V5",
              "From exact molecular identity to recoverable biochemical abstraction",
              "When exact analyte identity is lost after Ag-SERS, does the correct BROADER chemistry — "
              "component, motif, subclass, theme — still survive? This explorer measures graceful "
              "degradation up GAIRA's hierarchy, keeping a strict line between molecular identity, "
              "spectral motif, chemical subclass and broad theme.")
    st.markdown("Raman atlas → exact identity → components → MSS → subclass → family → themes → "
                "perturbation → matrix.")
    g = s.get("graded_tiers", {})
    ui.stats([(f"{s.get('exact_identity',{}).get('latent','—')}/51", "exact identity"),
              (f"{g.get('mss_present','—')}/48", "MSS present (top-3)"),
              (f"{g.get('mss_specific','—')}/48", "MSS specific"),
              (f"{g.get('theme_present','—')}/51", "theme present"),
              (f"{g.get('theme_specific','—')}/51", "theme specific")])
    ui.warn("Higher abstraction improves interpretability and may raise apparent PRESENCE — but it "
            "reduces molecular specificity. Presence is not recovery.")
    if D.reproduces_v4():
        ui.good("Reproduces V4 exact-identity counts (7/3/4). V5 adds abstraction-level evaluation only; "
                "no frozen asset changed; subclass is an evaluation overlay, not a learned layer.")
    ui.figure(D.figure("fig01_evaluation_hierarchy.png"), "The V5 evaluation hierarchy.")
    ui.caption(f"Frozen atlas {D.CANON_FINGERPRINT}, verified at load.")


def p02_dataset():
    ui.header("Foundation Dataset", "Raman trains; SERS validates",
              "The atlas is learned once from pure-Raman spectra. Ag-SERS is projected through the "
              "frozen basis — validation only, never training.")
    st.table(pd.DataFrame([
        ["Pure-Raman corpus", "Raman", "FOUNDATION (learns NMF)", "375 / 167"],
        ["Pure Ag-SERS", "Ag-SERS", "validation", "265 / 53 / 5 reps"],
        ["Unspiked serum blank", "Ag-SERS", "background control", "15"],
        ["Perturbation", "Ag-SERS", "functional", "adenine/ergo/uricase"],
        ["Serum spike", "Ag-SERS", "matrix", "phase-7"]], columns=["dataset", "modality", "role", "size"]))


def p03_latent():
    ui.header("Latent NMF Atlas", "24 emergent components — not subclasses",
              "GAIRA projects onto a frozen non-negative basis of 24 components. A component is an "
              "emergent spectral pattern, NOT a chemical subclass — subclass is a separate evaluation overlay.")
    ui.note("Exact latent-fingerprint identity is recovered for only 7/51 analytes (V4). Components "
            "rarely survive the Ag-SERS reshaping intact (2/51 component-set overlap).")


def p04_interpret():
    ui.header("How GAIRA Interprets a Spectrum", "The cascade — plus the optional subclass overlay",
              "spectrum → 24 coordinates → MSS motifs → broad themes / BSV → domain interpretation. The "
              "molecular-subclass evaluation is layered on SEPARATELY, never fed back into the ontology.")
    ui.warn("Abstraction increases apparent presence but not analyte specificity. The subclass overlay "
            "is used only to measure resolution, and is never a frozen GAIRA axis.")


def p05_exact():
    s = D.summary().get("exact_identity", {})
    ui.header("Exact Analyte Recovery", "The strictest test (inherited from V4)",
              "Can the correct molecule be distinguished from all others? Rank-1 + jackknife-stable, "
              "null-calibrated.")
    ui.stats([(f"{s.get('latent','—')}/51", "latent-specific"), (f"{s.get('MSS','—')}/51", "MSS-specific"),
              (f"{s.get('theme','—')}/51", "theme-specific")])
    ui.note("These V4 results are unchanged. Exact recovery is rare — which is exactly why V5 asks "
            "whether broader chemistry survives instead.")


def p06_component():
    ui.header("Component Evidence", "Do the emergent NMF components survive?",
              "Top-3 Raman↔Ag-SERS component overlap, component-mass retention, mismatched null.")
    df = D.analytes()
    ui.stats([(int(df.component_recovered.sum()), "component-recovered / 51"),
              (round(df.comp_top3_overlap.mean(), 2), "mean top-3 overlap"),
              (round(df.comp_mass_retained.mean(), 2), "mean mass retained")])
    fig = go.Figure(go.Histogram(x=df.comp_top3_overlap, nbinsx=7, marker_color=OI["verm"]))
    _lay(fig, 340, xaxis_title="Raman↔Ag-SERS top-3 component overlap", yaxis_title="analytes")
    st.plotly_chart(fig, use_container_width=True, config=CFG)
    ui.note("A component id is an emergent basis pattern, not a chemical subclass. Exact component "
            "evidence rarely survives (2/51).")


def p07_mss():
    ui.header("MSS Motif Recovery", "What spectral chemistry is present — not which molecule",
              "Expected motif assigned from chemistry + Raman activation (NOT Ag-SERS height). Graded: "
              "present / enriched / specific.")
    g = D.summary().get("graded_tiers", {})
    ui.stats([(f"{g.get('mss_present','—')}/48", "present (top-3)"),
              (f"{g.get('mss_enriched','—')}/48", "enriched > null"),
              (f"{g.get('mss_specific','—')}/48", "specific")])
    ui.figure(D.figure("fig06_mss_motif_ranking.png"), "Null-adjusted expected-motif recovery.")
    st.dataframe(D.analytes()[["analyte", "broad_family", "expected_mss", "mss_rank_S",
                               "mss_present_top3", "mss_motif_recovered", "mss_status"]],
                 use_container_width=True, hide_index=True)
    ui.warn("MSS says what spectral chemistry is present, NOT which exact molecule. Presence (40%) is "
            "common; specific recovery (2/48) is rare.")


def p08_subclass():
    ui.header("Molecular Subclass Recovery", "An evaluation overlay — not a frozen GAIRA axis",
              "Leave-one-analyte-out nearest-centroid classification in latent/MSS/theme spaces, with a "
              "same-modality Raman→Raman control.")
    ui.figure(D.figure("fig05_classification_control.png"),
              "Raman→Raman control (green) rises with abstraction; Ag-SERS cross-modal (bars) at chance.")
    ui.warn("Cross-modal subclass/family classification is AT CHANCE (balanced accuracy 0.03–0.18, all "
            "permutation p non-significant). The Raman control proves the classes ARE separable and "
            "abstraction helps within Raman — the Ag-SERS modality gap collapses it.")
    ui.figure(D.figure("fig07_confusion.png"), "Cross-modal confusion — off-diagonal dominates.")
    st.dataframe(D.classification(), use_container_width=True, hide_index=True)
    ui.note("Subclasses are an evaluation overlay only; 15 exploratory singletons are excluded from the "
            "primary accuracy denominator (they cannot be classified under leave-one-analyte-out).")


def p09_theme():
    ui.header("Biochemical Theme Recovery", "Broad interpretation present, but not specific identity",
              "Expected-theme rank, top-k inclusion, family-mismatched-null enrichment, and "
              "background (blank) correction. Raw cosine is never used alone.")
    g = D.summary().get("graded_tiers", {})
    ui.stats([(f"{g.get('theme_present','—')}/51", "present (top-3)"),
              (f"{g.get('theme_enriched','—')}/51", "enriched > null"),
              (f"{g.get('theme_specific','—')}/51", "specific")])
    ui.figure(D.figure("fig08_theme_recovery.png"), "Expected-theme rank + enrichment.")
    ui.warn("Half of analytes show the expected theme in the Ag-SERS top-3 (broad interpretation), but "
            "that presence is dominated by the shared Ag background; only 1/51 is specifically recovered.")


def p10_by_level():
    ui.header("Recovery by Abstraction Level ★", "How much useful chemistry survives, level by level?",
              "The centerpiece: exact counts, denominators, and the highest defensible level per analyte.")
    ui.figure(D.figure("fig02_recovery_by_level.png"),
              "PRESENCE (light) rises with abstraction; SPECIFIC recovery (dark) and classification "
              "(grey, at chance) stay low.")
    st.dataframe(D.ladder(), use_container_width=True, hide_index=True)
    ui.figure(D.figure("fig04_highest_level.png"), "Highest statistically-defensible level per analyte.")
    ui.takehome("Presence ≠ recovery. Abstraction raises apparent presence via a shared attractor, not "
                "analyte-specific recovery; the modality gap collapses class recovery to chance. Specific "
                "recovery beyond a strong-chemisorber minority comes only from functional perturbation.")


def p11_purine():
    ui.header("The Purine Attractor", "Genuine presence vs common-background attraction",
              "The serum blank is purine-dominant before any analyte (V4). V5 separates non-purines that "
              "retain their expected chemistry from those merely pulled toward purine.")
    ui.figure(D.figure("fig09_purine_correction.png"),
              "Non-purines that keep their expected motif in top-3 despite purine pull.")
    ui.warn("A purine-dominant Ag-SERS theme is often the background attractor, NOT recovered purine "
            "chemistry. For the CoA cofactors the purine theme is legitimate (they contain adenine).")


def p12_perturbation():
    ui.header("Perturbation Validation", "The strongest recovery beyond exact identity",
              "Functional response recovers class chemistry that static Ag-SERS cannot. Three analytes only.")
    t1, t2, t3 = st.tabs(["Adenine", "Ergothioneine", "Uricase"])
    with t1: ui.good("Adenine — exact identity weak, but dose → nucleic_purine ρ=0.996. Purine-level "
                     "dose behaviour is recoverable; no molecular ID claimed from the static spectrum.")
    with t2: ui.good("Ergothioneine — static identity weak, dose → sulfur_antioxidant ρ=0.927.")
    with t3: ui.good("Urate (uricase) — motif-specific depletion (oxopurine_carbonyl Δ=−0.060) stronger "
                     "than the broad-theme change; directional, at the motif layer.")
    ui.figure(D.figure("fig11_perturbation_overlay.png"), "Functional perturbation vs the static ladder.")
    ui.warn("Only 3/51 have perturbation data; GAIRA never extrapolates a functional claim.")


def p13_matrix():
    ui.header("Matrix Recoverability", "A separate property — kept separate",
              "Pure-analyte abstraction recovery does not guarantee serum visibility.")
    ui.figure(D.figure("fig12_abstraction_vs_serum.png"),
              "Pure abstraction recovery does not predict serum-strong tier.")
    ui.warn("Matrix competition, adsorption bias and concentration intervene. Serum is never used to "
            "define pure-analyte subclass or motif recovery.")


def p14_individual():
    ui.header("Individual Analytes", "The complete evidence card, every level",
              "Select an analyte to see its component / MSS / subclass / theme / perturbation / matrix evidence.")
    cards = D.cards()
    if not cards:
        st.info("Cards not found — run make_cards_v5.py."); return
    names = sorted(cards)
    a = st.selectbox("Analyte", names, index=(names.index("adenine") if "adenine" in names else 0))
    c = cards[a]
    st.markdown(f"### {a} · *{c['broad_family']} / {c['subclass']}*")
    ui.note(f"<b>Conclusion.</b> {c['conclusion']}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Exact identity** — {c['exact_identity']['verdict']}")
        st.markdown(f"**NMF component** — top-3 overlap {c['nmf_component']['top3_overlap']}, "
                    f"recovered {c['nmf_component']['recovered']}")
        st.markdown(f"**MSS motif** — expected `{c['mss_motif']['expected']}` (rank "
                    f"{c['mss_motif']['rank_agsers']}): **{c['mss_motif']['tier']}**")
    with col2:
        st.markdown(f"**Theme** — expected `{c['biochemical_theme']['expected']}` (rank "
                    f"{c['biochemical_theme']['rank_agsers']}): **{c['biochemical_theme']['tier']}**")
        st.markdown(f"**Subclass (overlay)** — NN same-subclass {c['molecular_subclass']['nn_same_subclass']}")
        st.markdown(f"**Perturbation** — {c['perturbation']['status']} · **Matrix** — {c['matrix']['tier']}")
    st.caption(f"Δpurine {c['delta_purine']} · OOD {c['ood_sers']} · confidence {c['confidence_sers']} "
               "· no molecular identification is claimed from motif/theme evidence alone.")


def p15_biological():
    ui.header("Biological Studies", "Real cohorts (unchanged)",
              "Biological validation is carried over unchanged; this pass does not reanalyse cohorts. "
              "See Explorer V1/V3 for the interactive biological pages.")
    ui.note("The abstraction limits here explain the biology: only strong Ag chemisorbers carry "
            "analyte-specific signal into serum, consistent with serum SERS ≈ uric acid + hypoxanthine.")


def p16_limitations():
    ui.header("Limitations", "What this analysis cannot claim", "Stated plainly.")
    for t in ["Raman-trained atlas; Ag-SERS surface-selection effects.",
              "Purine attractor present in the background before any analyte.",
              "Low exact identity (7/51); subclass imbalance (15 exploratory singletons).",
              "Only 3 perturbation examples; matrix competition.",
              "No learned Au-SERS / Raman→SERS observation model.",
              "Confidence is not identifiability.",
              "Cross-modal centroid classification carries the global-shift confound (reported with the Raman control)."]:
        st.markdown(f"- {t}")


def p17_dart():
    ui.header("Future — DART", "Dynamic trajectories recover what static spectra cannot",
              "Dynamic Analyte Response Tracking: perturb the sample and read the trajectory.")
    ui.good("Functional perturbation already recovers class chemistry that static Ag-SERS cannot "
            "(adenine's purine dose-response). Extending controlled perturbation is the route to "
            "class-specific recovery — the recommended next experiment. A learned Raman→SERS observation "
            "model (the Raman→Raman control shows the information exists) is the complementary path.")


def p18_methods():
    ui.header("Methods & Provenance", "Formulas, overlay, nulls, splits, fingerprint",
              "The full specification and provenance.")
    t1, t2, t3 = st.tabs(["Evaluation hierarchy & metrics", "Overlay provenance", "Full report"])
    with t1: st.markdown(D.doc("EVALUATION_HIERARCHY_AND_METRICS.md"))
    with t2: st.markdown(D.doc("ANALYTE_CLASSIFICATION_PROVENANCE.md"))
    with t3: st.markdown(D.doc("GAIRA_Pure_AgSERS_Abstraction_Recovery_V5.md"))


PAGES = [
    ("1 · Overview", p01_overview), ("2 · Foundation Dataset", p02_dataset),
    ("3 · Latent NMF Atlas", p03_latent), ("4 · How GAIRA Interprets a Spectrum", p04_interpret),
    ("5 · Exact Analyte Recovery", p05_exact), ("6 · Component Evidence", p06_component),
    ("7 · MSS Motif Recovery", p07_mss), ("8 · Molecular Subclass Recovery", p08_subclass),
    ("9 · Biochemical Theme Recovery", p09_theme), ("10 · Recovery by Abstraction Level ★", p10_by_level),
    ("11 · The Purine Attractor", p11_purine), ("12 · Perturbation Validation", p12_perturbation),
    ("13 · Matrix Recoverability", p13_matrix), ("14 · Individual Analytes", p14_individual),
    ("15 · Biological Studies", p15_biological), ("16 · Limitations", p16_limitations),
    ("17 · Future DART", p17_dart), ("18 · Methods & Provenance", p18_methods),
]
