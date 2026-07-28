"""Page 8 — Results Summary (eleven findings)."""
from __future__ import annotations
import streamlit as st

from .. import data as D, ui, theme as T


def render():
    h = D.headline()
    cls = D.component_classification()
    V = D.validation()
    corp = D.corpus_summary()
    f = ui.fmt

    ui.page_header("The eleven findings", "Results Summary",
                   "One card per audit conclusion — the science, not the report. Each states the "
                   "question, the evidence, and the conclusion.")

    tr = V.get("2_gobbato_sers_transfer", {})
    ad = V.get("3_adenine_dose", {})
    se = V.get("5_serum_spike", {})
    ur = V.get("6_uricase_depletion", {})

    findings = [
        ("Reproducibility",
         "Is the frozen coordinate system a deterministic object or a lucky run?",
         f"Rebuilding NMF k={h['k']} from the raw corpus reproduces the frozen components with max "
         f"difference {f(h['max_diff_vs_committed'],1)}; the full benchmark reproduces to ~1e-16 "
         f"with an identical ranking.",
         "The atlas is a deterministic function of the corpus + a seeded pipeline. Reproducible to "
         "the bit."),
        ("Grounding",
         "What knowledge built the model, and is SERS kept out?",
         f"{corp.get('n_spectra')} pure-Raman spectra from {h['n_sources']} sources; the loader "
         f"hard-asserts Raman-only. ~20 SERS datasets exist but are firewalled to validation.",
         "The representation is grounded in pure biochemistry, not surface physics."),
        ("Corpus honesty",
         "Is the corpus as clean as claimed?",
         f"{corp.get('n_analytes')} labels ≈ 161 distinct molecules (6 duplicate labels quantified); "
         f"thin nucleic-acid/porphyrin coverage named explicitly.",
         "Real gaps and label debt are surfaced, not hidden — and do not affect the representation."),
        ("Preprocessing",
         "Is the pipeline minimal and justified?",
         f"Crop→ASLS→SG→resample→L2→clip; the non-negativity clip removes "
         f"{f(D.preprocessing_stats().get('frac_absolute_mass_clipped_to_zero',0)*100,2)}% of "
         f"signal mass; a 120-pipeline search froze nothing (overprocessing).",
         "Standardization does the minimum for comparability and no more."),
        ("Representation choice",
         "Why NMF over PCA/ICA/autoencoder?",
         f"Raw benchmark winner is signed ICA k=32; a pre-stated non-negativity constraint selects "
         f"NMF k={h['k']} among a 5-way statistical tie.",
         "The choice is a physical one — a biochemical proportion cannot be negative — decided in "
         "advance, not by the third decimal."),
        ("Component quality",
         "Are the 24 axes stable, interpretable and distinct?",
         f"Bootstrap stability {f(cls.get('min_stability'),2)}–0.97 (mean {f(cls.get('mean_stability'),2)}); "
         f"max pairwise basis cosine {f(cls.get('max_pairwise_basis_cosine'),2)}; "
         f"{len(cls.get('chemically_clean',[]))} clean anchors.",
         "Stable, non-redundant, largely interpretable — with known collisions documented."),
        ("Interpretability bridge",
         "Does MSS genuinely turn maths into chemistry?",
         "13 motifs derived from the frozen atlas; purine/protein/lipid/glycan strongly grounded, "
         "heme/flavin honestly weak (matching corpus gaps).",
         "A deterministic, traceable path from a latent number to a named biochemical claim."),
        ("Semantic state",
         "Is the BSV a reading or another black box?",
         "11 named biochemical axes, each with confidence + OOD, traceable to reference chemistries; "
         "a frozen 24×13 weight matrix, nothing learned.",
         "The output is a semantic, falsifiable state — not an opaque embedding."),
        ("Raman fidelity & transfer",
         "Does the atlas represent Raman, and transfer to SERS?",
         f"In-domain OOD {f(V.get('1_gobbato_raman',{}).get('mean_ood'),3)}; SERS transfer median "
         f"cosine {f(tr.get('median_coord_cosine'),2)} with OOD 3.5× higher; strong adsorbers "
         f"preserved, weak ones scrambled.",
         "Faithful in-domain; SERS partially transfers and the gap is honestly flagged."),
        ("Dose response",
         "Do controlled perturbations move the right chemistry?",
         f"Adenine→purine ρ={f(ad.get('monotonicity_rho'),3)} (Langmuir K={f(ad.get('saturating_K_uM'),2)} µM); "
         f"ergothioneine→sulfur ρ={f(V.get('4_ergothioneine_dose',{}).get('monotonicity_rho'),3)}.",
         "The theme axes respond monotonically and saturably to the correct analytes."),
        ("Honest failure & localisation",
         "Does it fail safely, and localise a real perturbation?",
         f"Serum: {se.get('strong_recovery')} strong / {se.get('weak_recovery')} weak (no spurious "
         f"themes); uricase depletion localises to the oxopurine motif "
         f"(Δ={f(ur.get('delta_oxopurine_motif'),3)}).",
         "Where surface physics forbids recovery it returns nothing false; where a molecule is "
         "removed, the right motif drops."),
    ]

    for i, (title, q, ev, concl) in enumerate(findings, 1):
        st.markdown(
            f'<div class="card"><h4><span class="section-num">{i}</span>{title}</h4>'
            f'<div class="figmeta"><span class="tag">Question</span>{q}</div>'
            f'<div class="figmeta"><span class="tag">Evidence</span>{ev}</div>'
            f'<div class="takehome"><b>Conclusion.</b> {concl}</div></div>',
            unsafe_allow_html=True)

    ui.rule()
    ui.note("take",
            "Eleven independent lines of evidence — reproducibility, grounding, representation "
            "choice, component quality, interpretability, and validation on unseen data — converge "
            "on one conclusion: <b>the frozen biochemical coordinate system is scientifically "
            "trustworthy, and its limits are known.</b>")
    ui.report_expander("FINAL_ASSESSMENT.md", "Read the full scientific assessment (Part 11)")
