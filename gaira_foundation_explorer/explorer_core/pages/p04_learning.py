"""Page 4 — Learning the biochemical reference space (centerpiece)."""
from __future__ import annotations
import pandas as pd
import streamlit as st

from .. import data as D, ui, charts as C, theme as T


def render():
    h = D.headline()
    cmp = D.selection_comparison()
    ui.page_header("The centerpiece", "Learning the biochemical reference space",
                   "How 375 pure-Raman spectra become a frozen, 24-axis biochemical coordinate "
                   "system — and why non-negative matrix factorization is the right tool.")
    ui.question("What is the biochemical coordinate system, mathematically — and why NMF?")

    ui.stat_row([
        (f"k = {h['k']}", "latent components"),
        (ui.fmt(h["explained_variance"], 3), "explained variance"),
        (ui.fmt(h["mean_stability"], 2), "mean component stability"),
        (h["fingerprint"][:10] + "…", "frozen fingerprint"),
        ("0.0 diff", "rebuild vs frozen"),
    ])

    ui.rule()
    ui.section("4.1", "What NMF does:  V ≈ W · H")
    ui.flow([("Reference spectra", "375 × 676"), ("Matrix V", "spectra × bins"),
             ("NMF", "non-negative"), ("H · basis", "24 × 676"), ("W · coefficients", "375 × 24")],
            highlight={2})
    a, b = st.columns([1.1, 1])
    with a:
        st.markdown(
            "Stack the corpus into one matrix **V** (375 spectra × 676 wavenumber bins). NMF "
            "factors it into two non-negative matrices:")
        st.latex(r"V \;\approx\; W\,H,\qquad W\ge 0,\; H\ge 0")
        st.markdown(
            "- **H** (24 × 676) — the **basis spectra**: 24 learned Raman *motifs*, each a full "
            "spectrum with a few characteristic bands. These are the frozen coordinate axes.\n"
            "- **W** (375 × 24) — the **coefficients**: how strongly each analyte uses each motif. "
            "After normalization, a row of W is a *proportion* — the seed of the biochemical state "
            "vector.")
        st.markdown("Every spectrum is rebuilt as a **non-negative sum of the 24 motifs**:")
        st.latex(r"V_i \;\approx\; \sum_{j=1}^{24} W_{ij}\, H_j")
    with b:
        ui.card("Reading the matrices",
                "- **rows of V** = spectra\n- **cols of V** = wavenumbers\n"
                "- **rows of H** = basis spectra (motifs)\n- **cols of W** = latent components\n\n"
                "A *new* spectrum holds **H fixed** and solves only for its coordinates "
                "(non-negative least squares) — the atlas can never be altered by a query.")

    ui.rule()
    ui.section("4.2", "The 24 basis spectra — the coordinate axes")
    ui.figure_card(
        "basis_grid_24.png",
        question="What do the learned axes actually look like?",
        method="Plot each row of H (a basis spectrum) over 450–1800 cm⁻¹.",
        result="24 distinct, band-structured Raman motifs — e.g. adenine purine modes (722/1334), "
               "phenylalanine 1004, acyl CH₂, pyrimidine 780.",
        interpretation="The axes are readable spectra, not abstract directions of variance.",
        takehome_text="Because the basis is non-negative and band-localised, each axis can be "
                      "assigned to chemistry — the foundation of every interpretive layer that "
                      "follows.")

    ui.rule()
    ui.section("4.3", "Why Raman mixtures suit NMF")
    c1, c2, c3 = st.columns(3)
    with c1:
        ui.card("Additivity",
                "A mixture's Raman spectrum is (to first order) the **sum** of its components' "
                "spectra. That is exactly V = Σ W·H — the model *is* the physics.")
    with c2:
        ui.card("Non-negativity",
                "Intensity is non-negative and a molecule cannot be present in negative amount. "
                "Parts + weights ≥ 0 keeps every coordinate physically meaningful.")
    with c3:
        ui.card("Sparsity",
                "A molecule has a few bands. Non-negativity drives loadings to be localised bundles "
                "of bands — so a component looks like a spectrum you can name.")

    ui.rule()
    ui.section("4.4", "Why NMF, not PCA / ICA / autoencoder / dictionary learning")
    st.markdown(
        "All five were benchmarked identically on an **analyte-grouped, held-out** protocol "
        "(no analyte in both train and test), scored on six criteria that deliberately weight "
        "*structure and interpretability* over raw reconstruction.")
    df = D.benchmark()
    if len(df):
        st.plotly_chart(C.benchmark_scatter(df), width="stretch",
                        config={"displayModeBar": False})
        st.markdown('<div class="small">Selection score vs latent dimension k. NMF is highlighted; '
                    'the other four representations are context (distinct marker shapes, '
                    'direct-labelled). The gold band is the 0.02 statistical-tie zone. Hover any '
                    'point for its reconstruction error, stability and sparsity.</div>',
                    unsafe_allow_html=True)
    ui.figure_card(
        "nmf_selection_criteria.png",
        question="On which criteria does NMF actually win?",
        method="Break the winner NMF k=24 against raw-top ICA k=32 and PCA k=24 across the six "
               "normalized sub-scores.",
        result="NMF leads on component stability and interpretability (sparsity/band-localisation) "
               "and is the only non-negative candidate; ICA edges it only on reconstruction.",
        interpretation="ICA's advantages serve goals we down-weight; its components are signed and "
                       "delocalised.",
        takehome_text="NMF is the one representation that is simultaneously faithful to mixture "
                      "physics, interpretable, and reproducible.")

    # comparison table (data-driven)
    if cmp:
        rows = []
        for key, v in cmp.items():
            rows.append({"model": key.replace("_k", " k="), "score": v.get("total_score"),
                         "recon err": v.get("recon_rel_error"), "stability": v.get("component_stability"),
                         "sparsity": v.get("loading_sparsity"),
                         "non-neg": "✓" if v.get("nonneg") else "✗"})
        cdf = pd.DataFrame(rows).sort_values("score", ascending=False)
        st.dataframe(cdf, hide_index=True, width="stretch")

    ui.rule()
    ui.section("4.5", "The honest twist: ICA technically won, NMF is the scientific choice")
    raw = h.get("raw_top", {})
    sel = h.get("selected", {})
    ui.stat_row([
        (f"{raw.get('representation','ICA')} k={raw.get('k',32)}", "raw benchmark winner"),
        (ui.fmt(raw.get("total_score"), 4), "its score"),
        (f"{sel.get('representation','NMF')} k={sel.get('k',24)}", "selected"),
        (ui.fmt(sel.get("total_score"), 4), "its score"),
    ])
    st.markdown(
        f"The raw top score is **signed ICA k=32** ({ui.fmt(raw.get('total_score'),4)}); "
        f"**NMF k=24** is second ({ui.fmt(sel.get('total_score'),4)}) — a difference of ~0.0002, "
        "deep inside the noise of a 4-fold estimate. Five candidates tie within 0.02. The tie is "
        "broken by a **pre-stated physical rule**, not by the third decimal:")
    st.markdown(D.selection_repro().get("tie_break", ""))
    ui.note("info",
            "A biochemical <i>proportion</i> cannot be negative. Within the statistical tie, only "
            "non-negative (parts-based) decompositions are admissible → NMF; then by score, then by "
            "the smaller k → <b>NMF k=24</b>. The physics decides, and it is decided in advance.")

    ui.rule()
    ui.section("4.6", "Why k = 24, and why you can trust the whole thing")
    a, b = st.columns(2)
    with a:
        ui.card("Why 24 components",
                "The corpus's intrinsic dimensionality is ~15–16 (participation ratio 15.2; 90% of "
                "latent variance in 16 components). k=24 sits deliberately *above* that — mild "
                "over-completeness so chemically distinct-but-correlated motifs (purine vs "
                "pyrimidine, acyl vs sterol) separate instead of sharing an axis. Crucially, the "
                "extra components did **not** create duplicates: the largest pairwise basis cosine "
                "is only **0.52**.")
    with b:
        ui.card("Reproducible to the bit",
                f"Rebuilding NMF k=24 from the raw corpus reproduces the frozen components with "
                f"**max difference {ui.fmt(h['max_diff_vs_committed'],1)}** → an identical "
                f"fingerprint. The full 30-cell benchmark reproduces to ~1e-16 with an "
                f"**identical ranking** "
                f"({'✓' if h.get('ranking_identical') else '—'}). The coordinate system is a "
                f"deterministic function of the corpus + a seeded pipeline.")
    ui.note("take",
            "k=24 is re-derived as optimal under the same methodology; the selection hinges on an "
            "explicit, physically-motivated non-negativity constraint; and the entire result is "
            "byte-for-byte reproducible. <b>This is the frozen biochemical coordinate system.</b>")
    ui.report_expander("NMF_EXPLAINED.md", "Read the full NMF explanation (Part 5)")
    ui.report_expander("NMF_REBUILD.md", "Read the rebuild & k-selection report (Part 4)")
