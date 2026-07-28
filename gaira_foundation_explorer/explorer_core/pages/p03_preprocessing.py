"""Page 3 — Preprocessing."""
from __future__ import annotations
import streamlit as st

from .. import data as D, ui, theme as T

STEPS = [
    ("Crop", "Restrict to the 450–1800 cm⁻¹ analysis window shared by all sources.",
     "Everything outside the common window is undefined for some spectra; NaN-pad it."),
    ("Baseline (ASLS)", "Subtract the broad fluorescence/background continuum "
     "(asymmetric least squares, λ=1e5, p=0.01, 8 iterations).",
     "Biological Raman rides on a large sloping background that would otherwise dominate the fit."),
    ("Smooth (Savitzky-Golay)", "Denoise with a cubic 9-point filter.",
     "Removes shot noise while preserving peak height and width — a moving average would not."),
    ("Interpolate", "Resample onto a fixed 2 cm⁻¹ grid → 676 bins.",
     "Makes spectra directly comparable regardless of native instrument sampling."),
    ("Normalize (L2)", "Scale each spectrum to unit length.",
     "Removes concentration/intensity scale so the representation encodes spectral SHAPE."),
    ("Clip ≥ 0", "Set the few negative points (baseline undershoot) to zero for the NMF fit.",
     "Guarantees the parts-based, non-negative interpretation of the coordinates."),
]


def render():
    ps = D.preprocessing_stats()
    ui.page_header("Standardization", "Preprocessing",
                   "One deterministic pipeline, applied identically to every spectrum — reference or "
                   "query. A projection into a frozen basis is only valid if the query is expressed "
                   "the same way the basis was learned.")
    ui.question("How are spectra standardized so a serum sample and a pure powder can share a "
                "coordinate system?")

    ui.stat_row([
        (ps.get("n_bins", 676), "grid bins"),
        ("2 cm⁻¹", "grid step"),
        ("ASLS + SG + L2", "pipeline"),
        (f"{ps.get('frac_absolute_mass_clipped_to_zero', 0)*100:.1f}%", "signal mass clipped"),
        ("deterministic", "no random seed"),
    ])

    ui.rule()
    ui.section("3.1", "The pipeline, stage by stage")
    ui.figure_card(
        "preprocessing_stages.png",
        question="What does each step do to a real spectrum?",
        method="Apply the canonical pipeline to a raw 785 nm adenine spectrum, stage by stage.",
        result="The sloping background is removed and the sharp bands (e.g. the 722 cm⁻¹ purine "
               "ring-breathing mode) survive cleanly to the final clipped vector.",
        interpretation="Each step is conservative — it standardizes without carving into real bands.",
        takehome_text="By stage 6 the spectrum is a unit-norm, non-negative vector on a shared axis "
                      "— directly comparable to every other spectrum in the atlas.")

    for i, (name, why, rationale) in enumerate(STEPS, 1):
        with st.container():
            st.markdown(
                f'<div class="card"><h4>{i}. {name}</h4>'
                f'<div class="figmeta"><span class="tag">What</span>{why}</div>'
                f'<div class="figmeta"><span class="tag">Why</span>{rationale}</div></div>',
                unsafe_allow_html=True)

    ui.rule()
    ui.section("3.2", "The non-negativity clip removes almost nothing")
    a, b = st.columns([1, 1])
    with a:
        ui.stat_row([
            (f"{ps.get('frac_points_negative_before_clip', 0)*100:.1f}%", "points negative"),
            (f"{ps.get('frac_absolute_mass_clipped_to_zero', 0)*100:.2f}%", "of signal mass clipped"),
        ])
    with b:
        st.markdown(
            "After baseline subtraction and L2 normalization a spectrum has small negative lobes "
            "(undershoot). The NMF fit clips them to zero. Although about **one point in nine** is "
            "slightly negative, those points carry **less than 1% of the total signal mass** — so "
            "the clip is numerical hygiene, not a lossy transform. This is what lets GAIRA claim the "
            "parts-based (non-negative) interpretation at negligible cost to fidelity.")

    ui.rule()
    ui.section("3.3", "Why preprocessing was frozen")
    st.markdown(
        "The pipeline is not a tunable knob. A separate, aggressive search over **120 alternative "
        "pipelines** tried to beat it on cross-modal Raman↔SERS retrieval. The verdict was "
        "**\"apparent improvement is caused by overprocessing\"** — every gain came from stripping a "
        "broad shared component, which collapsed Ag-SERS replicate agreement (0.62 vs 0.95) without "
        "improving matched-vs-mismatched peak specificity. **No pipeline was frozen from the "
        "search.**")
    ui.note("take",
            "The conservative pipeline stands precisely because a thorough search proved that "
            "anything more aggressive damages real spectral structure. Preprocessing does the "
            "<i>minimum</i> required for comparability and no more.")
    ui.report_expander("PREPROCESSING_AUDIT.md", "Read the full preprocessing audit (Part 3)")
