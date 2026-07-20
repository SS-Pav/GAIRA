"""Page 1 — Overview. What GAIRA is, in one figure and a handful of numbers."""
from __future__ import annotations
import streamlit as st
from .. import components as C, figures as F


def render(bridge):
    s = bridge.platform_stats()
    C.page_header(
        "GAIRA · V6 Converged Reasoning Engine",
        "A biochemical reasoning engine for Raman & SERS",
        "GAIRA turns a raw vibrational spectrum into a provenance-carrying biochemical "
        "interpretation. It is not a classifier. A frozen Raman reference atlas defines a "
        "fixed biochemical coordinate system; every spectrum is projected into it, explained "
        "through validated spectral motifs, and reported with confidence, out-of-distribution "
        "score and domain caveats.")

    C.stat_row([
        (f"{s['n_reference_spectra']}", "reference spectra"),
        (f"{s['n_reference_analytes']}", "reference analytes"),
        (f"{s['n_components']}", "latent Raman motifs"),
        (f"{s['n_biochemical_mss']}", "spectral signatures (MSS)"),
        (f"{s['n_biochemical_themes']}", "biochemical themes"),
    ])
    st.write("")
    ev = s.get("explained_variance")
    C.stat_row([
        (f"{ev:.0%}" if ev else "—", "atlas variance explained"),
        (f"{s['grid'][0]:.0f}–{s['grid'][1]:.0f}", "cm⁻¹ window"),
        (f"{s['grid'][2]}", "spectral bins @ 2 cm⁻¹"),
        (f"{len(s['sources'])}", "reference sources"),
        ("3", "calibration studies"),
    ])
    st.markdown("<hr/>", unsafe_allow_html=True)

    left, right = st.columns([1.05, 1.0], gap="large")
    with left:
        st.markdown("### The architecture, end to end")
        C.figure(
            F.architecture_diagram(),
            cap="Reference spectra define the frozen atlas; a query is projected into 24 latent "
                "Raman motifs, explained by Molecular Spectral Signatures, aggregated into the "
                "Biochemical State Vector, interpreted for the sample's domain, and reported as "
                "evidence + a radar.",
            interp="The coordinate system is FROZEN. New observation models (Au-SERS, DART) and "
                    "new samples plug in without changing the biochemical coordinates.")
    with right:
        st.markdown("### What GAIRA knows")
        st.markdown(
            f"- **{s['n_reference_spectra']} pure Raman spectra** across "
            f"**{s['n_reference_analytes']} analytes**, harmonised to a common "
            f"{s['grid'][0]:.0f}–{s['grid'][1]:.0f} cm⁻¹ grid.\n"
            f"- Sources: " + ", ".join(f"**{k}** ({v})" for k, v in s["sources"].items()) + ".\n"
            f"- Multiple excitations (785, 1064, 532 nm …) folded into one excitation-invariant space.\n"
            f"- A **frozen NMF k=24** basis — chosen by benchmark, not by default — is the "
            f"canonical coordinate system (fingerprint `{s['fingerprint'][:12]}…`).")
        st.markdown("### Today's working hypothesis")
        st.markdown(
            '<div class="gaira-card">The atlas is built from <b>pure Raman</b>. Today those '
            'Raman-derived biochemical motifs are applied to <b>SERS</b> under one explicit '
            'hypothesis: <i>when adsorption is good, the Raman biochemical fingerprint remains '
            'informative</i>. Where adsorption is poor, GAIRA says so (high OOD, low confidence) '
            'rather than inventing a result. Future Au-SERS and DART will add <b>observation '
            'models</b> on top of — never inside — this frozen biochemical coordinate system.</div>',
            unsafe_allow_html=True)

    st.write("")
    C.takeaways([
        "One frozen biochemical coordinate system underlies every page of this demo.",
        "A spectrum becomes a Biochemical State Vector, not a class label.",
        "Molecular Spectral Signatures are the interpretable bridge from math (components) "
        "to chemistry (themes) — the layer this demo puts at the center.",
        "Every interpretation ships with confidence, an OOD score, and domain caveats.",
    ])
    st.write("")
    C.caveats([
        "The atlas is Raman-only; SERS and serum are out-of-domain by construction and are "
        "reported as such, not silently trusted.",
        "Themes are evidence-backed biochemical <i>systems</i>, not concentrations of single "
        "molecules — radar axes are not independent quantities.",
        "MSS motifs are validated spectral <i>patterns</i>, never exact molecule claims.",
    ])
    C.provenance_footer(s)
