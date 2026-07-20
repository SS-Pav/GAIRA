"""Page 5 — Serum Spike Stress Test. Where Ag-SERS works and fails. (Scaffold.)"""
from __future__ import annotations
import streamlit as st
from .. import components as C


def render(bridge):
    s = bridge.platform_stats()
    C.page_header(
        "Stress test · SERS recoverability",
        "Where Ag-SERS succeeds — and where it cannot",
        "Bulk concentration is not surface occupancy, and surface occupancy is not SERS signal. "
        "This page is deliberately honest about the boundary: strong Ag adsorbers are recovered, "
        "weak ones are not, and GAIRA flags the difference instead of hiding it.")
    C.question("For a controlled serum spike, does the analyte reach the silver surface strongly "
               "enough to be recovered — and does GAIRA's confidence reflect that?")

    st.markdown('<div class="gaira-card">'
                "<b>bulk concentration</b> &nbsp;≠&nbsp; <b>surface occupancy</b> &nbsp;≠&nbsp; "
                "<b>SERS signal</b><br>"
                "Ag adsorption, hotspot access, orientation, molecular competition and the serum "
                "matrix all sit between what is present and what is observed.</div>",
                unsafe_allow_html=True)
    st.write("")
    C.takeaways([
        "Strong adsorbers (purines, thiones) are recovered; weak adsorbers (e.g. phenylalanine) "
        "are not — BSV validation confirmed confidence tracks domain OOD, not analyte "
        "recoverability, which is the key limitation to communicate.",
    ])
    C.scaffold_note([
        "Strong / partial / poor recoverability tiers, each with serum baseline, spiked serum, "
        "difference spectrum, BSV change, theme change and an adsorption explanation.",
        "An explicit 'why this analyte cannot currently be recovered' panel per case.",
    ])
    C.caveats([
        "Serum on Ag colloid is out-of-domain for the Raman atlas; high OOD here is correct, "
        "not a failure of the engine.",
        "Confidence currently reports spectrum quality, not identifiability — a matrix-"
        "recoverability prior is a recommended (not-yet-implemented) upgrade.",
    ])
    C.provenance_footer(s)
