"""Page 7 — Future DART. How dynamics extend the frozen framework. (Scaffold — not implemented.)"""
from __future__ import annotations
import streamlit as st
from .. import components as C


def render(bridge):
    s = bridge.platform_stats()
    C.page_header(
        "Roadmap · dynamics",
        "How DART naturally extends GAIRA",
        "A static spectrum is one point in the biochemical state space. Calibration traces a path. "
        "DART (dynamic acquisition) turns interpretation into a <i>trajectory</i> through the same "
        "frozen coordinate system — no change to the biochemistry, only a new observation model.")
    C.question("If a single spectrum is one BSV point, what does a time-resolved acquisition look "
               "like in the same frozen coordinate system?")

    st.markdown(
        "- **Static spectrum** → one BSV point\n"
        "- **Calibration** → a trajectory (dose ladder)\n"
        "- **DART** → a dynamic biochemical trajectory through the frozen state space")
    st.markdown("**Expected trajectory classes:** scaling · redistribution · loops · hysteresis · "
                "thresholds · reversible · irreversible.")
    st.markdown('<div class="gaira-caveat"><b>Not implemented.</b> This page describes how DART '
                "would plug in via the abstract interfaces already stubbed in "
                "<code>gaira.engine.dart</code>. No DART data is claimed or shown.</div>",
                unsafe_allow_html=True)
    C.scaffold_note([
        "Illustrative trajectory-class diagrams (scaling / redistribution / loop / hysteresis).",
        "The static-point → trajectory conceptual animation over a fixed radar.",
    ])
    C.provenance_footer(s)
