"""Page 7 — Future DART. How the static GAIRA framework extends to controlled
electrochemical perturbation. CONCEPTUAL and interface-level only: no simulated
result is presented as experimental DART data. The only real data shown are the
completed calibration trajectories (Page 4), used as precedent.
"""
from __future__ import annotations
import streamlit as st

from .. import components as C, figures as F, calibration as CAL


def render(bridge):
    s = bridge.platform_stats()
    C.page_header(
        "Roadmap · dynamics",
        "How DART naturally extends GAIRA",
        "A static spectrum is one point in the frozen biochemical state space. A dose series traces "
        "a path. DART (dynamic acquisition under electrochemical control) turns interpretation into "
        "a <i>trajectory</i> through the same fixed coordinate system — a new observation model, not "
        "new biochemistry.")
    C.question("If a single spectrum is one BSV point, what does a time- and potential-resolved "
               "acquisition look like in the same frozen coordinate system?")
    st.markdown('<div class="gaira-caveat"><b>Not implemented.</b> This page is conceptual and '
                'interface-level. No DART data is claimed or shown; every synthetic diagram is '
                'labelled CONCEPTUAL. The abstract interfaces already exist in '
                '<code>gaira.engine.dart</code>.</div>', unsafe_allow_html=True)

    # Section A
    st.markdown("### A · Static point → dynamic trajectory")
    C.figure(F.dart_concept_ladder(),
             cap="A static spectrum is one BSV point; a dose series is a trajectory; DART adds "
                 "potential × time. (Conceptual.)")
    st.markdown("#### Real precedent — the calibration trajectory classes (Page 4)")
    J = CAL.joint_trajectories(bridge)
    C.figure(F.compare_trajectories([
        {"name": "adenine · redistribution", "proj": J["adenine"]["proj"], "color": F.T.PRIMARY, "marker": "o"},
        {"name": "ergothioneine · scaling", "proj": J["ergothioneine"]["proj"], "color": F.T.GOOD, "marker": "s"},
        {"name": "uricase · depletion", "proj": J["uricase"]["proj"], "color": F.T.UP, "marker": "^"}],
        title="REAL calibration trajectories (dose / depletion) — the DART precedent"),
        cap="These are REAL frozen-atlas trajectories from the Calibration page — the static "
            "analogue of dynamic DART trajectories.",
        interp="DART would replace the dose axis with a controlled potential/time axis while the "
                "biochemical coordinate system stays frozen.")

    # Section B
    st.markdown("### B · DART data model")
    C.figure(F.dart_data_model(),
             cap="The future acquisition object (intensity × wavenumber × potential × time × "
                 "electrode × waveform) and its processing path — each time point projected into "
                 "the SAME frozen atlas. (Conceptual.)")

    # Section C
    st.markdown("### C · Trajectory vocabulary")
    C.figure(F.trajectory_gallery(),
             cap="Conceptual electrochemical-trajectory classes. Illustrative shapes only — the "
                 "axes carry no measured values.")

    # Section D
    st.markdown("### D · Falsifiable predictions (hypotheses)")
    st.markdown(
        '<div class="gaira-card">These are <b>testable hypotheses</b>, not results:'
        "<ul>"
        "<li>A strongly adsorbing analyte with a known component should move along its associated "
        "MSS motifs under favourable electrochemical control.</li>"
        "<li>Reversing the potential should return the trajectory if adsorption is reversible.</li>"
        "<li>Different chemistries should show scaling vs component redistribution (as adenine and "
        "ergothioneine already do statically).</li>"
        "<li>Matrix background may become separable through dynamic response even where static "
        "spectra collide.</li>"
        "</ul></div>", unsafe_allow_html=True)

    # Section E
    st.markdown("### E · Future Au-SERS observation layer")
    st.markdown(
        '<div class="gaira-card">The biochemical reference system stays <b>frozen</b>. Future '
        'Au-SERS data will characterise an explicit <i>observation</i> layer on top of it: '
        'Au-specific background, analyte recoverability, adsorption preferences, potential-'
        'dependent enhancement, cross-substrate transfer, and trajectory reproducibility. '
        '<b>No Au correction model exists today</b> — this page shows the interface seam, not a '
        'learned layer.</div>', unsafe_allow_html=True)

    C.takeaways([
        "DART extends GAIRA by making interpretation a trajectory through the SAME frozen space.",
        "The calibration trajectory classes are the real, static precedent for dynamic DART.",
        "Au-SERS will add an observation model on top of — never inside — the frozen biochemistry.",
    ])
    C.caveats([
        "Everything on this page except the calibration-trajectory figure is conceptual.",
        "No DART or Au-SERS data has been acquired or modelled; predictions are hypotheses.",
    ])
    C.provenance_footer(s)
