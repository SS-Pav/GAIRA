"""Page 10 — Future Architecture."""
from __future__ import annotations
import streamlit as st

from .. import data as D, ui, theme as T


def render():
    ui.page_header("Where GAIRA goes next", "Future Architecture",
                   "The frozen Raman coordinate system is the foundation. Every future modality — "
                   "Ag-SERS, Au-SERS, EC-SERS, DART — attaches as an <i>observation model</i> on "
                   "top of it, never as a replacement for it.")
    ui.question("How do we add new measurement modalities without re-fitting — and losing — the "
                "universal coordinate system?")

    ui.rule()
    ui.section("10.1", "The observation-model architecture")
    ui.flow([("Raman", "molecular truth"), ("Frozen biochemical state", "the coordinate system"),
             ("Observation model", "per-substrate"), ("Ag / Au / EC-SERS · DART", "measurements")],
            highlight={1, 2})
    st.markdown(
        "The biochemical state space stays **frozen and universal**. On top of it sits a thin, "
        "learnable **observation model** for each measurement modality — a map from *what is "
        "biochemically present* to *what that instrument would see*. Ag-SERS, Au-SERS, "
        "electrochemical SERS and DART each get their own observation model; they share the same "
        "underlying biochemical state.")
    a, b = st.columns(2)
    with a:
        ui.card("Why not just re-fit on SERS?",
                "Because the coordinate system would then be **specific to one surface**. PC/NMF "
                "axes fit on Ag-SERS mean something different on Au-SERS; the universal reference "
                "frame — the entire value of a foundation model — would be lost. Every lab would be "
                "back to local coordinates.")
    with b:
        ui.card("Why an observation model is better",
                "Surface physics (adsorption, orientation, enhancement) is separated from "
                "biochemistry and modelled **explicitly**. The same biochemical state predicts an "
                "Ag spectrum *and* an Au spectrum through two different observation models — and the "
                "difference between them becomes interpretable, not confounding.")

    ui.rule()
    ui.section("10.2", "The seed already exists")
    st.markdown(
        "This is not hypothetical. The audit's Raman→SERS transfer (Page 7, Tab 2) is exactly the "
        "empirical shape of an Ag observation model: 51 analytes with both a pure Raman and a pure "
        "Ag-SERS spectrum, projected through the same frozen atlas, showing precisely which motifs "
        "survive the surface and which are reshaped. A learned, validated version of that map — per "
        "substrate — is the observation model.")
    ui.note("info",
            "The strong Ag adsorbers (oxopurines) transfer with cosine ~0.84; weak adsorbers "
            "(glucose, uracil) are reshaped. An observation model turns that measured gap into a "
            "correction, so a weak-adsorber's biochemistry can be inferred rather than lost.")

    ui.rule()
    ui.section("10.3", "From static states to dynamic trajectories")
    ui.flow([("Static spectrum", "one point"), ("Dose series", "a path"),
             ("DART / EC-SERS", "controlled trajectory"), ("Process inference", "biochemical dynamics")],
            highlight={3}, arrow="→")
    st.markdown(
        "A single spectrum is one point in the frozen state space. A dose series (Page 7) is a "
        "*path*. Dynamic acquisition under electrochemical control (DART / EC-SERS) turns "
        "interpretation into a **trajectory** through the same fixed coordinate system — replacing "
        "the dose axis with a controlled potential/time axis while the biochemistry stays frozen. "
        "That is how GAIRA moves from reading static compositions to **inferring biochemical "
        "processes**.")
    ui.note("take",
            "One frozen biochemical state space; many observation models on top; trajectories "
            "through it over time. New modalities <b>extend</b> GAIRA — they never overwrite the "
            "reference frame.")
