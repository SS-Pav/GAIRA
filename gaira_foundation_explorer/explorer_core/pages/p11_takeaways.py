"""Page 11 — Scientific Takeaways."""
from __future__ import annotations
import streamlit as st

from .. import data as D, ui, theme as T


def render():
    h = D.headline()
    ui.page_header("The key message", "Scientific Takeaways",
                   "What the audit demonstrates, and the vision it makes credible.")

    ui.rule()
    ui.section("11.1", "What the audit demonstrates")
    points = [
        ("Deterministic reconstruction",
         "The frozen atlas rebuilds byte-for-byte from the raw corpus."),
        ("Reproducibility",
         "The full representation benchmark reproduces to floating-point identity."),
        ("Interpretable latent components",
         "24 stable, non-redundant Raman motifs — five clean chemical anchors."),
        ("Chemically meaningful motifs",
         "An MSS layer that maps components to named spectral patterns and themes."),
        ("Universal biochemical coordinates",
         "One frozen 11-axis state space for every spectrum, from any instrument."),
        ("Raman→SERS transfer where physics allows",
         "Strong adsorbers transfer; correct themes recovered under dose and depletion."),
        ("Honest reporting where physics limits transfer",
         "Weak adsorbers flagged out-of-domain; no spurious themes invented."),
        ("Representation ≠ measurement",
         "A clean separation between biochemistry and the surface that observes it."),
    ]
    cols = st.columns(2)
    for i, (t, d) in enumerate(points):
        with cols[i % 2]:
            st.markdown(
                f'<div class="card" style="padding:.8rem 1rem"><b style="color:{T.NAVY_D}">✓ {t}</b>'
                f'<div class="small" style="margin-top:.2rem">{d}</div></div>',
                unsafe_allow_html=True)

    ui.rule()
    ui.section("11.2", "The vision")
    st.markdown(
        f'<div class="card" style="border:1px solid #dce7ef;background:linear-gradient(180deg,#f7fbfd,#eef5fa)">'
        f'<div style="font-family:Newsreader,serif;font-size:1.5rem;line-height:1.4;color:{T.NAVY_D}">'
        f'GAIRA is not a molecule classifier.<br>It is a biochemical coordinate system.</div>'
        f'<div style="margin-top:.8rem;font-size:1.06rem;color:{T.MUTED};line-height:1.6">'
        f'A frozen, reproducible, interpretable reference frame — learned once from pure Raman, '
        f'validated on data it never saw, and honest about where surface physics limits it. '
        f'Future DART and EC-SERS measurements will estimate <b>trajectories</b> through this state '
        f'space, allowing inference of biochemical <b>processes</b> rather than static compositions. '
        f'Every new modality attaches as an observation model on top of the same frozen '
        f'biochemistry — so the coordinates keep meaning the same thing, forever.</div></div>',
        unsafe_allow_html=True)

    ui.stat_row([
        (h["n_spectra"], "pure-Raman spectra"),
        (f"{h['representation']} k={h['k']}", "frozen representation"),
        (ui.fmt(h["mean_stability"], 2), "mean component stability"),
        ("byte-identical", "reproducible"),
        (h["fingerprint"][:10] + "…", "one fingerprint, forever"),
    ])

    ui.rule()
    st.markdown(
        '<div class="small">This interactive review was generated entirely from the GAIRA '
        'Foundation Model audit at <code>results/v5_rebuild/foundation_audit/</code>. '
        'Every figure, table and number is reproducible from the scripts documented there. '
        'The existing GAIRA reasoning demo is a separate application and was not modified.</div>',
        unsafe_allow_html=True)
