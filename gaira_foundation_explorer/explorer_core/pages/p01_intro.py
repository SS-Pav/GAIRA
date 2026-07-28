"""Page 1 — Introduction."""
from __future__ import annotations
import streamlit as st

from .. import data as D, ui, theme as T


def render():
    h = D.headline()
    ui.page_header(
        "An interactive review article",
        "The GAIRA Foundation Model",
        "Building and validating a <i>frozen</i> Raman biochemical reference space — a single "
        "coordinate system into which any spectrum, from any instrument or biofluid, can be "
        "projected and read biochemically.")

    ui.stat_row([
        (h["n_spectra"], "pure-Raman spectra"),
        (h["n_analytes"], "reference analytes"),
        (f"{h['representation']} · k={h['k']}", "frozen representation"),
        (ui.fmt(h["explained_variance"], 3), "explained variance"),
        ("byte-identical", "rebuild vs frozen"),
    ])

    ui.rule()
    ui.section("1.1", "The problem: Raman has no universal coordinate system")
    ui.question("Two labs measure the same biochemistry and get two incompatible models. "
                "Why can't Raman spectra be compared on a common footing?")
    st.markdown(
        "Raman and SERS spectroscopy are exquisitely information-rich, but the field builds a "
        "**new model for every dataset**:")
    c1, c2, c3 = st.columns(3)
    with c1:
        ui.card("Dataset-specific PCA",
                "Principal components are re-fit on each new dataset. PC1 of study A and PC1 of "
                "study B are *different directions* — the coordinates mean nothing across studies.")
    with c2:
        ui.card("Dataset-specific classifiers",
                "A CNN or SVM trained to separate disease vs control learns *that cohort's* "
                "confounds (instrument, batch, substrate). It rarely transfers, and it explains "
                "nothing biochemically.")
    with c3:
        ui.card("No transferability",
                "Because every representation is local, there is no shared language: no way to say "
                "\"this serum sits *here* in biochemical space\" in terms another lab can reuse.")
    ui.note("caveat",
            "The result is a literature of one-off models. Each is internally valid and externally "
            "useless — a coordinate system that exists only inside its own dataset.")

    ui.rule()
    ui.section("1.2", "Local vs global coordinate systems")
    ui.question("What would it take to give every spectrum the <i>same</i> address?")
    a, b = st.columns(2)
    with a:
        st.markdown("**Local (today)** — coordinates re-derived per dataset")
        ui.flow([("Dataset A", "fit PCA→axes_A"), ("Dataset B", "fit PCA→axes_B"),
                 ("Dataset C", "fit PCA→axes_C")], arrow="✕")
        st.markdown('<div class="small">axes_A ≠ axes_B ≠ axes_C — not comparable.</div>',
                    unsafe_allow_html=True)
    with b:
        st.markdown("**Global (GAIRA)** — one frozen coordinate system for all")
        ui.flow([("Frozen atlas", "learned once"), ("Project A", "same axes"),
                 ("Project B", "same axes")], highlight={0}, arrow="→")
        st.markdown('<div class="small">One basis; every spectrum gets coordinates in it.</div>',
                    unsafe_allow_html=True)

    ui.rule()
    ui.section("1.3", "The GAIRA philosophy")
    ui.flow([
        ("Pure Raman references", "167 analytes"),
        ("Learn latent basis", "NMF"),
        ("FREEZE", f"k={h['k']} · {str(h['fingerprint'])[:8]}…"),
        ("Project any spectrum", "same coordinates"),
        ("Read biochemistry", "themes + confidence"),
    ], highlight={2})
    st.markdown(
        "GAIRA learns a biochemical coordinate system **once**, from pure-compound Raman spectra, "
        "then **freezes it** — pins it to a cryptographic fingerprint so it can never drift. Every "
        "future spectrum (serum, SERS, EV, a dose series, a clinical sample) is *projected into the "
        "same space*, never used to re-fit it. The coordinates therefore mean the same thing "
        "forever, across instruments, substrates and studies.")
    ui.note("take",
            "This is the difference between a <b>classifier</b> (learns a boundary for one dataset) "
            "and a <b>coordinate system</b> (a reusable reference frame for all datasets). GAIRA is "
            "the latter.")

    ui.rule()
    ui.section("1.4", "What this application is")
    st.markdown(
        "This is an interactive companion to a complete, first-principles **audit** of that frozen "
        "model. It walks the full construction — corpus → preprocessing → representation learning → "
        "the frozen coordinate system → interpretability → validation — and shows, with every number "
        "regenerated from the audit, *why the resulting biochemical coordinate system is scientifically "
        "trustworthy*. The headline: the atlas was **rebuilt from scratch and reproduced "
        "byte-for-byte** "
        f"(max difference vs the committed build {ui.fmt(h['max_diff_vs_committed'], 1)}), and it "
        "validates on data it never trained on.")
    ui.flow([("Problem"), ("Grounding corpus"), ("Representation"), ("Frozen coordinates"),
             ("Interpretability"), ("Validation"), ("Conclusions")], arrow="→")
    st.caption("Use the contents in the sidebar. Each page answers one scientific question; every "
               "figure carries a question, a method, a result, an interpretation, and a take-home.")
