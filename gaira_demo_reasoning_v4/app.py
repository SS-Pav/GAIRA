"""GAIRA V6 — Scientific Reasoning Demo.

The public-facing demonstration of the V6 Converged Reasoning Engine, presented as
an interactive scientific paper. Built ENTIRELY on the frozen engine
(``gaira.engine``) + the derived MSS layer (``gaira.engine.mss``); the demo is
presentation only and modifies no science.

Run (from this folder):
    streamlit run app.py
or:
    ./run_demo.sh
"""
from __future__ import annotations
import streamlit as st

from demo_core import components as C
from demo_core.engine_bridge import Bridge
from demo_core.pages import (
    p1_overview, p2_reference_atlas, p3_reasoning, p4_calibration,
    p5_serum, p6_biological, p7_dart, p8_methods,
)

st.set_page_config(page_title="GAIRA — V6 Reasoning Engine", page_icon="🧬",
                   layout="wide", initial_sidebar_state="expanded")


@st.cache_resource(show_spinner="Loading the frozen GAIRA engine…")
def _bridge():
    return Bridge()


PAGES = [
    ("1 · Overview", p1_overview),
    ("2 · Reference Atlas", p2_reference_atlas),
    ("3 · How GAIRA Reasons", p3_reasoning),
    ("4 · Calibration", p4_calibration),
    ("5 · Serum Spike Stress Test", p5_serum),
    ("6 · Biological Studies", p6_biological),
    ("7 · Future DART", p7_dart),
    ("8 · Methods & Provenance", p8_methods),
]


def main():
    C.inject_css()
    bridge = _bridge()
    labels = [p[0] for p in PAGES]
    # consume a cross-page navigation request from a "related pages" button
    if st.session_state.get("_pending_nav") in labels:
        st.session_state["_page"] = st.session_state.pop("_pending_nav")
    if st.session_state.get("_page") not in labels:
        st.session_state["_page"] = labels[0]
    with st.sidebar:
        st.markdown("### GAIRA")
        st.caption("V6 Converged Reasoning Engine")
        choice = st.radio("Sections", labels, index=labels.index(st.session_state["_page"]),
                          label_visibility="collapsed")
        st.session_state["_page"] = choice
        st.markdown("---")
        st.caption(f"atlas `{bridge.eng.atlas.meta['fingerprint'][:10]}…` · frozen · deterministic")
    dict(PAGES)[choice].render(bridge)


if __name__ == "__main__":
    main()
