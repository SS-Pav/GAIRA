"""GAIRA Foundation Model Explorer — an interactive Nature-Methods-style walkthrough of
the frozen Raman biochemical foundation model: what built it, how it learns latent
biochemical components, how those become chemistry, how spectra are interpreted, how it
was validated and how it performs on biological datasets, and where it is heading.

    streamlit run gaira_foundation_explorer/app.py

Composes two verified, frozen-atlas UIs into one 8-section narrative:
  - the data-driven review pages (explorer_core) — corpus, NMF, components, limits, future
  - the live reasoning-engine pages (gaira_demo_reasoning_v4/demo_core) — inference,
    calibration, biological cohorts
Both read the SAME frozen atlas (fingerprint 09ed804a…). Nothing is retrained or modified.
"""
from __future__ import annotations
import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))                       # explorer_core
sys.path.insert(0, str(REPO / "gaira_demo_reasoning_v4"))   # demo_core (live reasoning)
sys.path.insert(0, str(REPO / "src"))

from explorer_core import data as D, ui, theme as T
from explorer_core.pages import (p01_intro, p02_grounding, p03_preprocessing, p04_learning,
                                 p05_components, p06_biochemistry, p_calibration,
                                 p09_limitations, p10_future, p11_takeaways)
from demo_core import components as VC                       # v4 CSS + helpers
from demo_core.engine_bridge import get_bridge
from demo_core.pages import p3_reasoning, p6_biological

# The v4 pages emit their own internal "related pages" nav buttons (keyed by v4 page
# labels). In this composed app those cross-links are irrelevant AND collide when two v4
# pages render in one section, so suppress them (no effect on the standalone v4 demo).
VC.related = lambda *a, **k: None

st.set_page_config(page_title="GAIRA Foundation Model Explorer", page_icon="🧬",
                   layout="wide", initial_sidebar_state="expanded")


def _rule():
    st.markdown("<hr style='border:0;border-top:1px solid #e3e7ec;margin:2.2rem 0 1.4rem'>",
                unsafe_allow_html=True)


# ── the 8 sections (compose verified page renders) ──
def s1_overview():
    p01_intro.render()


def s2_dataset():
    p02_grounding.render(); _rule(); p03_preprocessing.render()


def s3_latent():
    p04_learning.render(); _rule(); p05_components.render(); _rule(); p06_biochemistry.render()


def s4_inference():
    p3_reasoning.render(get_bridge())


def s5_calibration():
    p_calibration.render()          # Reference Raman → Pure Ag-SERS → Adenine → Ergo → Uricase → Serum


def s6_biological():
    p6_biological.render(get_bridge())


def s7_limitations():
    p09_limitations.render()


def s8_future():
    p10_future.render(); _rule(); p11_takeaways.render()


SECTIONS = [
    ("1 · Overview", s1_overview),
    ("2 · Foundation Dataset", s2_dataset),
    ("3 · Latent Representation", s3_latent),
    ("4 · Inference Engine", s4_inference),
    ("5 · Calibration & Validation", s5_calibration),
    ("6 · Biological Datasets", s6_biological),
    ("7 · Limitations", s7_limitations),
    ("8 · Future Directions", s8_future),
]


def main():
    ui.inject_css()          # explorer base + classes
    VC.inject_css()          # v4 reasoning classes (gaira-*) — coexists; different class names

    with st.sidebar:
        st.markdown(
            f'<div style="font-family:Newsreader,serif;font-size:1.35rem;font-weight:600;'
            f'color:{T.NAVY_D};line-height:1.15">GAIRA<br>Foundation Model Explorer</div>'
            f'<div style="color:{T.FAINT};font-size:.82rem;margin:.3rem 0 .9rem">'
            f'What built GAIRA · how it learns · how it reasons · how it was validated</div>',
            unsafe_allow_html=True)
        if not D.audit_present():
            st.error("Audit artifacts not found at `results/v5_rebuild/foundation_audit/`.")
            st.stop()
        choice = st.radio("Contents", [s[0] for s in SECTIONS], label_visibility="collapsed")
        h = D.headline()
        st.markdown("<hr style='margin:.8rem 0'>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:.78rem;color:{T.FAINT};line-height:1.6'>"
            f"<b style='color:{T.MUTED}'>Frozen atlas</b><br>{h['representation']} · k={h['k']}<br>"
            f"{h['n_spectra']} spectra · {h['n_analytes']} analytes<br>"
            f"fingerprint<br><code style='font-size:.72rem'>{str(h['fingerprint'])[:24]}…</code></div>",
            unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:.72rem;color:{T.FAINT};margin-top:1rem'>Learned: the 24 NMF "
            f"basis. Derived: registry, themes, MSS, normalization, BSV. SERS is validation, not "
            f"training.</div>", unsafe_allow_html=True)

    dict(SECTIONS)[choice]()


if __name__ == "__main__":
    main()
