"""GAIRA Foundation Explorer — an interactive review article documenting and validating
the frozen Raman biochemical reference space.

    streamlit run gaira_foundation_explorer/app.py

Standalone. Reads everything from results/v5_rebuild/foundation_audit/. Modifies nothing.
"""
from __future__ import annotations
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explorer_core import data as D, ui, theme as T
from explorer_core.pages import (p01_intro, p02_grounding, p03_preprocessing, p04_learning,
                                 p05_components, p06_biochemistry, p07_validation, p08_results,
                                 p09_limitations, p10_future, p11_takeaways)

st.set_page_config(page_title="GAIRA Foundation Explorer", page_icon="🧬",
                   layout="wide", initial_sidebar_state="expanded")

PAGES = [
    ("1 · The GAIRA Foundation Model", p01_intro),
    ("2 · The Grounding Corpus", p02_grounding),
    ("3 · Preprocessing", p03_preprocessing),
    ("4 · Learning the Reference Space", p04_learning),
    ("5 · Understanding the Components", p05_components),
    ("6 · From Components to Biochemistry", p06_biochemistry),
    ("7 · Validation", p07_validation),
    ("8 · Results Summary", p08_results),
    ("9 · Limitations", p09_limitations),
    ("10 · Future Architecture", p10_future),
    ("11 · Scientific Takeaways", p11_takeaways),
]


def main():
    ui.inject_css()

    with st.sidebar:
        st.markdown(
            f'<div style="font-family:Newsreader,serif;font-size:1.35rem;font-weight:600;'
            f'color:{T.NAVY_D};line-height:1.15">GAIRA<br>Foundation Explorer</div>'
            f'<div style="color:{T.FAINT};font-size:.82rem;margin:.3rem 0 .9rem">'
            f'Building &amp; validating the frozen Raman biochemical reference space</div>',
            unsafe_allow_html=True)

        if not D.audit_present():
            st.error("Audit artifacts not found at\n`results/v5_rebuild/foundation_audit/`.")
            st.stop()

        labels = [p[0] for p in PAGES]
        choice = st.radio("Contents", labels, label_visibility="collapsed")

        h = D.headline()
        st.markdown("<hr style='margin:.8rem 0'>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:.78rem;color:{T.FAINT};line-height:1.6'>"
            f"<b style='color:{T.MUTED}'>Atlas under review</b><br>"
            f"{h['representation']} · k={h['k']}<br>"
            f"{h['n_spectra']} spectra · {h['n_analytes']} analytes<br>"
            f"fingerprint<br><code style='font-size:.72rem'>{str(h['fingerprint'])[:24]}…</code></div>",
            unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:.72rem;color:{T.FAINT};margin-top:1rem'>"
            f"An interactive companion to the GAIRA Foundation Model audit. "
            f"Every figure and number is generated from "
            f"<code>results/v5_rebuild/foundation_audit/</code>.</div>",
            unsafe_allow_html=True)

    page = dict(PAGES)[choice]
    page.render()


if __name__ == "__main__":
    main()
