"""GAIRA Foundation Model Explorer V2 — Cross-Modal Transfer.

An interactive walkthrough of the four-level validation framework: latent fingerprint
preservation, biochemical theme preservation (with the null control that keeps it honest),
perturbation sensitivity, and matrix recoverability — for the Raman → Ag-SERS jump.

    streamlit run gaira_foundation_explorer_v2/app.py

Additive to gaira_foundation_explorer (V1 is untouched and still runs). Reads ONLY the
committed theme-preservation artifacts + the frozen atlas (fingerprint 09ed804a…). Nothing
is retrained or modified.
"""
from __future__ import annotations
import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))          # v2_core
sys.path.insert(0, str(REPO / "src"))  # gaira (frozen engine, for fingerprint check)

from v2_core import data as D, ui
from v2_core.pages import PAGES

st.set_page_config(page_title="GAIRA Explorer V2 · Cross-Modal Transfer", page_icon="🔬",
                   layout="wide", initial_sidebar_state="expanded")


def main():
    ui.inject_css()
    with st.sidebar:
        st.markdown(
            '<div style="font-family:Newsreader,serif;font-size:1.3rem;font-weight:600;'
            f'color:{ui.INK};line-height:1.15">GAIRA Explorer V2</div>'
            f'<div style="color:{ui.FAINT};font-size:.82rem;margin:.3rem 0 .9rem">'
            'Cross-Modal Transfer · the four-level validation framework</div>',
            unsafe_allow_html=True)
        if not D.present():
            st.error("Theme-preservation artifacts not found at "
                     "`results/v5_rebuild/pure_ag_sers_theme_preservation/`. "
                     "Run the three scripts in its `code/`.")
            st.stop()
        choice = st.radio("Contents", [p[0] for p in PAGES], label_visibility="collapsed")
        fp_ok = D.fingerprint_ok()
        st.markdown("<hr style='margin:.8rem 0'>", unsafe_allow_html=True)
        badge = ("✓ frozen atlas verified" if fp_ok else "⚠ fingerprint mismatch")
        color = ui.OI["green"] if fp_ok else ui.OI["verm"]
        st.markdown(
            f"<div style='font-size:.78rem;color:{ui.FAINT};line-height:1.6'>"
            f"<b style='color:{color}'>{badge}</b><br>"
            f"<code style='font-size:.72rem'>{D.CANON_FINGERPRINT[:24]}…</code><br>"
            "51 analytes · Raman → Ag-SERS<br>SERS validates, never trains.</div>",
            unsafe_allow_html=True)

    dict(PAGES)[choice]()


if __name__ == "__main__":
    main()
