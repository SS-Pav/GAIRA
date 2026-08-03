"""GAIRA Foundation Model Explorer V5 — From exact molecular identity to recoverable biochemical
abstraction.

    streamlit run gaira_foundation_explorer_v5/app.py

Additive: Explorers V1–V4 are untouched and still run. Reads ONLY committed V5 abstraction-recovery
artifacts + the frozen atlas fingerprint (09ed804a…), verified at load. Nothing retrained; V5
changes analysis and interpretation only; subclass is an evaluation overlay, not a learned layer.
"""
from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / "src"))
from v5_core import data as D, ui
from v5_core.pages import PAGES

st.set_page_config(page_title="GAIRA Explorer V5 · Abstraction Recovery", page_icon="🪜",
                   layout="wide", initial_sidebar_state="expanded")


def main():
    ui.inject_css()
    with st.sidebar:
        st.markdown('<div style="font-family:Newsreader,serif;font-size:1.3rem;font-weight:600;'
                    f'color:{ui.INK};line-height:1.15">GAIRA Explorer V5</div>'
                    f'<div style="color:{ui.FAINT};font-size:.82rem;margin:.3rem 0 .9rem">'
                    'From molecular identity to biochemical abstraction</div>', unsafe_allow_html=True)
        if not D.present():
            st.error("V5 artifacts not found at results/v5_rebuild/abstraction_recovery_v5/. "
                     "Run its code/ scripts."); st.stop()
        choice = st.radio("Contents", [p[0] for p in PAGES], label_visibility="collapsed")
        st.markdown("<hr style='margin:.8rem 0'>", unsafe_allow_html=True)
        fp = D.fingerprint_ok(); repro = D.reproduces_v4()
        fc = ui.OI["green"] if fp else ui.OI["verm"]
        st.markdown(f"<div style='font-size:.78rem;color:{ui.FAINT};line-height:1.6'>"
                    f"<b style='color:{fc}'>{'✓ frozen atlas verified' if fp else '⚠ fingerprint mismatch'}</b><br>"
                    f"<code style='font-size:.72rem'>{D.CANON_FINGERPRINT[:24]}…</code><br>"
                    f"{'✓ reproduces V4 identity' if repro else '· V4 reproducibility unrecorded'}<br>"
                    "51 analytes · V1–V4 retained · subclass = overlay, not an axis.</div>", unsafe_allow_html=True)
    dict(PAGES)[choice]()


if __name__ == "__main__":
    main()
