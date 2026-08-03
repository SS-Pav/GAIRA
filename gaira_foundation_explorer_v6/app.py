"""GAIRA Foundation Model Explorer V6 — a detection gate before recovery.

    streamlit run gaira_foundation_explorer_v6/app.py

Additive: Explorers V1–V5 are untouched and still run. Reads ONLY committed V6 detection-gate + V5
abstraction artifacts + the frozen atlas fingerprint (09ed804a…), verified at load. Nothing retrained.
"""
from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / "src"))
from v6_core import data as D, ui
from v6_core.pages import PAGES

st.set_page_config(page_title="GAIRA Explorer V6 · Detection Gate", page_icon="🔬",
                   layout="wide", initial_sidebar_state="expanded")


def main():
    ui.inject_css()
    with st.sidebar:
        st.markdown('<div style="font-family:Newsreader,serif;font-size:1.3rem;font-weight:600;'
                    f'color:{ui.INK};line-height:1.15">GAIRA Explorer V6</div>'
                    f'<div style="color:{ui.FAINT};font-size:.82rem;margin:.3rem 0 .9rem">'
                    'Can we see it? → recover it? → would transfer help?</div>', unsafe_allow_html=True)
        if not D.present():
            st.error("V6 artifacts not found at results/v5_rebuild/detection_gate_v6/. Run its code/ scripts.")
            st.stop()
        choice = st.radio("Contents", [p[0] for p in PAGES], label_visibility="collapsed")
        st.markdown("<hr style='margin:.8rem 0'>", unsafe_allow_html=True)
        s = D.det_summary(); fp = D.fingerprint_ok()
        fc = ui.OI["green"] if fp else ui.OI["verm"]
        st.markdown(f"<div style='font-size:.78rem;color:{ui.FAINT};line-height:1.6'>"
                    f"<b style='color:{fc}'>{'✓ frozen atlas verified' if fp else '⚠ fingerprint mismatch'}</b><br>"
                    f"<code style='font-size:.72rem'>{D.CANON_FINGERPRINT[:24]}…</code><br>"
                    f"{s.get('n_pass','—')}/51 detectable · V1–V5 retained<br>reuses V5 recovery flags.</div>",
                    unsafe_allow_html=True)
    dict(PAGES)[choice]()


if __name__ == "__main__":
    main()
