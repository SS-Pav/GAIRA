"""GAIRA Foundation Model Explorer V3 — The Representation Hierarchy.

An interactive walkthrough of Raman → Ag-SERS transfer organised as a five-level hierarchy of
representations (latent → MSS motif → biochemical theme → perturbation → matrix), with new
rank-preservation and top-k metrics, the purine attractor quantified, and honest null controls
throughout.

    streamlit run gaira_foundation_explorer_v3/app.py

Additive: Foundation Explorer V1 and V2 are untouched and still run. Reads ONLY the committed
representation-hierarchy artifacts + the frozen atlas fingerprint (09ed804a…), verified at load.
Nothing is retrained or modified.
"""
from __future__ import annotations
import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))          # v3_core
sys.path.insert(0, str(REPO / "src"))  # gaira (frozen engine, fingerprint check)

from v3_core import data as D, ui
from v3_core.pages import PAGES

st.set_page_config(page_title="GAIRA Explorer V3 · Representation Hierarchy", page_icon="🪜",
                   layout="wide", initial_sidebar_state="expanded")


def main():
    ui.inject_css()
    with st.sidebar:
        st.markdown(
            '<div style="font-family:Newsreader,serif;font-size:1.3rem;font-weight:600;'
            f'color:{ui.INK};line-height:1.15">GAIRA Explorer V3</div>'
            f'<div style="color:{ui.FAINT};font-size:.82rem;margin:.3rem 0 .9rem">'
            'The Representation Hierarchy · Raman → Ag-SERS transfer</div>',
            unsafe_allow_html=True)
        if not D.present():
            st.error("Representation-hierarchy artifacts not found at "
                     "`results/v5_rebuild/representation_hierarchy_v3/`. Run its `code/` scripts.")
            st.stop()
        choice = st.radio("Contents", [p[0] for p in PAGES], label_visibility="collapsed")
        st.markdown("<hr style='margin:.8rem 0'>", unsafe_allow_html=True)
        fp_ok = D.fingerprint_ok(); repro = D.reproducible_vs_v2()
        fp_txt = "✓ frozen atlas verified" if fp_ok else "⚠ fingerprint mismatch"
        fp_col = ui.OI["green"] if fp_ok else ui.OI["verm"]
        repro_txt = ("✓ V2 reproduced exactly" if repro else "· V2 reproducibility unrecorded")
        st.markdown(
            f"<div style='font-size:.78rem;color:{ui.FAINT};line-height:1.6'>"
            f"<b style='color:{fp_col}'>{fp_txt}</b><br>"
            f"<code style='font-size:.72rem'>{D.CANON_FINGERPRINT[:24]}…</code><br>"
            f"{repro_txt}<br>51 analytes · SERS validates, never trains.</div>",
            unsafe_allow_html=True)

    dict(PAGES)[choice]()


if __name__ == "__main__":
    main()
