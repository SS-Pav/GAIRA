"""Page 4 — Calibration. The strongest evidence: controlled perturbations. (Scaffold + live.)"""
from __future__ import annotations
import numpy as np
import streamlit as st
from .. import components as C, figures as F, data as D


def render(bridge):
    s = bridge.platform_stats()
    C.page_header(
        "Validation · controlled perturbations",
        "Calibration: does the coordinate system move correctly?",
        "Three controlled studies — adenine dose-response (redistribution), ergothioneine "
        "dose-response (scaling), and uricase depletion (knock-out). If GAIRA is real, a known "
        "biochemical change must move the right motif in the right direction, monotonically.")
    C.question("When we deliberately add or remove a known analyte, does the corresponding MSS "
               "motif and biochemical theme respond monotonically and specifically?")

    cal_key = st.radio("Calibration study", [c.key for c in D.CALIBRATIONS], horizontal=True,
                       format_func=lambda k: D.calibration(k).title)
    cal = D.calibration(cal_key)
    st.markdown(f'<div class="gaira-card">{cal.story}</div>', unsafe_allow_html=True)
    st.write("")

    try:
        Z, meta = D.load_projection(cal.projection)
    except FileNotFoundError:
        C.caveats([f"Cached projection <code>{cal.projection}</code> is not on this machine; "
                   "mount the data volume to see the live dose-response."])
        C.provenance_footer(s); return

    if cal.level_col and cal.level_col in meta.columns:
        levels = np.asarray(meta[cal.level_col], float)
        comp = [bridge.infer(Z[i], domain="buffer").bsv.composition[cal.target_theme]
                for i in range(len(Z))]
        C.figure(F.dose_response(levels, comp, xlabel=f"{cal.analyte} ({cal.level_col})",
                                 ylabel=f"{F.THEME_SHORT.get(cal.target_theme, cal.target_theme)} share",
                                 title=f"{cal.title} — target theme response"),
                 cap=f"Live: each dose projected through the frozen atlas → BSV → "
                     f"{cal.target_theme} composition.",
                 interp="BSV validation found every calibration relation monotonic and saturating "
                        "(all permutation p = 0.002).")
        # MSS view at low vs high dose
        lo, hi = int(levels.argmin()), int(levels.argmax())
        cL, cH = st.columns(2, gap="large")
        for col, idx, tag in [(cL, lo, "lowest dose"), (cH, hi, "highest dose")]:
            with col:
                _, acts = bridge.bsv_and_mss(Z[idx], domain="buffer")
                C.figure(F.mss_hierarchy(acts, title=f"MSS motifs — {tag}"))
    else:
        st.markdown("**Before / after depletion** — difference isolates purine-specific evidence.")
        C.scaffold_note(["Before/after/difference spectra, difference-BSV, and theme waterfall "
                         "for the uricase knock-out."])

    C.takeaways([
        f"{cal.title}: mechanism is <b>{cal.mechanism}</b> (BSV validation Part 5).",
        "Adenine redistributes across components; ergothioneine scales one; uricase depletes purines.",
    ])
    C.scaffold_note([
        "Representative raw spectra + latent-component and MSS evolution across the dose ladder.",
        "Per-dose BSV radar, top components, confidence and OOD trajectory.",
        "Uricase difference-spectrum → difference-components → difference-MSS → theme waterfall.",
    ])
    C.provenance_footer(s)
