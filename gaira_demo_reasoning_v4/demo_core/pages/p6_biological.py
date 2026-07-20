"""Page 6 — Biological Studies. Every cohort, one standardized V6 template. (Scaffold.)"""
from __future__ import annotations
import streamlit as st
from .. import components as C, data as D


def render(bridge):
    s = bridge.platform_stats()
    C.page_header(
        "Application · biological cohorts",
        "Biological studies, one standardized frame",
        "Every biological analysis is rebuilt on the V6 engine and shown with the same template: "
        "group radar, delta radar, confidence, OOD, sample embeddings, MSS, contributing "
        "components, evidence report and domain caveats.")
    C.question("Across EV, serum, diabetes and SHINE cohorts, what biochemical systems separate "
               "the groups — and how much should we trust each separation?")

    present = [c for c in D.BIOLOGICAL_COHORTS if D.cohort_available(c["key"])]
    st.markdown("**Cohorts resolvable on this machine:** "
                + (", ".join(f"`{c['key']}`" for c in present) if present
                   else "none (mount the data volume for cached projections)"))
    st.write("")
    C.scaffold_note([
        "Standardized per-cohort template: overview · sample/spectra counts · reference context.",
        "Group radar + delta radar with confidence and OOD; 2-D sample embeddings.",
        "MSS motif deltas + contributing components + evidence report + domain caveats.",
        "Repeat identically for EV, Serum, Diabetes, SHINE.",
    ])
    C.caveats([
        "Biological cohorts are SERS in complex matrices — out-of-domain for the Raman atlas; "
        "group separations must be read with OOD and confidence, never as absolute quantitation.",
        "Cohort differences reflect biochemical <i>systems</i>, not single-molecule concentrations.",
    ])
    C.provenance_footer(s)
