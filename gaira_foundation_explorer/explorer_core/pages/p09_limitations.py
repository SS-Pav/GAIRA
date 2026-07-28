"""Page 9 — Limitations."""
from __future__ import annotations
import streamlit as st

from .. import data as D, ui, theme as T


def render():
    corp = D.corpus_summary()
    ui.page_header("Honest limits", "Limitations",
                   "What the model cannot yet do, stated plainly — and the crucial distinction "
                   "between a failure of the representation and a failure of the measurement.")
    ui.question("Where does GAIRA fall short, and whose fault is each shortfall — the atlas, or the "
                "physics of the measurement?")

    ui.rule()
    ui.section("9.1", "The two kinds of failure")
    a, b = st.columns(2)
    with a:
        st.markdown('<div class="card" style="border-left:4px solid #b26a00">'
                    '<h4>Measurement failures — not the atlas</h4>', unsafe_allow_html=True)
        st.markdown(
            "- **Weak-adsorber SERS transfer** (glucose, uracil): the SERS spectrum is reshaped by "
            "adsorption physics.\n"
            "- **39/53 serum spikes not recoverable**: competition, matrix suppression, steric "
            "hindrance.\n\n"
            "In every case the atlas does the *right* thing — it raises OOD and returns no spurious "
            "theme. These are fixed by an **observation model** (Page 10), not a better basis.")
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card" style="border-left:4px solid #2166ac">'
                    '<h4>Representation limits — the atlas to own</h4>', unsafe_allow_html=True)
        st.markdown(
            "- **Coverage gaps**: no isolated porphyrins/heme; sparse flavins, phospholipids, "
            "nucleic-acid polymers → the heme/sterol/redox axes are under-grounded.\n"
            "- **Duplicate labels**: ~6 unmerged molecule groups.\n"
            "- **Provenance gap**: the amino-acid grounding sheet is undocumented.\n"
            "- **Compositional closure**: absolute radars look static under a dominant background.")
        st.markdown("</div>", unsafe_allow_html=True)
    ui.note("take",
            "This separation is the point. A model that confuses the two either over-claims "
            "(blaming physics on the algorithm) or hides real gaps. GAIRA does neither.")

    ui.rule()
    ui.section("9.2", "Corpus gaps, quantified")
    ccl = corp.get("class_counts_analytes", {})
    ui.stat_row([
        (ccl.get("Purine", 5), "purine analytes"),
        (ccl.get("Pyrimidine", 3), "pyrimidine analytes"),
        (ccl.get("Nucleic acid", 3), "DNA/RNA polymers"),
        ("0", "isolated porphyrins"),
        (corp.get("n_cross_source_duplicates"), "cross-source dup labels"),
    ])
    st.markdown(
        "Because the model can only ground a biochemical theme it has seen *pure* examples of, the "
        "thinly-covered chemistries (heme, flavin, phospholipid, nucleic-acid polymers) are exactly "
        "the axes that should carry the least interpretive weight — and the audit's MSS/BSV layers "
        "flag them as provisional. The gaps in the corpus, the weak motifs, and the soft radar axes "
        "**all agree on where the model is soft**, which is itself a consistency check.")

    ui.rule()
    ui.section("9.3", "Raman vs SERS — a representation, not an observer (yet)")
    ui.flow([("Raman", "biochemistry ✓"), ("Frozen state", "coordinate system ✓"),
             ("SERS observation", "not yet modelled")], highlight={2}, arrow="→")
    st.markdown(
        "GAIRA today is a **biochemical representation**. It does not yet model *how a given "
        "surface observes* that biochemistry — the substrate-specific transfer that turns a "
        "biochemical state into an Ag- or Au-SERS spectrum. That observation layer is the single "
        "highest-value next build (Page 10), and the 51 matched Raman↔SERS pairs already on disk "
        "are its empirical seed.")
    ui.note("caveat",
            "None of these limitations is hidden or fatal. Each is named, quantified, and — where it "
            "belongs to the measurement rather than the representation — correctly disowned. The "
            "recommended fixes are all additive and versioned; the frozen atlas is not unfrozen to "
            "make them.")
    ui.report_expander("FINAL_ASSESSMENT.md", "Read the full assessment, incl. what to change (Part 11)")
