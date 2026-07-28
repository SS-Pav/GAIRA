"""Page 2 — The Grounding Corpus."""
from __future__ import annotations
import pandas as pd
import streamlit as st

from .. import data as D, ui, charts as C, theme as T

# citation/instrument metadata for the 3 training sources (from GROUNDING_AUDIT.md)
SRC_META = {
    "RamanBioLib": dict(
        cite="Terán et al., Chemom. Intell. Lab. Syst. 264 (2025) 105476 · ODbL",
        instr="digitized reference library, 9 excitations", quality="Rich"),
    "gobbato_raman_metabolites": dict(
        cite="Gobbato et al. 2025, Anal. Bioanal. Chem. · DOI 10.1007/s00216-025-06192-5",
        instr="B&WTek i-Raman Plus, 785 nm powders", quality="Sparse (filename-encoded)"),
    "amino_acid_raman_grounding": dict(
        cite="origin undocumented in repo — provenance gap",
        instr="785 nm (hard-coded); pure amino-acid + metabolite sheet", quality="None"),
}
VALIDATION_SETS = [
    ("Pure Gobbato SERS (265)", "Ag-SERS 785 nm", "Raman→SERS transfer test"),
    ("Adenine dose series", "Ag-SERS, European ILS (15 labs)", "dose-response of purine theme"),
    ("Ergothioneine dose series", "Ag-SERS, Fornasaro/Zenodo 13785349", "dose-response of sulfur theme"),
    ("Serum spike-in (53)", "Ag-SERS serum, Gobbato", "recoverability boundary"),
    ("Uricase depletion", "Ag-SERS serum ± uricase, Gobbato", "purine-specific subtraction"),
    ("COVID serum Raman (~477)", "biological serum Raman", "out-of-domain projection / OOD"),
]


def render():
    corp = D.corpus_summary()
    ui.page_header("The knowledge base", "The Grounding Corpus",
                   "The complete biochemical knowledge used to build the atlas — and the firewall "
                   "that keeps SERS out of it.")
    ui.question("What knowledge was used to build the model — and what was deliberately withheld?")

    src_sp = corp.get("sources_spectra", {})
    src_an = corp.get("sources_analytes", {})
    ui.stat_row([
        (corp.get("n_spectra"), "total spectra"),
        (corp.get("n_analytes"), "analyte labels"),
        (len(src_sp), "training sources"),
        (f'{corp.get("window_cm",["",""])[0]:.0f}–{corp.get("window_cm",["",""])[1]:.0f}', "cm⁻¹ window"),
        (corp.get("n_bins"), "spectral bins"),
    ])

    ui.rule()
    ui.section("2.1", "Training sources — pure Raman only")
    rows = []
    for s, n in src_sp.items():
        meta = SRC_META.get(s, {})
        rows.append({"Source": s, "Spectra": n, "Analytes": src_an.get(s, "—"),
                     "Instrument": meta.get("instr", "—"), "Citation": meta.get("cite", "—"),
                     "Metadata": meta.get("quality", "—")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    ui.pills([("TRAINING — feeds the frozen NMF", "train")])
    ui.note("caveat",
            "Provenance note surfaced by the audit: the <b>amino-acid grounding sheet</b> has no "
            "citation or instrument record, and the peak-assignment \"knowledge core\" used to "
            "corroborate axis names is <b>self-authored</b>, not external literature. Neither affects "
            "the representation, but both are honest gaps.")

    ui.rule()
    ui.section("2.2", "Validation sources — SERS & biological, never used to fit")
    vdf = pd.DataFrame(VALIDATION_SETS, columns=["Dataset", "Modality / origin", "Purpose in the audit"])
    st.dataframe(vdf, hide_index=True, width="stretch")
    ui.pills([("VALIDATION — projected through the atlas, never trains it", "val")])

    ui.rule()
    ui.section("2.3", "Why SERS is intentionally excluded from representation learning")
    a, b = st.columns([1.05, 1])
    with a:
        st.markdown(
            "SERS intensity depends on how a molecule **adsorbs to a metal surface** — its "
            "orientation, affinity, and the substrate's enhancement profile. Those are properties of "
            "the *measurement*, not of the biochemistry. If SERS spectra were used to fit the basis, "
            "the coordinate system would be **biased toward one silver or gold surface** and would no "
            "longer be a neutral biochemical reference.")
        st.markdown(
            "So GAIRA fits the atlas on **pure spontaneous Raman** (the intrinsic molecular "
            "fingerprint) and treats every SERS dataset as an *external test*. This is what makes the "
            "later Raman→SERS transfer result (Page 7) meaningful rather than circular.")
    with b:
        ui.flow([("Pure Raman", "biochemistry"), ("Frozen atlas", "neutral frame"),
                 ("SERS = test", "surface physics")], highlight={1}, arrow="→")
        ui.note("info", "The atlas loader hard-asserts <code>modality == 'raman'</code>; SERS cannot "
                        "leak into the fit.")

    ui.rule()
    ui.section("2.4", "Corpus composition")
    ui.figure_card(
        "per_source.png",
        question="How much does each source contribute, in spectra and unique analytes?",
        method="Count spectra and distinct canonical analytes per source from the frozen corpus.",
        result="RamanBioLib supplies breadth (202 sp / 141 analytes); Gobbato supplies the clinical "
               "metabolite core (153 / 51); the amino-acid sheet reinforces the backbone (20 / 19).",
        interpretation="Three complementary sources: general chemistry + serum metabolites + amino "
                       "acids.",
        takehome_text="No single library is sufficient; the union covers general, clinical and "
                      "amino-acid chemistry.")
    c1, c2 = st.columns(2)
    with c1:
        ui.figure_card(
            "class_balance_analytes.png",
            question="What chemistry does the corpus actually span?",
            method="Assign each analyte a chemical class by molecular identity (rule-based).",
            result="Protein-, saccharide-, amino-acid- and lipid-rich; a solid purine/pyrimidine core.",
            interpretation="Breadth across the major biochemical families of serum and tissue.",
            takehome_text="The basis can only ground a theme it has seen pure examples of — this map "
                          "is that ground truth.")
    with c2:
        ui.figure_card(
            "spectra_per_analyte.png",
            question="How deep is the coverage per analyte?",
            method="Histogram of spectra per unique analyte.",
            result=f"Median {corp.get('spectra_per_analyte',{}).get('median','2')}, "
                   f"{corp.get('spectra_per_analyte',{}).get('n_singletons','80')} singletons "
                   f"of {corp.get('n_analytes')} analytes.",
            interpretation="Broad but shallow — about half the analytes have a single spectrum.",
            takehome_text="Replicate-based stability metrics rest on the ~half of analytes that have "
                          "repeats.")

    with st.expander("Excitation distribution + cross-source replication (detail)"):
        d1, d2 = st.columns(2)
        with d1:
            ui.figure_card("excitation_distribution.png",
                           caption="785 nm dominates; excitation is tracked as a nuisance factor and "
                                   "its leakage into the latent space is near zero (Page 4).")
        with d2:
            dup = D.cross_source_duplicates()
            st.markdown(f"**{len(dup)} analytes appear in ≥2 sources** — genuine replication that "
                        "strengthens the within-analyte stability signal.")
            if len(dup):
                st.dataframe(dup[["analyte", "report_class", "n_spectra"]].head(18),
                             hide_index=True, width="stretch", height=300)

    ui.rule()
    ui.section("2.5", "Coverage gaps & data quality (honest)")
    g1, g2 = st.columns(2)
    with g1:
        ui.card("Coverage gaps",
                "- Nucleic acids thin (3 DNA/RNA + 5 purines + 3 pyrimidines)\n"
                "- **No isolated porphyrins / heme** → the heme axis is under-grounded\n"
                "- Flavins/vitamins folded into a small 'cofactor' bucket\n"
                "- Phospholipids sparse (2) — a limit for EV/membrane work")
    with g2:
        ndup = corp.get("n_cross_source_duplicates")
        ui.card("Data-quality finding",
                f"The audit found **6 unmerged duplicate molecule groups** "
                f"(`alb`/`albumin`, `gluth`/`glutathione`, `ure`/`urea`, a riboflavin ligature, …) "
                f"→ the {corp.get('n_analytes')} labels correspond to **≈161 distinct molecules**. "
                f"Impact on the representation is negligible; it is flagged for a future rebuild.")
    ui.note("take",
            "This is the complete biochemical knowledge used to build the atlas: three pure-Raman "
            "sources, SERS firewalled out, gaps and duplicate-label debt named rather than hidden.")
    ui.report_expander("GROUNDING_AUDIT.md", "Read the full grounding audit (Part 1)")
    ui.report_expander("FOUNDATION_CORPUS_REPORT.md", "Read the full corpus report (Part 2)")
