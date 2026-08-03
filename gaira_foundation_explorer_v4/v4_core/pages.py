"""The 15 pages of Foundation Explorer V4. Each renders without exception under AppTest.
Static science = audited committed figures; interactivity = plotly + filters."""
from __future__ import annotations
import numpy as np, pandas as pd
import plotly.graph_objects as go
import streamlit as st
from . import data as D
from . import ui
from .ui import OI, INK, LEVEL_COLOR

CFG = {"displayModeBar": False}


def _lay(fig, h=430, **k):
    fig.update_layout(height=h, margin=dict(l=50, r=20, t=40, b=50), plot_bgcolor="white",
                      paper_bgcolor="white", font=dict(color=INK, size=13), **k)
    fig.update_xaxes(gridcolor="#eef1f4", zeroline=False); fig.update_yaxes(gridcolor="#eef1f4", zeroline=False)
    return fig


# 1 Overview
def p01_overview():
    s = D.summary()
    ui.header("GAIRA Foundation Model Explorer V4",
              "Hierarchical biochemical recovery across Raman, Ag-SERS, perturbation and biological matrix",
              "Transfer is not one score — and not one level. Every representation metric is calibrated "
              "against an analyte-mismatched null, so 'recovery' means statistically specific, not a raw "
              "cosine above a threshold.")
    c = {r.level: r for _, r in D.counts().iterrows()}
    ui.stats([(f"{int(c['latent'].n_recovered)}/51", "latent-specific"),
              (f"{int(c['MSS'].n_recovered)}/51", "MSS-specific"),
              (f"{int(c['theme'].n_recovered)}/51", "theme-specific"),
              (f"{int(c['perturbation'].n_recovered)}/51", "perturbation"),
              (f"{int(c['matrix'].n_recovered)}/51", "matrix (serum-tested)")])
    st.markdown("#### The hierarchy")
    st.markdown("Raman reference atlas → NMF coordinates → MSS motifs → biochemical themes → "
                "controlled perturbations → matrix recovery → *(future) dynamic trajectories*.")
    st.table(pd.DataFrame([
        ["Learned", "the 24-component NMF basis (frozen), from pure-Raman spectra"],
        ["Derived", "component registry, ontology, MSS motifs, theme weights, BSV"],
        ["Curated", "family→theme map, recovery rules, null thresholds"],
        ["Validated", "Ag-SERS transfer, perturbation, serum matrix — never used to train"]],
        columns=["role", "what"]))
    ui.warn("<b>Central caution:</b> raw cosines are high (MSS 0.74, theme 0.92) but barely exceed "
            "their mismatched nulls (+0.008, +0.002). A high raw cosine is broad interpretation, not "
            "analyte identity.")
    if D.reproduces_v3():
        ui.good("Every matched value reproduces V3 bit-for-bit (max abs diff 0.0). V4 adds null "
                "calibration and recovery definitions only — no frozen asset changed.")
    ui.figure(D.figure("fig01_representation_hierarchy.png"),
              "The null-calibrated hierarchy: the per-analyte identity signal is tiny at every level.")
    ui.caption(f"Frozen atlas {D.CANON_FINGERPRINT}, verified at load.")


# 2 Foundation Dataset
def p02_dataset():
    ui.header("Foundation Dataset", "Raman trains; SERS validates",
              "The frozen atlas is learned once, from pure-compound Raman spectra only. Every "
              "cross-modal dataset below is projected through the fixed basis — never used to fit it.")
    st.table(pd.DataFrame([
        ["Pure-Raman reference corpus", "Raman", "FOUNDATION (learns the NMF basis)", "375 spectra / 167 analytes"],
        ["Pure Ag-SERS metabolites", "Ag-SERS", "validation (modality gap)", "265 spectra / 53 analytes / 5 reps"],
        ["Unspiked serum on Ag (blank)", "Ag-SERS", "background control", "15 spectra"],
        ["Adenine / ergothioneine dose", "Ag-SERS", "perturbation", "concentration series"],
        ["Uricase depletion", "Ag-SERS", "perturbation (directional)", "serum ± enzyme"],
        ["Serum spike-in", "Ag-SERS", "matrix recoverability", "phase-7"]],
        columns=["dataset", "modality", "role", "size"]))
    ui.note("51 analytes match between the Raman reference and pure Ag-SERS — the basis for the "
            "cross-modal analysis. SERS is <b>validation</b>, never training.")


# 3 Latent Representation
def p03_latent():
    ui.header("Latent Representation", "24 NMF coordinates — strict and substrate-sensitive",
              "GAIRA projects every spectrum onto a frozen non-negative basis of 24 biochemical "
              "components. These coordinates are the finest-grained representation — and the most "
              "sensitive to surface physics.")
    s = D.level_null().iloc[0]
    ui.stats([(s.matched_median, "latent matched (median)"), (s.null_median, "mismatched null"),
              (s.separation, "identity signal"), (f"{int(s.n_recovered)}/51", "recovered")])
    ui.note("A low latent cosine is a <b>surface effect</b>, not a representation failure. The latent "
            "coordinates carry the most analyte identity of any cosine metric — and even so only 7/51 "
            "analytes are uniquely retrievable from their Ag-SERS.")


# 4 How GAIRA interprets a spectrum
def p04_interpret():
    ui.header("How GAIRA Interprets a Spectrum", "The cascade — and why abstraction ≠ identity",
              "A spectrum becomes 24 coordinates → MSS motifs → biochemical themes → a BSV/radar with "
              "confidence and out-of-distribution flags. Crucially, moving up this cascade does NOT make "
              "the representation more analyte-specific.")
    st.markdown("`spectrum → 24 coordinates → 12 MSS motifs → 11 themes → BSV / radar → evidence + caveats`")
    ui.warn("Higher abstraction compresses toward shared biochemical structure, so the raw theme "
            "cosine is high for nearly everything. Analyte identity actually <b>decreases</b> up the "
            "cascade once the shared background is removed.")
    ui.figure(D.figure("fig05_trajectory.png"),
              "Each analyte's trajectory across levels. Raw theme is high for all; latent and identity "
              "residual separate the strong chemisorbers.")


# 5 Cross-modal validation
def p05_crossmodal():
    ui.header("Cross-Modal Validation", "Matched Raman→Ag-SERS vs the null",
              "The full metric hierarchy, each calibrated against an analyte-mismatched null. Where "
              "matched and null distributions overlap, the metric does not carry analyte identity.")
    ui.figure(D.figure("fig04_matched_vs_null.png"),
              "Matched (blue) vs mismatched-null (grey) at every level — heavy overlap everywhere.")
    st.dataframe(D.level_null(), use_container_width=True, hide_index=True)


# 6 MSS motif recovery
def p06_mss():
    ui.header(ui.level_badge(2, "MSS Motif Recovery"), "Does intermediate motif information survive?",
              "MSS motifs sit between exact coordinates and coarse themes. The candidate hypothesis: "
              "they are the primary cross-modal metric. The null test decides.")
    s = D.level_null().iloc[1]
    ui.stats([(s.matched_median, "MSS matched"), (s.null_median, "mismatched null"),
              (s.separation, "identity signal"), (f"{int(s.n_recovered)}/51", "recovered")])
    ui.warn("<b>Hypothesis rejected.</b> MSS cosine (0.74) is mostly shared background — its null "
            "separation (0.008) is <i>smaller</i> than the latent coordinates' (0.024), and its 3 "
            "recovered analytes are a strict subset of the 7 recovered at the latent level. MSS is "
            "<b>not</b> primary; it is supporting.")
    ui.figure(D.figure("fig06_mss_specificity.png"),
              "MSS ranked by null-adjusted specificity (matched − null95), not raw cosine.")
    st.dataframe(D.mss_rank(), use_container_width=True, hide_index=True)


# 7 Biochemical theme interpretation
def p07_theme():
    ui.header(ui.level_badge(3, "Biochemical Theme Interpretation"), "Five theme views, clearly separated",
              "Raw BSV answers 'what broad biochemistry is present?'; the other views probe analyte-specific "
              "identity. Do not read raw cosine as identification.")
    for letter, title, q in [
        ("A", "Raw BSV / radar", "What broad biochemical evidence is present? (high for all — 0.92 median)"),
        ("B", "Identity-specific residual", "What analyte-specific deviation survives? (Raman-centered; 4/51 recovered)"),
        ("C", "Spearman", "Is gross ordering retained? (matched ≈ null → descriptive only)"),
        ("D", "Top-k", "Are major themes retained? (top-3 median 0.67; 28/51 above chance)"),
        ("E", "Argmax", "Does the single dominant label match? (35%, brittle, purine-driven)")]:
        st.markdown(f"**{letter} · {title}** — {q}")
    ui.warn("The raw theme cosine and the argmax are dominated by a common baseline and the Ag purine "
            "attractor. Never present raw cosine as analyte-identification evidence.")
    ui.figure(D.figure("fig07_broad_vs_identity.png"),
              "Raw theme cosine clusters near 0.9 for all analytes; the identity residual separates "
              "only the strong adsorbers.")


# 8 Recoverable analytes — CENTERPIECE
def p08_recoverable():
    ui.header("Recoverable Analytes", "How many analytes retain analyte-specific information?",
              "The centerpiece. Recovery = the analyte's own Ag-SERS is uniquely nearest (rank-1) and "
              "jackknife-stable — never a raw cosine threshold. Independent flags per level.")
    c = {r.level: r for _, r in D.counts().iterrows()}
    ui.stats([(f"{int(c[k].n_recovered)}/{int(c[k].denominator)}", lab) for k, lab in
              [("latent", "latent"), ("MSS", "MSS"), ("theme", "theme"),
               ("perturbation", "perturbation"), ("matrix", "matrix (serum)")]])
    ui.figure(D.figure("fig02_recoverable_by_level.png"),
              "Fractions with 95% CI. Matrix denominator = serum-tested; all others = 51 matched.")
    df = D.evidence()
    st.markdown("#### Filter the 51 analytes")
    col1, col2, col3 = st.columns(3)
    fam = col1.selectbox("Family", ["(all)"] + sorted(df.family.unique()))
    lvl = col2.selectbox("Recovered at level", ["(any)", "latent", "MSS", "theme", "perturbation", "matrix", "none"])
    status = col3.selectbox("Status", ["(all)", "perturbation-tested", "serum-tested"])
    d = df.copy()
    if fam != "(all)": d = d[d.family == fam]
    if lvl == "latent": d = d[d.latent_recovered]
    elif lvl == "MSS": d = d[d.MSS_recovered]
    elif lvl == "theme": d = d[d.theme_recovered]
    elif lvl == "perturbation": d = d[d.perturbation_validated]
    elif lvl == "matrix": d = d[d.matrix_recovered]
    elif lvl == "none": d = d[~(d.latent_recovered | d.MSS_recovered | d.theme_recovered
                                | d.perturbation_validated | d.matrix_recovered)]
    if status == "perturbation-tested": d = d[d.perturbation_status != "not tested"]
    elif status == "serum-tested": d = d[d.serum_tested]
    show = d[["analyte", "family", "latent_recovered", "MSS_recovered", "theme_recovered",
              "perturbation_status", "serum_tier", "matrix_recovered", "evidence_profile"]]
    st.caption(f"{len(show)} analytes")
    st.dataframe(show, use_container_width=True, hide_index=True)
    with st.expander("Per-analyte recovery matrix (all 51)"):
        ui.figure(D.figure("fig03_recovery_matrix.png"),
                  "✓ recovered · blank not · dot not-tested.")
    with st.expander("Overlap & threshold sensitivity"):
        st.markdown("**Overlap matrix** (analytes recovered at both levels):")
        st.dataframe(D.overlap(), use_container_width=True)
        st.markdown("**Null-threshold sensitivity** (rank-1 is percentile-independent; supporting tier varies):")
        st.dataframe(D.threshold(), use_container_width=True, hide_index=True)


# 9 Purine attractor
def p09_purine():
    pb = D.purine_blank().iloc[0]
    ui.header("The Purine Attractor", "Present in the background before any analyte",
              "Ag-SERS collapses many analytes toward a purine-dominated theme. V4 adds the decisive "
              "control: project the unspiked-serum-on-Ag blank.")
    ui.stats([(pb.serum_blank_purine_theme, "blank purine share"),
              (pb.serum_blank_dominant_theme, "blank dominant theme"),
              (int(pb.n_delta_purine_positive), "analytes gaining purine")])
    ui.warn(f"The unspiked-serum-on-Ag <b>blank is already purine-dominant</b> (share "
            f"{pb.serum_blank_purine_theme}) before any analyte is added. The attractor is substantially "
            "a background/substrate phenomenon — <b>not</b> analyte binding alone. No pure Ag-colloid "
            "buffer blank exists in the dataset, so the exact mechanism is not fully isolated.")
    ui.figure(D.figure("fig08_purine_controls.png"),
              "Blank projection, per-analyte Δpurine, and Δpurine vs latent/MSS.")
    st.markdown("**Δpurine correlations:**")
    st.dataframe(D.delta_purine_corr(), use_container_width=True, hide_index=True)


# 10 Perturbation
def p10_perturbation():
    ui.header(ui.level_badge(4, "Perturbation Validation"), "The strongest evidence — for three analytes",
              "A controlled change that moves the correct motif/theme is stronger than any static "
              "similarity. Available for exactly three analytes; every other says 'not tested'.")
    t1, t2, t3 = st.tabs(["Adenine", "Ergothioneine", "Uricase"])
    with t1:
        ui.good("Adenine — dose → nucleic_purine: ρ=0.996, Langmuir K=0.89 µM, R²=0.993. Headline the "
                "purine <b>Δ</b>, not the absolute background composition.")
    with t2:
        ui.good("Ergothioneine — dose → sulfur_antioxidant: ρ=0.927, K=1.52 µM. Monotonic, saturating.")
    with t3:
        ui.good("Uricase (urate) — <b>directional depletion, not a dose</b>: oxopurine-carbonyl motif "
                "Δ=−0.060, purine-ring motif ≈0, broad purine theme diffuse — correct sign at the motif layer.")
    ui.figure(D.figure("fig09_perturbation.png"), "The three validated perturbations.")
    ui.warn("Only 3/51 analytes have any perturbation series. GAIRA never extrapolates a functional "
            "claim to untested analytes.")


# 11 Matrix recoverability
def p11_matrix():
    ui.header(ui.level_badge(5, "Matrix Recoverability"), "Does pure transfer predict serum recovery?",
              "A separate, matrix-dependent property. Tested as a predictor from every pure metric with "
              "effect sizes and confidence intervals.")
    ui.figure(D.figure("fig10_matrix_prediction.png"),
              "Pure-metric predictors of serum displacement (95% CI). Only confidence is significant.")
    ui.warn("No pure transfer metric significantly predicts serum recovery (latent r=0.17, p=0.24). The "
            "only significant correlate is overall confidence (r=0.71) — but this likely reflects "
            "<b>signal strength</b>, not analyte identity. Matrix competition, adsorption bias and "
            "concentration all intervene.")
    st.dataframe(D.matrix_pred(), use_container_width=True, hide_index=True)


# 12 Biological studies
def p12_biological():
    ui.header("Biological Studies", "Real cohorts (unchanged from V1/V3)",
              "The biological validation is carried over unchanged — this pass does not alter biological "
              "results. See Foundation Explorer V1/V3 for the interactive biological pages.")
    ui.note("The cross-modal recovery limits established here explain what to expect biologically: only "
            "strong Ag chemisorbers (oxopurines, thiols) carry analyte-specific signal into a complex "
            "matrix, consistent with serum SERS being dominated by uric acid + hypoxanthine.")


# 13 Limitations
def p13_limitations():
    ui.header("Limitations", "What this analysis cannot claim",
              "Stated plainly, because honesty about limits is the point of the null calibration.")
    for t in ["Raman-trained atlas — no learned modality (Raman→SERS) correction.",
              "The purine attractor's mechanism is not fully isolated (no pure Ag-colloid buffer blank).",
              "Only three analytes have controlled perturbation data.",
              "Recovery depends on null thresholds and the discrete retrieval p-floor (1/51) — BH-FDR is degenerate at N=51.",
              "Matrix recoverability is matrix-dependent and not predicted by pure transfer.",
              "Confidence ≠ analyte identifiability (confidence predicts serum displacement, likely via signal strength).",
              "Incomplete Au-SERS grounding; 5 replicates/analyte limit jackknife resolution."]:
        st.markdown(f"- {t}")


# 14 Future DART
def p14_dart():
    ui.header("Future — DART", "Dynamic perturbation as the highest evidence level",
              "The clearest path beyond static cross-modal similarity is Dynamic Analyte Response "
              "Tracking (DART): perturb the sample and read the trajectory.")
    ui.good("Functional perturbation is already the strongest rung (adenine's dose-response beats every "
            "static cosine). Extending controlled perturbation from 3 analytes to many would convert "
            "'broad-theme only' analytes into functionally-validated ones — the highest evidence GAIRA "
            "can offer, and the recommended next experiment.")


# 15 Methods & provenance
def p15_methods():
    ui.header("Methods & Provenance", "Formulas, roles, fingerprints, sources",
              "The full specification and the source audit.")
    t1, t2, t3 = st.tabs(["Metrics & decision rules", "Full report", "V3 audit"])
    with t1: st.markdown(D.doc("METRICS_AND_DECISION_RULES.md"))
    with t2: st.markdown(D.doc("GAIRA_Hierarchical_Cross_Modal_Validation_V4.md"))
    with t3: st.markdown(D.doc("AUDIT_OF_V3_METRICS.md"))


PAGES = [
    ("1 · Overview", p01_overview),
    ("2 · Foundation Dataset", p02_dataset),
    ("3 · Latent Representation", p03_latent),
    ("4 · How GAIRA Interprets a Spectrum", p04_interpret),
    ("5 · Cross-Modal Validation", p05_crossmodal),
    ("6 · MSS Motif Recovery", p06_mss),
    ("7 · Biochemical Theme Interpretation", p07_theme),
    ("8 · Recoverable Analytes ★", p08_recoverable),
    ("9 · The Purine Attractor", p09_purine),
    ("10 · Perturbation Validation", p10_perturbation),
    ("11 · Matrix Recoverability", p11_matrix),
    ("12 · Biological Studies", p12_biological),
    ("13 · Limitations", p13_limitations),
    ("14 · Future DART", p14_dart),
    ("15 · Methods & Provenance", p15_methods),
]
