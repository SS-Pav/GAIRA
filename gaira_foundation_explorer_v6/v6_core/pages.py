"""The pages of Foundation Explorer V6. Narrative: can we SEE it? → IDENTIFY it? → RECOVER its
chemistry? → would a learned TRANSFER model help? Each renders without exception under AppTest."""
from __future__ import annotations
import numpy as np, pandas as pd
import plotly.graph_objects as go
import streamlit as st
from . import data as D
from . import ui
from .ui import OI, INK

CFG = {"displayModeBar": False}
TIERCOL = {"GOOD": OI["green"], "MODERATE": OI["sky"], "POOR": OI["orange"], "UNDETECTABLE": OI["verm"]}


def _lay(fig, h=430, **k):
    fig.update_layout(height=h, margin=dict(l=50, r=20, t=40, b=50), plot_bgcolor="white",
                      paper_bgcolor="white", font=dict(color=INK, size=13), **k)
    fig.update_xaxes(gridcolor="#eef1f4", zeroline=False); fig.update_yaxes(gridcolor="#eef1f4", zeroline=False)
    return fig


def p01_overview():
    s = D.det_summary()
    ui.header("GAIRA Foundation Model Explorer V6",
              "Can we see it? → identify it? → recover its chemistry? → would transfer help?",
              "V6 inserts a Stage-0 DETECTION GATE before any recovery analysis, separating measurement "
              "failure (an analyte is invisible on Ag-SERS) from representation failure (it is measured "
              "but GAIRA cannot recover the chemistry). Only detectable analytes are evaluated further.")
    r = D.restricted()
    ui.stats([(f"{s.get('n_pass','—')}/51", "detectable (pass Stage 0)"),
              (f"{s.get('n_fail','—')}", "undetectable"),
              (f"{r.get('abstraction_improves_after_gate',{}).get('exact_detectable',0):.0%}", "exact identity (detectable)"),
              (f"{r.get('roadmap_groups',{}).get('potentially recoverable (transfer worth trying)','—')}", "transfer-worth-trying")])
    ui.warn("This mirrors real spectroscopy: confirm the analyte gives reproducible signal on the "
            "substrate BEFORE interpreting it. An invisible analyte cannot be 'recovered' by any "
            "representation — that is a substrate problem, not a GAIRA problem.")
    if D.fingerprint_ok():
        ui.good("Reuses V5 recovery flags unchanged; frozen atlas verified; nothing retrained.")
    ui.figure(D.figure("fig01_detection_hierarchy.png"), "Stage 0 gates every later stage.")
    ui.caption(f"Frozen atlas {D.CANON_FINGERPRINT}, verified at load.")


def p02_gate():
    ui.header("The Detection Gate", "A measurement question, not a GAIRA question",
              "Does this Ag-SERS spectrum contain reproducible analyte information above noise and "
              "background? If not, no representation can recover it.")
    ui.note("V5 treated all 51 analytes equally — including analytes whose Ag-SERS is essentially "
            "blank. That conflated two very different failures. V6 separates them: <b>measurement</b> "
            "failure (Stage 0) vs <b>representation</b> failure (Stages 1–3).")
    ui.warn("Thresholds were <b>validated before freezing</b> (see the validation notebook): the "
            "anchors adenine / ergothioneine / urate / xanthine clearly PASS and glucose / tyrosine / "
            "oleate clearly FAIL for physical adsorption reasons — not an arbitrary cutoff.")


def p03_metrics():
    s = D.det_summary()
    ui.header("Detection Metrics", "A transparent, deterministic score — no ML",
              "Detection Confidence (0–1) combines complementary spectroscopic metrics on the 5 "
              "Ag-SERS replicates and the Ag background.")
    st.table(pd.DataFrame([
        ["replicate Pearson", "reproducibility of mean-centred peak structure", "0.45"],
        ["replicate Spearman", "rank reproducibility", "0.10"],
        ["peak SNR", "max local peak signal vs replicate noise (log-scaled)", "0.20"],
        ["variance concentration", "fraction of spectral variance in the top peaks", "0.15"],
        ["reproducible peaks", "# peaks with local SNR > 3", "0.10"]], columns=["metric", "what", "weight"]))
    ui.warn("Replicate <b>cosine</b> was rejected as a metric — it is baseline-inflated (0.93–0.99 for "
            "every analyte), the same shared-background artefact as raw theme cosine. Mean-centred "
            "correlation (Pearson) is the honest reproducibility signal.")


def p04_spectra():
    ui.header("Representative Spectra", "Why analytes pass or fail — visually",
              "Ag-SERS (blue) vs the Ag blank (grey) and their difference (red), with detected peaks. "
              "Strong adsorbers show sharp reproducible peaks; weak adsorbers are blank-like noise.")
    ui.figure(D.figure("fig03_representative_spectra.png"),
              "Top row PASS (sharp reproducible peaks); bottom rows FAIL (structureless, blank-like).")
    ui.figure(D.figure("fig04_blank_overlays.png"),
              "A detectable analyte rises clearly above the Ag background; an undetectable one tracks it.")


def p05_confidence():
    s = D.det_summary()
    ui.header("Detection Confidence", "The distribution and tier boundaries",
              "GOOD ≥ 0.65 · MODERATE ≥ 0.50 · POOR ≥ 0.40 · UNDETECTABLE < 0.40; pass = ≥ 0.50.")
    tc = s.get("tier_counts", {})
    ui.stats([(tc.get("GOOD", 0), "GOOD"), (tc.get("MODERATE", 0), "MODERATE"),
              (tc.get("POOR", 0), "POOR"), (tc.get("UNDETECTABLE", 0), "UNDETECTABLE")])
    ui.figure(D.figure("fig02_detection_distribution.png"), "Detection confidence per analyte.")


def p06_detectable():
    ui.header("Detectable vs Undetectable", "The 22/29 split",
              "Filter the analytes by detection tier and family. Only the detectable set proceeds to "
              "identity / motif / theme evaluation.")
    df = D.detection()
    c1, c2 = st.columns(2)
    tier = c1.selectbox("Tier", ["(all)", "GOOD", "MODERATE", "POOR", "UNDETECTABLE"])
    fam = c2.selectbox("Family", ["(all)"] + sorted(df.broad_family.unique()))
    d = df.copy()
    if tier != "(all)": d = d[d.detection_tier == tier]
    if fam != "(all)": d = d[d.broad_family == fam]
    st.caption(f"{len(d)} analytes")
    st.dataframe(d[["analyte", "broad_family", "rep_pearson", "peak_snr", "var_concentration",
                    "detection_confidence", "detection_tier", "detection_pass"]].sort_values(
        "detection_confidence", ascending=False), use_container_width=True, hide_index=True)


def p07_recovery():
    ui.header("Recovery Hierarchy", "Recovery among DETECTABLE analytes only",
              "The key comparison: does recovery improve once measurement failures are removed?")
    r = D.restricted().get("abstraction_improves_after_gate", {})
    ui.stats([(f"{r.get('exact_all',0):.0%}→{r.get('exact_detectable',0):.0%}", "exact identity (all→detectable)"),
              (f"{r.get('mss_present_all',0):.0%}→{r.get('mss_present_detectable',0):.0%}", "MSS present"),
              (f"{r.get('mss_specific_detectable',0):.0%}", "MSS specific (detectable)"),
              (f"{r.get('theme_specific_detectable',0):.0%}", "theme specific (detectable)")])
    ui.figure(D.figure("fig05_recovery_detectable.png"), "All 51 vs detectable-only.")
    ui.figure(D.figure("fig06_abstraction_gain.png"), "Recovery gain once measurement failure is removed.")
    ui.takehome("Removing measurement failure ~doubles exact identity and lifts presence — but "
                "analyte-SPECIFIC recovery stays low even among the 22 detectable analytes. The residual "
                "failure is genuinely representational, not merely measurement.")


def p08_recoverable():
    ui.header("Recoverable Analytes", "Detection + recovery per analyte",
              "The per-analyte ladder: detection first (green), then the recovery flags. Recovery lives "
              "almost entirely inside the detected block.")
    ui.figure(D.figure("fig09_per_analyte_ladder.png"), "Sorted by detection confidence.")
    st.dataframe(D.transfer()[["analyte", "broad_family", "detection_tier", "detection_pass",
                               "latent_identity_recovered", "mss_present_top3", "theme_present_top3",
                               "matrix_recovered"]], use_container_width=True, hide_index=True)


def p09_transfer():
    ui.header("Transfer-Function Assessment", "Would a learned Raman→SERS model actually help?",
              "The decision that matters for future work: for which analytes is a transfer model "
              "scientifically justified?")
    tc = D.restricted().get("transfer_cases", {})
    st.table(pd.DataFrame([
        ["A · measurement-limited", tc.get("A · measurement-limited", 0), "no transfer model helps; need a better substrate"],
        ["C · already recoverable", tc.get("C · already recoverable", 0), "detectable & identity recovered — transfer unnecessary"],
        ["B · representation-limited (promising)", tc.get("B · representation-limited (promising)", 0), "detectable, chemistry present, identity lost — a transfer model MAY help"],
        ["B · representation-limited (hard)", tc.get("B · representation-limited (hard)", 0), "detectable but no chemistry present — help uncertain"]],
        columns=["case", "n", "meaning"]))
    ui.figure(D.figure("fig07_transfer_decision.png"), "The decision tree.")


def p10_roadmap():
    ui.header("Roadmap", "The concrete target set for future modality correction",
              "Grouping every analyte by how worthwhile a learned Raman→SERS transfer model would be.")
    rg = D.restricted().get("roadmap_groups", {})
    ui.stats([(rg.get("already recoverable", 0), "already recoverable"),
              (rg.get("potentially recoverable (transfer worth trying)", 0), "worth trying"),
              (rg.get("probably impossible (weak signal)", 0) + rg.get("probably impossible (no chemistry present)", 0), "probably impossible"),
              (rg.get("impossible (measurement-limited)", 0), "impossible (measurement)")])
    ui.figure(D.figure("fig08_transfer_roadmap.png"), "Learned-transfer roadmap.")
    lists = D.restricted().get("roadmap_lists", {})
    for g in ["potentially recoverable (transfer worth trying)", "already recoverable"]:
        if g in lists:
            st.markdown(f"**{g}** ({len(lists[g])}): {', '.join(lists[g])}")
    ui.good("The ~11 'potentially recoverable' analytes — detectable with broad chemistry present but "
            "exact identity lost — are the concrete target for a future learned modality-correction model.")


def p11_individual():
    ui.header("Individual Analytes", "Detection + recovery evidence per analyte",
              "Select an analyte to see whether it is detectable, and if so, what GAIRA recovers.")
    det = D.detection().set_index("analyte"); tr = D.transfer().set_index("analyte")
    cards = D.v5_cards()
    names = sorted(det.index)
    a = st.selectbox("Analyte", names, index=(names.index("adenine") if "adenine" in names else 0))
    d = det.loc[a]; t = tr.loc[a]
    st.markdown(f"### {a} · *{d.broad_family}*")
    tier = d.detection_tier
    (ui.good if d.detection_pass else ui.warn)(
        f"<b>Stage 0 — detection: {tier}</b> (confidence {d.detection_confidence:.2f}). "
        + ("Detectable — eligible for recovery evaluation." if d.detection_pass
           else "Undetectable on Ag-SERS — no representation can recover it (measurement failure)."))
    st.markdown(f"**Transfer case:** {t.transfer_case}  \n{t.transfer_note}")
    if d.detection_pass and a in cards:
        c = cards[a]
        st.markdown(f"**Exact identity** — {c['exact_identity']['verdict']}  \n"
                    f"**MSS** — {c['mss_motif']['expected']}: {c['mss_motif']['tier']}  \n"
                    f"**Theme** — {c['biochemical_theme']['expected']}: {c['biochemical_theme']['tier']}  \n"
                    f"**Perturbation** — {c['perturbation']['status']} · **Matrix** — {c['matrix']['tier']}")
    st.caption(f"replicate Pearson {d.rep_pearson:.2f} · peak SNR {d.peak_snr:.1f} · "
               f"variance concentration {d.var_concentration:.2f} · roadmap: {t.roadmap_group}")


def p12_limitations():
    ui.header("Limitations", "What the gate can and cannot claim", "Stated plainly.")
    for t in ["No pure Ag-colloid buffer blank — the serum blank is used as the Ag background reference.",
              "Only 5 replicates per analyte limit reproducibility resolution.",
              "The metric weighting is a transparent, validated choice — not unique.",
              "The gate is conservative: creatinine and thymine were identity-recovered yet fall just below it.",
              "Raman-trained atlas; no learned Raman→SERS modality model yet (this analysis scopes one).",
              "Detection confidence is a measurement quality, not analyte identifiability."]:
        st.markdown(f"- {t}")


def p13_conclusions():
    ui.header("Final Conclusions", "Measurement failure separated from representation failure",
              "The honest bottom line.")
    ui.takehome("A measurement gate before interpretation cleanly separates 'we can't see it' from 'we "
                "can't recover it.' Half the analytes (29/51) are simply invisible on this Ag substrate — "
                "a substrate problem, not a GAIRA problem. Among the 22 that ARE visible, recovery "
                "improves (exact identity ~doubles, presence rises) but analyte-specific chemistry still "
                "largely fails to transfer — a real representation gap. A learned Raman→SERS transfer "
                "model is justified for ~11 detectable, representation-limited analytes; the rest need a "
                "better substrate. Extend dynamic perturbation (DART) — the only route that recovered "
                "class chemistry in V5.")
    with st.expander("Read the full report", expanded=True):
        st.markdown(D.doc("GAIRA_Pure_AgSERS_Evaluation_V6.md"))


PAGES = [
    ("1 · Overview", p01_overview), ("2 · Detection Gate", p02_gate), ("3 · Detection Metrics", p03_metrics),
    ("4 · Representative Spectra", p04_spectra), ("5 · Detection Confidence", p05_confidence),
    ("6 · Detectable vs Undetectable", p06_detectable), ("7 · Recovery Hierarchy", p07_recovery),
    ("8 · Recoverable Analytes", p08_recoverable), ("9 · Transfer Function Assessment", p09_transfer),
    ("10 · Roadmap", p10_roadmap), ("11 · Individual Analytes", p11_individual),
    ("12 · Limitations", p12_limitations), ("13 · Final Conclusions", p13_conclusions),
]
