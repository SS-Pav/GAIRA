"""The 11 pages of Foundation Explorer V2. Each render function is self-contained and must
render without exception under Streamlit AppTest (verified in tests/). Static science comes
from the audited committed figures; interactivity (hover, per-analyte drill-down) is plotly.
"""
from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import data as D
from . import ui
from .ui import OI, QUADRANT_COLOR, INK, MUTED

PLOTLY_CFG = {"displayModeBar": False}


def _layout(fig, h=440, **kw):
    fig.update_layout(height=h, margin=dict(l=50, r=20, t=40, b=50),
                      plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(color=INK, size=13), legend=dict(font=dict(size=11)), **kw)
    fig.update_xaxes(gridcolor="#eef1f4", zeroline=False)
    fig.update_yaxes(gridcolor="#eef1f4", zeroline=False)
    return fig


# ══════════════════════════ 1 · Overview ══════════════════════════
def p01_overview():
    s = D.summary()
    ui.header("Foundation Explorer V2 · Cross-Modal Transfer",
              "What survives when a spectrum moves Raman → Ag-SERS",
              "The original pure-Ag-SERS stage reports one number — the cosine between the 24 "
              "NMF coordinates — and calls it <i>recoverability</i>. That conflates four "
              "different questions. This explorer separates them into a four-level validation "
              "framework and tests, honestly, how much of the biochemical interpretation "
              "actually transfers to silver.")
    ui.stats([
        (s.get("n_matched", "—"), "matched analytes"),
        (f'{s.get("component_cosine", {}).get("median", "—")}', "median latent fingerprint (L1)"),
        (f'{s.get("theme_cosine_distinct", {}).get("median", "—")}', "median distinctive theme (L2)"),
        (f'{s.get("mss_cosine", {}).get("median", "—")}', "median MSS motif"),
        (f'{s.get("dominant_theme_preserved_rate", 0)*100:.0f}%', "dominant-theme match"),
    ])
    st.markdown("#### The four levels of transfer")
    st.table(pd.DataFrame([
        ["1 · Latent fingerprint preservation", "Do the 24 NMF coordinates line up?",
         "component cosine", f'{s.get("component_cosine",{}).get("median","—")} (median)'],
        ["2 · Biochemical theme preservation", "Does the broad interpretation survive?",
         "distinctive theme cosine · dominant-match",
         f'{s.get("theme_cosine_distinct",{}).get("median","—")} · {s.get("dominant_theme_preserved_rate",0)*100:.0f}%'],
        ["2 · (motif) MSS preservation", "Do mid-level motifs survive?",
         "MSS cosine", f'{s.get("mss_cosine",{}).get("median","—")} (median)'],
        ["3 · Perturbation sensitivity", "Would a controlled change still register?",
         "dose ρ · directional Δ", "measured for 3 analytes only"],
        ["4 · Matrix recoverability", "Does it survive serum competition?",
         "serum spike displacement", "9 strong / 24 mod / 18 weak"],
    ], columns=["Level", "Question", "Metric", "Result"]))
    ui.note("These are <b>not tiers of one quantity</b> — an analyte can be weak at one level "
            "and strong at another. <b>Adenine</b> is weak at Level 1 (component 0.36) yet strong "
            "at Level 3 (dose ρ = 0.996): a single cosine would have called it a failure.")
    ui.caption(f"Frozen atlas fingerprint {D.CANON_FINGERPRINT} · verified at load. "
               "Nothing on any page is retrained; every dataset is projected through the fixed basis.")


# ══════════════════════════ 2 · The metric problem ══════════════════════════
def p02_metric_problem():
    s = D.summary()
    ui.header("The metric problem", "Why one Raman→SERS cosine is too coarse — and how a "
              "high theme cosine can fool you",
              "Raw theme cosine (median 0.92) sits far above component cosine (0.42) for every "
              "analyte. Read naively, that 'proves' the biochemical theme almost always survives. "
              "It does not — and seeing why is the whole point of this explorer.")
    c1, c2 = st.columns(2)
    with c1:
        ui.warn("<b>The trap.</b> Every analyte's 11-theme composition is dominated by the same "
                "few high-share background themes (compositional closure). Two <i>unrelated</i> "
                "analytes already sit at cosine ≈ 0.9 <i>before</i> any preservation. Raw theme "
                "cosine cannot distinguish 'preserved' from 'shared background'.")
    with c2:
        ui.good("<b>The fix.</b> Subtract the shared baseline (mean Raman composition) and measure "
                "each analyte's <i>distinctive</i> deviation against a <b>null</b> — every other "
                "analyte's SERS profile. Now the number means something.")
    tr = s.get("theme_cosine_raw", {}); td = s.get("theme_cosine_distinct", {})
    tn = s.get("theme_null_mean", {}); sep = s.get("theme_separation", {})
    sr = s.get("self_theme_rank", {})
    ui.stats([
        (tr.get("median", "—"), "RAW theme cosine (inflated)"),
        (td.get("median", "—"), "DISTINCTIVE theme cosine"),
        (tn.get("median", "—"), "null floor (other analytes)"),
        (sep.get("median", "—"), "separation (distinct − null)"),
        (f'{sr.get("median","—")} / {sr.get("chance_median","—")}', "self-rank / chance"),
    ])
    ui.note(f"Baseline-corrected, identity-specific theme preservation is <b>weak on average</b>: "
            f"median distinctive cosine {td.get('median','—')}, separation just "
            f"{sep.get('median','—')} (positive for {sep.get('positive_count','—')}/"
            f"{s.get('n_matched','—')}), and the distinctive SERS profile identifies the correct "
            f"analyte in only {s.get('self_is_nearest_theme','—')}/{s.get('n_matched','—')} cases — "
            f"a median self-rank of {sr.get('median','—')} against a chance median of "
            f"{sr.get('chance_median','—')}. Preservation is real but <b>selective</b>.")
    ui.takehome("Rule adopted across GAIRA: <b>never quote raw theme cosine alone.</b> Always with "
                "its null and separation. Raw theme cosine is a baseline, not a preservation score.")


# ══════════════════════════ 3 · Cross-Modal Validation (CENTERPIECE) ══════════════════════════
def p03_cross_modal():
    ui.header("Centerpiece · Cross-Modal Validation",
              "Latent fingerprint vs biochemical theme, honestly",
              "Each analyte placed on latent preservation (x) against theme preservation (y). "
              "Toggle between the naive raw theme cosine and the baseline-subtracted distinctive "
              "cosine to see the illusion resolve.")
    df = D.metrics()
    mode = st.radio("Theme axis", ["Distinctive (honest)", "Raw (baseline-inflated)"],
                    horizontal=True)
    ycol = "theme_cosine_distinct" if mode.startswith("Distinctive") else "theme_cosine"
    fig = go.Figure()
    for q, sub in df.groupby("quadrant"):
        fig.add_trace(go.Scatter(
            x=sub.component_cosine, y=sub[ycol], mode="markers", name=q.split(" ", 1)[1],
            marker=dict(size=11, color=QUADRANT_COLOR.get(q, OI["grey"]),
                        line=dict(width=1, color="white")),
            text=sub.analyte, customdata=sub[["family", "theme_cosine", "theme_cosine_distinct"]],
            hovertemplate="<b>%{text}</b><br>family=%{customdata[0]}<br>"
                          "component=%{x:.3f}<br>theme(raw)=%{customdata[1]:.3f}<br>"
                          "theme(distinct)=%{customdata[2]:.3f}<extra></extra>"))
    fig.add_shape(type="line", x0=-1, y0=-1, x1=1, y1=1, line=dict(dash="dash", color=OI["grey"]))
    fig.add_vline(x=0.55, line=dict(dash="dot", color=OI["grey"], width=1))
    if ycol == "theme_cosine_distinct":
        fig.add_hline(y=0.50, line=dict(dash="dot", color=OI["grey"], width=1))
    _layout(fig, h=520, xaxis_title="Latent fingerprint preservation (component cosine)",
            yaxis_title=f"Theme cosine — {'distinctive' if ycol.endswith('distinct') else 'raw'}")
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)
    q = D.summary().get("quadrants", {})
    ui.stats([(v, k.split(" ", 1)[0]) for k, v in sorted(q.items())])
    ui.note("<b>Q2 — latent redistribution, theme survives</b> is the hypothesis quadrant: the "
            "fingerprint redistributes yet the distinctive theme holds. It is real but a "
            "minority (adenine, riboflavin, phosphate, thymine). Its canonical member, "
            "<b>adenine</b>, is exactly where a dose-response independently confirms a functional theme.")
    with st.expander("The audited static figure (both panels side by side)"):
        ui.figure(D.figure("fig1_component_vs_theme.png"),
                  "Left: raw theme cosine above the diagonal for all 51 — baseline inflation. "
                  "Right: baseline-subtracted, only strong chemisorbers keep an identity-specific theme.")


# ══════════════════════════ 4 · The purine attractor ══════════════════════════
def p04_attractor():
    df = D.metrics()
    ui.header("The purine attractor", "Why the dominant theme 'matches' for only a third — and "
              "what that number really means",
              "Silver colloid has strong affinity for N-heterocycles, so oxopurine-like signal "
              "dominates the SERS of weak adsorbers. The result: nearly every analyte's dominant "
              "theme collapses onto nucleic_purine.")
    n_purine = int((df.sers_dominant == "nucleic_purine").sum())
    ui.stats([(f"{n_purine}/{len(df)}", "Ag-SERS dominant = nucleic_purine"),
              (int(df.dominant_theme_match.sum()), "dominant-theme 'matches'"),
              (int((df.dominant_theme_match & (df.raman_dominant == 'nucleic_purine')).sum()),
               "…that were ALREADY purine in Raman")])
    ui.warn(f"All {int(df.dominant_theme_match.sum())} dominant-theme 'matches' are exactly the "
            "analytes that were <i>already</i> purine-dominant in Raman. The 35% is not 'a third "
            "keep their theme' — it is 'purine-dominant analytes stay purine, and Ag makes almost "
            "everything else look purine too.'")
    c1, c2 = st.columns(2)
    with c1:
        ui.figure(D.figure("fig6_dominant_theme_confusion.png"),
                  "Dominant-theme confusion: nearly all off-diagonal mass is the nucleic_purine column.")
    with c2:
        ui.figure(D.figure("fig3_theme_heatmap.png"),
                  "Raman theme structure is rich and analyte-specific (left); Ag-SERS homogenises "
                  "it toward a purine-dominated profile (right).")
    ui.note("The true theme often survives at rank 2–3 (expected-theme top-3 retained for "
            f"{D.summary().get('expected_theme_top3_retained','—')}/{len(df)}), just not at rank 1. "
            "The attractor is silver physics — an observation-model target, not a defect of the "
            "frozen representation.")


# ══════════════════════════ 5 · Theme redistribution ══════════════════════════
def p05_redistribution():
    ui.header("Theme redistribution", "Where each analyte's composition moves on silver",
              "A high redistribution with a preserved distinctive theme is the hypothesis case; "
              "a high redistribution with a collapsed theme is scrambling. Here is where the "
              "share actually goes.")
    ui.figure(D.figure("fig5_redistribution_waterfalls.png"),
              "Six exemplars — green = theme share gained on silver, red = lost. Adenine and "
              "riboflavin redistribute yet keep purine/sulfur structure; glucose and uracil scramble.")
    df = D.metrics()[["analyte", "family", "gained_theme", "lost_theme", "gained_mss", "lost_mss",
                      "theme_jsd", "l1_theme_shift", "quadrant"]].copy()
    fam = st.selectbox("Filter by family", ["(all)"] + sorted(df.family.unique()))
    if fam != "(all)":
        df = df[df.family == fam]
    st.dataframe(df.sort_values("theme_jsd", ascending=False), use_container_width=True,
                 hide_index=True)


# ══════════════════════════ 6 · MSS motif preservation ══════════════════════════
def p06_mss():
    s = D.summary()
    ui.header("MSS motif preservation", "The layer between coordinates and themes",
              "Between the 24 latent coordinates and the 11 themes sits the MSS motif layer. "
              "Motif preservation is intermediate — more robust than exact coordinates, still "
              "adsorption-sensitive.")
    ui.stats([(s.get("component_cosine", {}).get("median", "—"), "latent (median)"),
              (s.get("mss_cosine", {}).get("median", "—"), "MSS motif (median)"),
              (s.get("theme_cosine_raw", {}).get("median", "—"), "theme raw (median)")])
    ui.figure(D.figure("fig4_family_comparison.png"),
              "Preservation rises latent → motif → theme for almost every family; the gap is "
              "largest for weak adsorbers (amino acids, sugars, lipids).")
    st.markdown("#### Per-analyte MSS preservation")
    st.dataframe(D.mss().sort_values("mss_cosine", ascending=False),
                 use_container_width=True, hide_index=True)


# ══════════════════════════ 7 · Perturbation validation ══════════════════════════
def p07_perturbation():
    ui.header("Level 3 · Perturbation validation", "Does a controlled change still move the "
              "correct theme?",
              "The operationally strongest form of preservation — but measurable only where a "
              "perturbation series exists. In GAIRA that is exactly three analytes; every other "
              "analyte is <b>Not tested</b>, never imputed.")
    ui.figure(D.figure("fig8_perturbation_sensitivity.png"),
              "Adenine and ergothioneine: monotonic, saturating dose-responses. Uricase: "
              "directional urate depletion — the oxopurine motif drops, the theme layer is diffuse.")
    p = D.perturbation()
    st.markdown("#### The measured perturbations")
    st.dataframe(p, use_container_width=True, hide_index=True)
    ui.good("<b>Adenine</b> (Q2, weak latent 0.36) drives the purine theme monotonically "
            "(ρ = 0.996) along a saturating Langmuir law (K = 0.89 µM) — the theme is not merely "
            "present but <b>functional</b>. This is the strongest evidence any static cosine can't give.")
    ui.warn("<b>Uricase is directional, not a dose series.</b> It validates perturbation "
            "<i>direction and localisation</i> at the oxopurine motif (Δ = −0.060), not a dose "
            "magnitude. It is never scored as a concentration response.")


# ══════════════════════════ 8 · Matrix recoverability ══════════════════════════
def p08_matrix():
    ui.header("Level 4 · Matrix recoverability", "Which analytes survive real serum competition",
              "Serum adds competition on top of the modality gap. The strong serum recoverers are "
              "the same strong Ag chemisorbers that pass Levels 1–3 — the dividing line is "
              "adsorption, end to end.")
    ui.figure(D.figure("fig9_matrix_recoverability.png"),
              "Serum recovery vs pure-transfer latent preservation. Oxopurines + adenine + "
              "ergothioneine cluster as the strong recoverers.")
    m = D.matrix()
    counts = m[m.serum_tested == True].serum_recovery_tier.value_counts()
    ui.stats([(int(counts.get("strong", 0)), "serum strong"),
              (int(counts.get("moderate", 0)), "serum moderate"),
              (int(counts.get("weak", 0)), "serum weak")])
    show = m[["analyte", "family", "serum_recovery_tier", "serum_spike_displacement",
              "serum_replicate_direction_cos", "component_cosine", "theme_cosine_distinct",
              "quadrant"]]
    st.dataframe(show.sort_values("serum_spike_displacement", ascending=False),
                 use_container_width=True, hide_index=True)
    ui.note("Serum data validate <b>matrix</b> recoverability, not pure-analyte theme "
            "preservation — the two are kept separate throughout the framework.")


# ══════════════════════════ 9 · Per-analyte transfer cards ══════════════════════════
def p09_cards():
    ui.header("Per-analyte transfer cards", "Drill into any of the 51 analytes across all four levels",
              "Every card carries Levels 1–4. Where a perturbation or serum measurement does not "
              "exist, the card says <b>Not tested</b> — never an imputed number.")
    cards = D.cards()
    if not cards:
        st.info("Cards not found — run `code/make_cards_and_layers.py`."); return
    names = sorted(cards)
    default = names.index("adenine") if "adenine" in names else 0
    a = st.selectbox("Analyte", names, index=default)
    c = cards[a]
    L1, L2 = c["level1_latent_fingerprint"], c["level2_biochemical_theme"]
    L3, L4 = c["level3_perturbation_sensitivity"], c["level4_matrix_recoverability"]
    st.markdown(f"### {a}  ·  *{c['family']}* — {c['quadrant']}")
    # 4-cosine bar
    fig = go.Figure(go.Bar(
        x=["L1 component", "MSS motif", "theme raw", "theme distinct"],
        y=[L1["component_cosine"], L2["mss_cosine"], L2["theme_cosine_raw"], L2["theme_cosine_distinct"]],
        marker_color=[OI["verm"], OI["orange"], OI["sky"], OI["blue"]],
        text=[L1["component_cosine"], L2["mss_cosine"], L2["theme_cosine_raw"], L2["theme_cosine_distinct"]],
        textposition="outside"))
    _layout(fig, h=330, yaxis_title="cosine", yaxis_range=[-1, 1.05])
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**L1 · Latent fingerprint** — {L1['component_cosine']} ({L1['tier']})")
        st.markdown(f"**L2 · Theme** — dominant {L2['raman_dominant']} → {L2['sers_dominant']} "
                    f"({'preserved' if L2['dominant_theme_match'] else 'not preserved'})  \n"
                    f"distinctive {L2['theme_cosine_distinct']} · null {L2['theme_null_mean']} · "
                    f"separation {L2['theme_separation']}  \n"
                    f"expected theme rank {L2['expected_rank_raman']} → {L2['expected_rank_sers']} "
                    f"(top-3 retained: {L2['expected_retained_top3']})  \n"
                    f"redistribution: gained **{L2['redistribution']['gained_theme']}**, "
                    f"lost **{L2['redistribution']['lost_theme']}**")
    with col2:
        if L3.get("tested") is False or L3.get("status") == "Not tested":
            ui.warn(f"<b>L3 · Perturbation</b> — Not tested ({L3.get('why','no data')})")
        else:
            ui.good(f"<b>L3 · Perturbation</b> — {L3.get('perturbation_kind')} on "
                    f"{L3.get('target_theme')}: {L3.get('statement')}")
        if not L4.get("serum_tested"):
            ui.warn(f"<b>L4 · Matrix</b> — Not tested ({L4.get('why','not in serum panel')})")
        else:
            ui.note(f"<b>L4 · Matrix (serum)</b> — tier <b>{L4['serum_recovery_tier']}</b> "
                    f"(displacement {L4['serum_spike_displacement']}, "
                    f"direction cos {L4['serum_replicate_direction_cos']})")
    ui.caption(f"OOD(SERS) {c['ood_sers']} · confidence(SERS) {c['confidence_sers']} · "
               "frozen atlas 09ed804a…")


# ══════════════════════════ 10 · Framework & methods ══════════════════════════
def p10_framework():
    ui.header("Framework & methods", "The four-level validation framework, in full",
              "The complete methodology — metric definitions, null controls, and provenance.")
    tab1, tab2 = st.tabs(["Multi-level framework", "Metric specification"])
    with tab1:
        st.markdown(D.framework_doc())
    with tab2:
        st.markdown(D.doc("METRICS_SPECIFICATION.md"))


# ══════════════════════════ 11 · Verdict ══════════════════════════
def p11_verdict():
    ui.header("Verdict", "Does the biochemical theme survive when the fingerprint does not?",
              "The honest assessment, stated plainly.")
    ui.takehome("<b>Partially supported — the framing is right, the strong universal reading is "
                "not.</b> Theme and fingerprint preservation are genuinely distinct metrics (theme "
                "cosine exceeds component cosine for all 51 analytes), and GAIRA should measure "
                "both. But 'theme cosine 0.92, so theme always survives' is a compositional-"
                "baseline illusion: baseline-corrected, identity-specific preservation is weak and "
                "selective, concentrated in the strong silver adsorbers, because Ag-SERS homogenises "
                "nearly everything toward a purine attractor. A real minority — adenine foremost — "
                "does redistribute its latent profile while keeping a <b>dose-responsive</b> theme.")
    with st.expander("Read the full assessment", expanded=True):
        st.markdown(D.doc("THEME_PRESERVATION_ASSESSMENT.md"))


PAGES = [
    ("1 · Overview", p01_overview),
    ("2 · The Metric Problem", p02_metric_problem),
    ("3 · Cross-Modal Validation ★", p03_cross_modal),
    ("4 · The Purine Attractor", p04_attractor),
    ("5 · Theme Redistribution", p05_redistribution),
    ("6 · MSS Motif Preservation", p06_mss),
    ("7 · Perturbation Validation", p07_perturbation),
    ("8 · Matrix Recoverability", p08_matrix),
    ("9 · Per-Analyte Cards", p09_cards),
    ("10 · Framework & Methods", p10_framework),
    ("11 · Verdict", p11_verdict),
]
