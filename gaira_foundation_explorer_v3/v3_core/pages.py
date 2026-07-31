"""The 15 pages of Foundation Explorer V3, organised around the Representation Hierarchy.
Each render function is self-contained and must render without exception under AppTest.
Static science = audited committed figures; interactivity (Sankey, scatter, drill-down) = plotly.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import data as D
from . import ui
from .ui import OI, INK, LEVEL_COLOR

CFG = {"displayModeBar": False}


def _layout(fig, h=440, **kw):
    fig.update_layout(height=h, margin=dict(l=50, r=20, t=40, b=50), plot_bgcolor="white",
                      paper_bgcolor="white", font=dict(color=INK, size=13),
                      legend=dict(font=dict(size=11)), **kw)
    fig.update_xaxes(gridcolor="#eef1f4", zeroline=False)
    fig.update_yaxes(gridcolor="#eef1f4", zeroline=False)
    return fig


# ═════════ 1 · Overview ═════════
def p01_overview():
    s = D.summary(); L = s.get("layers", {})
    ui.header("Foundation Explorer V3 · The Representation Hierarchy",
              "Transfer is a hierarchy, not a score",
              "Raman → Ag-SERS transfer is best described not by one number but by how far up a "
              "hierarchy of representations the agreement survives — from latent coordinates, to "
              "motifs, to biochemical themes, to dynamic perturbation, to matrix robustness. V3 "
              "adds rank-preservation and top-k metrics, quantifies the purine attractor, and "
              "keeps every earlier metric for transparency.")
    ui.stats([
        (L.get("L1_latent_fingerprint", {}).get("median", "—"), "L1 latent (median)"),
        (L.get("L2_mss_motif", {}).get("median", "—"), "L2 MSS motif"),
        (L.get("L3a_theme_raw", {}).get("median", "—"), "L3 theme raw"),
        (L.get("L4_theme_rank_rho", {}).get("median", "—"), "L3 rank ρ"),
        (L.get("L5_top3_overlap", {}).get("median", "—"), "L3 top-3"),
        (L.get("L3b_theme_identity", {}).get("median", "—"), "L3 identity"),
    ])
    ui.warn("<b>The core caution:</b> the raw agreement RISES up the hierarchy (0.42 → 0.74 → "
            "0.92), but that top-level agreement is largely a compositional baseline. The honest, "
            "identity-specific signal FALLS at the top (identity 0.11, argmax 35%). Both are true.")
    if D.reproducible_vs_v2():
        ui.good("Every V2 metric is reproduced here bit-for-bit (max abs diff 0.0). V3 is purely "
                "additive interpretation — no frozen asset changed.")
    ui.caption(f"Frozen atlas fingerprint {D.CANON_FINGERPRINT}, verified at load.")


# ═════════ 2 · Representation Hierarchy ★ ═════════
def p02_hierarchy():
    ui.header("Centerpiece · The Representation Hierarchy",
              "Five levels, from surface physics to biochemical meaning",
              "Each level is more abstract and less governed by silver's adsorption physics than "
              "the one above it. The question GAIRA asks is: how far up does agreement survive?")
    hs = D.summary().get("representation_hierarchy", {})
    levels = [
        (1, "Latent fingerprint", "24 NMF coordinates", hs.get("level1_latent_fingerprint", {})),
        (2, "MSS motif", "12 biochemical motifs", hs.get("level2_mss_motif", {})),
        (3, "Biochemical theme", "11 themes · raw→identity→rank→top-k→argmax", hs.get("level3_theme", {})),
        (4, "Perturbation validation", "dynamic dose / directional", hs.get("level4_perturbation", {})),
        (5, "Matrix robustness", "serum competition", hs.get("level5_matrix", {})),
    ]
    for n, name, sub, d in levels:
        st.markdown(ui.level_badge(n, f"<b>{name}</b> — {sub}"), unsafe_allow_html=True)
        if n == 3:
            st.caption(f"raw {d.get('raw',{}).get('median','—')} · identity "
                       f"{d.get('identity_specific',{}).get('median','—')} · rank ρ "
                       f"{d.get('rank_rho',{}).get('median','—')} · top-3 "
                       f"{d.get('top3',{}).get('median','—')} · argmax "
                       f"{d.get('argmax_agreement_rate','—')} · limitation: {d.get('limitation','')}")
        elif n == 4:
            st.caption(f"{d.get('n_validated','—')} analytes ({', '.join(d.get('analytes', []))}) · "
                       f"limitation: {d.get('limitation','')}")
        elif n == 5:
            reg = d.get("regression", {})
            st.caption(f"{d.get('strong','—')} strong · pure→serum r={reg.get('r','—')}, "
                       f"p={reg.get('p_value','—')} (n.s.) · limitation: {d.get('limitation','')}")
        else:
            st.caption(f"median {d.get('median','—')} · std {d.get('std','—')} · "
                       f"range [{d.get('min','—')}, {d.get('max','—')}] · limitation: {d.get('limitation','')}")
    ui.figure(D.figure("fig_h1_representation_hierarchy.png"),
              "The conceptual ladder (left) and per-level distributions across 51 analytes (right). "
              "Raw theme and raw rank cluster high but are baseline-inflated; the identity metric "
              "spreads wide — the honest, selective signal.")
    ui.takehome("Read the hierarchy top-to-bottom: raw agreement rises, identity-specific agreement "
                "falls. GAIRA reports the whole ladder, never a single collapsed 'preservation score'.")


# ═════════ 3 · Layer 1 latent ═════════
def p03_latent():
    s = D.summary().get("layers", {}).get("L1_latent_fingerprint", {})
    ui.header(ui.level_badge(1, "Latent fingerprint preservation"), "The finest-grained view",
              "Cosine between the 24 NMF coordinates. Dominated by adsorption physics — a low "
              "value is a surface effect, not a representation failure.")
    ui.stats([(s.get("median", "—"), "median"), (s.get("mean", "—"), "mean"),
              (s.get("std", "—"), "std"), (f'[{s.get("min","—")}, {s.get("max","—")}]', "range")])
    df = D.metrics()
    fig = go.Figure(go.Histogram(x=df.L1_latent_fingerprint, nbinsx=18, marker_color=LEVEL_COLOR[1]))
    _layout(fig, h=360, xaxis_title="component cosine", yaxis_title="analytes")
    st.plotly_chart(fig, use_container_width=True, config=CFG)
    ui.note("Median 0.42 — partial and adsorption-selective. Strong Ag chemisorbers (oxopurines) "
            "score high; weak physisorbers (pyrimidines, sugars) low. This is surface physics.")


# ═════════ 4 · Layer 2 MSS ═════════
def p04_mss():
    s = D.summary().get("layers", {}).get("L2_mss_motif", {})
    ui.header(ui.level_badge(2, "MSS motif preservation"), "The mid-level layer",
              "Cosine over the 12 biochemical MSS motif activations — between the latent "
              "coordinates and the themes in both abstraction and robustness.")
    ui.stats([(s.get("median", "—"), "median"), (s.get("mean", "—"), "mean"),
              (s.get("std", "—"), "std")])
    df = D.metrics()
    fig = go.Figure(go.Histogram(x=df.L2_mss_motif, nbinsx=18, marker_color=LEVEL_COLOR[2]))
    _layout(fig, h=360, xaxis_title="MSS motif cosine", yaxis_title="analytes")
    st.plotly_chart(fig, use_container_width=True, config=CFG)
    ui.note("Median 0.74 — sits between latent (0.42) and raw theme (0.92). Motif structure is "
            "more robust than exact coordinates but still adsorption-sensitive.")


# ═════════ 5 · Layer 3 theme (raw + identity) ═════════
def p05_theme():
    s = D.summary().get("layers", {})
    ui.header(ui.level_badge(3, "Biochemical theme preservation"),
              "Raw similarity vs identity-specific preservation",
              "The theme level is a ladder of strictness. Start with the two cosines: raw "
              "(baseline-inflated) and identity-specific (baseline-subtracted vs a null).")
    c1, c2 = st.columns(2)
    with c1:
        ui.warn("<b>Raw theme similarity</b> — median "
                f"{s.get('L3a_theme_raw',{}).get('median','—')}. <b>Contains compositional baseline "
                "inflation</b>: unrelated analytes already agree at ≈0.9. Never read alone.")
    with c2:
        ui.good("<b>Identity-specific theme preservation</b> — median "
                f"{s.get('L3b_theme_identity',{}).get('median','—')}. Baseline-subtracted vs a null "
                "over other analytes. The honest, selective signal.")
    df = D.metrics()
    fig = go.Figure()
    fig.add_trace(go.Violin(y=df.L3a_theme_raw, name="raw", line_color=OI["blue"], meanline_visible=True))
    fig.add_trace(go.Violin(y=df.L3b_theme_identity, name="identity-specific", line_color=OI["purple"],
                            meanline_visible=True))
    _layout(fig, h=420, yaxis_title="theme cosine")
    st.plotly_chart(fig, use_container_width=True, config=CFG)
    ui.takehome("Raw theme similarity is a baseline, not a preservation score. Identity-specific "
                "preservation is real but selective — concentrated in the strong Ag chemisorbers.")


# ═════════ 6 · Layer 4 theme rank (Spearman) NEW ═════════
def p06_rank():
    s = D.summary().get("layers", {})
    ui.header(ui.level_badge(3, "Theme RANK preservation (Spearman)") + "  · NEW",
              "Is the ORDERING of biochemical themes preserved?",
              "Instead of argmax or magnitude cosine, compute Spearman ρ between the Raman and "
              "Ag-SERS theme rankings. This uses the full ordering, including minor themes, and is "
              "robust to argmax instability — likely the most representative single view.")
    ui.stats([(s.get("L4_theme_rank_rho", {}).get("median", "—"), "rank ρ (raw)"),
              (s.get("L4_rank_separation", {}).get("median", "—"), "rank separation"),
              (D.summary().get("rank_positive_separation", "—"), "positive separation / 51")])
    ui.warn("<b>Tested, not assumed:</b> raw rank ρ (0.87) is <b>also baseline-inflated</b> — its "
            "null (rank vs OTHER analytes) is ≈0.85, so the identity-specific rank separation is "
            "only +0.01. Rank carries a slim identity edge (positive for 34/51 vs 28/51 for "
            "cosine) — real, but small.")
    ui.figure(D.figure("fig_h4_topk_and_rank_null.png"),
              "Right panel: raw rank ρ ≈ its null; the honest signal is the small positive "
              "separation. Left panel: top-k overlap (next page).")
    tr = D.theme_rank()
    st.markdown("#### Per-analyte rank preservation")
    st.dataframe(tr.sort_values("L4_theme_rank_rho", ascending=False), use_container_width=True,
                 hide_index=True)


# ═════════ 7 · Layer 5 top-k overlap NEW ═════════
def p07_topk():
    s = D.summary().get("layers", {})
    ui.header(ui.level_badge(3, "Top-k theme overlap") + "  · NEW",
              "Do the leading themes stay in the top set?",
              "If purine, protein and organic remain within the top three, that is partial "
              "preservation even if the single argmax flips. Top-k avoids argmax instability and "
              "is less baseline-inflated than cosine/rank because it looks only at the leading themes.")
    ui.stats([(s.get("L5_top2_overlap", {}).get("median", "—"), "top-2 overlap (median)"),
              (s.get("L5_top3_overlap", {}).get("median", "—"), "top-3 overlap (median)")])
    tk = D.topk()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=tk.L5_top3_overlap, name="top-3", marker_color=OI["green"], opacity=0.75))
    fig.add_trace(go.Histogram(x=tk.L5_top2_overlap, name="top-2", marker_color=OI["blue"], opacity=0.6))
    fig.update_layout(barmode="overlay")
    _layout(fig, h=380, xaxis_title="fraction of top themes shared", yaxis_title="analytes")
    st.plotly_chart(fig, use_container_width=True, config=CFG)
    ui.note("Median top-3 = 0.67 → typically 2 of 3 leading themes are retained. This is the "
            "interpretable middle-ground of the theme ladder.")
    st.dataframe(tk.sort_values("L5_top3_overlap", ascending=False), use_container_width=True,
                 hide_index=True)


# ═════════ 8 · Layer 6 argmax ═════════
def p08_argmax():
    df = D.metrics()
    ui.header(ui.level_badge(3, "Dominant-theme agreement (argmax)"),
              "The strict, unstable corner of the theme ladder",
              "Whether the single most-active theme survives. Intentionally the strictest test — "
              "and the least stable, because Ag-SERS collapses almost everything onto purine.")
    n_purine = int((df.sers_dominant == "nucleic_purine").sum())
    ui.stats([(f"{int(df.L6_argmax_agreement.sum())}/{len(df)}", "argmax agreement (35%)"),
              (f"{n_purine}/{len(df)}", "Ag-SERS dominant = purine"),
              (int((df.L6_argmax_agreement & (df.raman_dominant == 'nucleic_purine')).sum()),
               "…already purine in Raman")])
    ui.warn("All 18 agreements are analytes ALREADY purine-dominant in Raman. Argmax 'agreement' "
            "here is mostly the purine attractor, not per-analyte theme survival — which is exactly "
            "why GAIRA reports the softer rank / top-k metrics alongside it.")


# ═════════ 9 · Purine attractor (Sankey + ΔPurine) ═════════
def p09_attractor():
    ui.header("The purine attractor, quantified",
              "How silver pulls the biochemical abstraction toward N-heterocycles",
              "Ag colloid binds N-heterocycles strongly, so oxopurine-like signal dominates the "
              "SERS of weak adsorbers. Here is the flow, and how much purine every analyte gains.")
    # interactive Sankey
    sk = D.sankey_flow()
    nodes = list(dict.fromkeys(sk.raman_dominant.tolist() + sk.sers_dominant.tolist()))
    idx = {n: i for i, n in enumerate(nodes)}
    src = [idx[r] for r in sk.raman_dominant]; tgt = [idx[s] for s in sk.sers_dominant]
    fig = go.Figure(go.Sankey(
        node=dict(label=[f"{n} (Raman)" if n in set(sk.raman_dominant) else n for n in nodes],
                  pad=16, thickness=16,
                  color=[OI["verm"] if n == "nucleic_purine" else OI["sky"] for n in nodes]),
        link=dict(source=src, target=tgt, value=sk.n.tolist(),
                  color="rgba(213,94,0,0.35)")))
    fig.update_layout(height=430, font=dict(size=12), margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True, config=CFG)
    ui.caption("Raman dominant theme (left) → Ag-SERS dominant theme (right). Nearly all flow "
               "converges on nucleic_purine.")
    s = D.summary().get("delta_purine", {})
    ui.stats([(s.get("median", "—"), "median ΔPurine share"),
              (f'{s.get("n_increase","—")}/51', "analytes gaining purine"),
              ("r=−0.38, p=0.006", "ΔPurine vs latent fidelity")])
    c1, c2 = st.columns(2)
    with c1:
        ui.figure(D.figure("fig_h5_delta_purine.png"),
                  "Per-analyte ΔPurine: weak adsorbers gain purine share; purine-rich analytes lose it.")
    with c2:
        ui.figure(D.figure("fig_h6_delta_purine_vs_component.png"),
                  "Significant negative relationship: weaker adsorption fidelity → stronger attractor pull.")
    ui.note("This is adsorption-driven observation bias — a property of the silver surface and a "
            "target for a future observation model, not a defect of the frozen representation.")


# ═════════ 10 · Cross-modal transfer ═════════
def p10_cross_modal():
    ui.header("Cross-modal transfer map", "Latent fidelity vs theme identity, per analyte",
              "Every analyte on latent fingerprint (x) against identity-specific theme preservation "
              "(y). The upper-left — latent redistribution with retained abstraction — is where "
              "adenine lives.")
    df = D.metrics()
    fig = go.Figure()
    for fname, sub in df.groupby("family"):
        fig.add_trace(go.Scatter(
            x=sub.L1_latent_fingerprint, y=sub.L3b_theme_identity, mode="markers", name=fname,
            marker=dict(size=10, line=dict(width=1, color="white")), text=sub.analyte,
            customdata=sub[["L4_theme_rank_rho", "L5_top3_overlap", "delta_purine"]],
            hovertemplate="<b>%{text}</b><br>latent=%{x:.2f}<br>theme identity=%{y:.2f}<br>"
                          "rank ρ=%{customdata[0]:.2f}<br>top-3=%{customdata[1]:.2f}<br>"
                          "ΔPurine=%{customdata[2]:.2f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(dash="dot", color=OI["grey"], width=1))
    fig.add_vline(x=0.55, line=dict(dash="dot", color=OI["grey"], width=1))
    _layout(fig, h=520, xaxis_title="L1 latent fingerprint preservation",
            yaxis_title="L3 identity-specific theme preservation")
    st.plotly_chart(fig, use_container_width=True, config=CFG)
    ui.figure(D.figure("fig_h2_metric_comparison.png"),
              "The same analytes across six metrics: preservation is not one number. Raw theme/rank "
              "cluster high; identity spreads wide.")


# ═════════ 11 · Perturbation ═════════
def p11_perturbation():
    ui.header(ui.level_badge(4, "Perturbation validation"),
              "Dynamic response — the strongest evidence, and the rarest",
              "A controlled change that still moves the correct theme is stronger evidence than any "
              "static similarity. It exists for EXACTLY three analytes; every other analyte is "
              "'not measured', never implied.")
    ui.figure(D.figure("fig_h8_perturbation_summary.png"),
              "The three validated perturbations. Adenine and ergothioneine are dose-responses; "
              "uricase is a directional depletion (not a dose).")
    ui.good("Why dynamic > static: a saturating dose-response (adenine ρ=0.996, K=0.89 µM) shows "
            "the purine abstraction is <b>functional</b> — it tracks concentration along an "
            "adsorption isotherm — not a coincidental static overlap. Even where the latent "
            "fingerprint redistributes, the biochemical response is recovered.")
    ui.warn("Only 3 of 51 analytes have any perturbation series. GAIRA never extrapolates a "
            "dynamic claim to untested analytes.")


# ═════════ 12 · Matrix robustness ═════════
def p12_matrix():
    ui.header(ui.level_badge(5, "Matrix robustness"),
              "Does pure transfer PREDICT serum recoverability?",
              "V2 showed the top oxopurines survive serum. V3 asks the quantitative question across "
              "all 51 analytes — with regression and confidence.")
    reg = D.summary().get("matrix_regression", {}).get("predictor_latent_fingerprint", {})
    ui.stats([(reg.get("r", "—"), "Pearson r"), (reg.get("r2", "—"), "R²"),
              (reg.get("p_value", "—"), "p-value"), (reg.get("spearman_rho", "—"), "Spearman ρ")])
    ui.warn("<b>Honest downgrade:</b> pure Ag transfer is only a WEAK, non-significant predictor of "
            f"serum recovery (r={reg.get('r','—')}, R²={reg.get('r2','—')}, p={reg.get('p_value','—')}). "
            "The top oxopurines survive both, but there is no tight per-analyte law — serum adds "
            "matrix-specific competition beyond pure adsorption strength.")
    ui.figure(D.figure("fig_h7_matrix_regression.png"),
              "Pure transfer vs serum displacement with 95% CI. The slope is shallow and its CI "
              "spans zero — a categorical top-set agreement, not a quantitative law.")


# ═════════ 13 · Per-analyte cards ═════════
def p13_cards():
    ui.header("Per-analyte assessment cards", "Nine layers per analyte",
              "Latent · MSS · theme cosine · rank ρ · top-3 · argmax · family · interpretation · "
              "limitations — in physics-aware language, with 'not measured' where honest.")
    cards = D.cards()
    if not cards:
        st.info("Cards not found — run `code/make_cards_v3.py`."); return
    names = sorted(cards)
    a = st.selectbox("Analyte", names, index=(names.index("adenine") if "adenine" in names else 0))
    c = cards[a]
    L1 = c["layer1_latent_fingerprint"]; L2 = c["layer2_mss_motif"]; L3 = c["layer3_theme_cosine"]
    L4 = c["layer4_theme_rank_correlation"]; L5 = c["layer5_top3_overlap"]; L6 = c["layer6_argmax_agreement"]
    L7 = c["layer7_family"]
    st.markdown(f"### {a} · *{L7['family']}*")
    fig = go.Figure(go.Bar(
        x=["L1 latent", "L2 MSS", "L3 raw", "L3 identity", "L4 rank ρ", "L5 top-3"],
        y=[L1["component_cosine"], L2["mss_cosine"], L3["raw"], L3["identity_specific"],
           L4["spearman_rho"], L5["top3"]],
        marker_color=[LEVEL_COLOR[1], LEVEL_COLOR[2], OI["sky"], OI["purple"], OI["blue"], OI["green"]],
        text=[L1["component_cosine"], L2["mss_cosine"], L3["raw"], L3["identity_specific"],
              L4["spearman_rho"], L5["top3"]], textposition="outside"))
    _layout(fig, h=340, yaxis_title="preservation", yaxis_range=[-1, 1.08])
    st.plotly_chart(fig, use_container_width=True, config=CFG)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**L6 · Argmax** — {L6['raman_dominant']} → {L6['sers_dominant']} "
                    f"({'agree' if L6['agree'] else 'differ'})")
        st.markdown(f"**L7 · Family** — {L7['family']}; ΔPurine {L7['delta_purine']} "
                    f"({L7['purine_share_raman']} → {L7['purine_share_sers']})")
    with col2:
        st.markdown(f"**Rank separation** {L4['rank_separation']} · **identity separation** "
                    f"{L3['identity_separation']}")
    ui.note(f"<b>Interpretation.</b> {c['layer8_interpretation']}")
    st.markdown("**Limitations**")
    for lim in c["layer9_limitations"]:
        st.markdown(f"- {lim}")


# ═════════ 14 · Framework & methods ═════════
def p14_framework():
    ui.header("Framework & methods", "Every metric, its equation, purpose and limitation",
              "The full specification and the interpretation guide.")
    t1, t2, t3 = st.tabs(["Metric specification", "Interpretation guide", "Changelog (V1→V3)"])
    with t1:
        st.markdown(D.doc("HIERARCHY_METRICS_SPECIFICATION.md"))
    with t2:
        st.markdown(D.doc("INTERPRETATION_GUIDE.md"))
    with t3:
        st.markdown(D.doc("CHANGELOG.md"))


# ═════════ 15 · Verdict ═════════
def p15_verdict():
    ui.header("Verdict", "How far up the hierarchy does agreement survive?",
              "The faithful conclusion — optimised for spectroscopic honesty, not a higher score.")
    ui.takehome("Raman → Ag-SERS transfer is a preservation <b>hierarchy</b>, not a score. Latent "
                "coordinates transfer partially (0.42); motifs better (0.74); the broad biochemical "
                "neighbourhood best in raw terms (theme 0.92, rank 0.87) — but that is largely a "
                "compositional baseline. Corrected, identity-specific theme preservation is selective "
                "(identity 0.11, rank separation +0.01, top-3 0.67, argmax 35%), because silver "
                "homogenises most analytes toward a purine attractor (ΔPurine ∝ −adsorption fidelity, "
                "p=0.006). A minority — adenine foremost — redistribute their latent fingerprint yet "
                "retain a dose-responsive theme; that functional validation, though rare, is the "
                "strongest rung. GAIRA separates surface physics from biochemical meaning, and reports "
                "the whole ladder.")
    with st.expander("Read the full Representation Hierarchy narrative", expanded=True):
        st.markdown(D.doc("REPRESENTATION_HIERARCHY.md"))


PAGES = [
    ("1 · Overview", p01_overview),
    ("2 · Representation Hierarchy ★", p02_hierarchy),
    ("3 · L1 · Latent fingerprint", p03_latent),
    ("4 · L2 · MSS motifs", p04_mss),
    ("5 · L3 · Theme (raw + identity)", p05_theme),
    ("6 · L3 · Theme rank ρ (NEW)", p06_rank),
    ("7 · L3 · Top-k overlap (NEW)", p07_topk),
    ("8 · L3 · Argmax agreement", p08_argmax),
    ("9 · The Purine Attractor", p09_attractor),
    ("10 · Cross-Modal Transfer", p10_cross_modal),
    ("11 · L4 · Perturbation", p11_perturbation),
    ("12 · L5 · Matrix Robustness", p12_matrix),
    ("13 · Per-Analyte Cards", p13_cards),
    ("14 · Framework & Methods", p14_framework),
    ("15 · Verdict", p15_verdict),
]
