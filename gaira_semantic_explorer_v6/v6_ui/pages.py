"""The twelve pages of Foundation Explorer V6. Every page teaches, then shows."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from . import data as D, ui


# ── 1 ────────────────────────────────────────────────────────────────────────
def p01_overview():
    h = D.headline()
    ui.header("GAIRA V6", "A rebuilt semantic hierarchy above a frozen foundation",
              "V6 changes nothing about the spectral model. It rebuilds everything above it: "
              "MSS becomes an independent spectroscopy layer, and chemical themes are derived "
              "from MSS rather than from components.")
    ui.stats([(f"{h['n_components']}", "NMF components (frozen)"),
              (f"{h['n_motifs']}", "MSS motifs"),
              (f"{h['n_themes']}", "chemical themes"),
              (h["fingerprint"][:10] + "…", "atlas fingerprint"),
              ("0 bytes", "changed in assets/")])
    ui.rule()
    ui.question("What problem does V6 solve?")
    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.markdown(
            "In V1 the MSS layer was **not independent of the theme layer**. A quarter of every "
            f"component→motif weight was copied from the component→theme matrix — on average "
            f"**{h['leak_mean']:.1%}** of the raw score, and **{h['leak_edges']} of "
            f"{h['leak_total']}** contributor edges existed only because of it.\n\n"
            "That makes a *themes-from-MSS* hierarchy circular. V6 removes the term, rebuilds MSS "
            "from spectroscopy alone, and only then derives themes as groupings of motifs.")
        st.latex(r"\text{theme}(x) \;=\; T^{\mathsf T} M^{\mathsf T}\,\text{coord}(x)")
        st.caption("A composition of two non-negative linear maps. Nothing is learned above the atlas.")
    with c2:
        ui.card("What is frozen", "The NMF basis H, the component registry, preprocessing, the NNLS "
                                  "projection and the fingerprint. V6 reads them; it never writes them.")
        ui.card("What is new", "M (component→motif, spectroscopy only) and T (motif→theme, a hard "
                               "partition chosen by a Pareto study).")
        ui.card("What is deferred", "Biological-state themes. A static Raman spectrum does not carry "
                                    "the functional evidence they need.")
    ui.rule()
    ui.figure(D.fig("f01_hierarchy.png"))
    ui.take(f"V6 result: theme top-1 <b>{h['theme_top1']:.3f}</b> against a permutation null of "
            f"<b>{h['null']:.3f}</b> (κ = {h['kappa']:.3f}), top-3 <b>{h['theme_top3']:.3f}</b>, "
            f"over {h['n_labelled']} labelled Raman analytes. Every theme is a nameable chemical class.")


# ── 2 ────────────────────────────────────────────────────────────────────────
def p02_hierarchy():
    a = D.js("p0_p1_audit.json")
    ui.header("Semantic hierarchy", "Where theme information leaked into MSS",
              "The audit that made the rest of V6 necessary.")
    ui.question("Was the MSS layer ever independent of the themes it is supposed to explain?")
    ui.warn("<b>No.</b> <code>src/gaira/engine/mss.py:196</code> — "
            "<code>raw = 0.40·band + 0.35·exemplar + <b>0.25·theme</b></code>, where "
            "<code>theme = ontology.W[component, parent_theme]</code>.")
    ui.stats([(f"{a['mean_theme_share_of_raw_score']:.1%}", "mean theme share"),
              (f"{a['max_theme_share_of_raw_score']:.1%}", "max theme share"),
              (f"{a['n_edges_that_would_drop_below_keep_threshold']}/{a['n_contributor_edges']}",
               "edges that vanish without it"),
              ("1/3", "V1 breadth, for every motif")])
    ui.figure(D.fig("f02_leakage.png"))
    ui.rule()
    st.markdown("##### Every leakage site")
    st.dataframe(pd.DataFrame([{"code site": k, "statement": v}
                               for k, v in a["leakage_code_sites"].items()]),
                 use_container_width=True, hide_index=True)
    ui.take("A fifth of the V1 MSS graph depended on theme information for its existence. "
            "Deriving themes from that layer would have been circular — so V6 rebuilds MSS first.")


# ── 3 ────────────────────────────────────────────────────────────────────────
def p03_mss_explorer():
    reg = D.mss_registry()
    aud = D.tb("p2_motif_audit.csv")
    ui.header("MSS explorer", "17 motifs, defined by spectroscopy alone",
              "A motif is a recurring Raman band pattern, not a molecule. V6 derives every motif "
              "from four evidence lines, none of which is a theme.")
    ui.question("What evidence defines a motif in V6?")
    w = reg["derivation"]["weights"]
    c = st.columns(4)
    for col, (k, v) in zip(c, w.items()):
        col.metric(k.replace("_", " "), f"{v:.2f}")
    ui.good("<b>theme_evidence_used: false</b> — no theme label, parent theme or ontology weight "
            "enters any V6 MSS quantity.")
    ui.rule()
    ids = [m["id"] for m in reg["motifs"]]
    pick = st.selectbox("Motif", ids, index=0)
    m = next(x for x in reg["motifs"] if x["id"] == pick)
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown(f"#### {m['name']}")
        st.caption(m["description"])
        st.markdown(f"**chemical class** `{m.get('chemical_class','—')}`  \n"
                    f"**bands (cm⁻¹)** {', '.join(str(int(b)) for b in m['bands_cm'])}  \n"
                    f"**exemplars** {', '.join(m['exemplars'][:10])}"
                    + (" …" if len(m["exemplars"]) > 10 else ""))
        ui.stats([(ui.fmt(m["confidence"], 3), "confidence"),
                  (ui.fmt(m["stability"], 3), "stability"),
                  (ui.fmt(m["evidence_breadth"], 3), "breadth"),
                  (ui.fmt(m["spectral_purity"], 3), "purity")])
    with c2:
        st.markdown("**Contributing components**")
        st.dataframe(pd.DataFrame(m["contributors"])[
            ["component", "weight", "band", "basis_cosine", "exemplar", "perturbation"]],
            use_container_width=True, hide_index=True)
    ui.rule()
    ui.figure(D.fig("f09_motif_gallery.png"),
              "Each motif's implied Raman spectrum (red, M[:,m]ᵀH) against its declared band "
              "profile (blue dashed). Agreement means the motif's components really do carry "
              "the bands it claims.")
    ui.rule()
    st.markdown("##### V1 motif audit — what drove the V6 changes")
    st.dataframe(aud[["motif", "corpus_coverage_n", "discriminative_auc", "band_fidelity",
                      "spectral_purity", "confidence", "top_activating_family"]],
                 use_container_width=True, hide_index=True)
    ui.figure(D.fig("f04_motif_audit.png"))


# ── 4 ────────────────────────────────────────────────────────────────────────
def p04_optimisation():
    o = D.js("p4_theme_optimisation.json")
    sw = D.tb("p4_theme_sweep.csv")
    ui.header("Theme optimisation", "How many chemical themes, and chosen how?",
              "Sixteen levels × five generation methods, every one scored against a "
              "label-permutation null.")
    ui.question("Why is raw accuracy the wrong objective?")
    st.markdown(
        "Because it rises **mechanically** as K falls: a two-theme hierarchy is right half the "
        "time by guessing. The sweep therefore reports chance-corrected recoverability")
    st.latex(r"\kappa \;=\; \frac{\text{top-1} - \text{null}}{1 - \text{null}}")
    st.markdown("and an interpretability composite that explicitly penalises trivial coarseness:")
    st.latex(r"I \;=\; 0.4\,C_{\text{chem}} \;+\; 0.3\,C_{\text{spec}} \;+\; 0.3\,\frac{\log K}{\log K_{\max}}")
    st.caption(o["interpretability_definition"])
    ui.figure(D.fig("f05_optimisation.png"))
    ui.rule()
    ui.question("The composite score is flat across the front — so what breaks the tie?")
    ui.warn(f"<b>A pre-stated chemical-admissibility constraint.</b> {o['admissibility_rule']} "
            f"Only <b>{o['n_admissible']} of {o['n_total']}</b> partitions qualify. The raw score "
            f"optimum ({o['raw_score_optimum']['method']} K={o['raw_score_optimum']['K']}) is "
            f"<b>inadmissible</b> — it merges polysaccharide with protein backbone.")
    st.dataframe(pd.DataFrame(o["pareto_front"])[
        ["method", "K", "top1", "null_top1", "kappa", "interpretability",
         "score_kappa_x_interp", "chemically_admissible"]],
        use_container_width=True, hide_index=True)
    ui.take(f"Selected: <b>{o['selected']['method']} at K = {o['selected']['K']}</b> — "
            f"κ {o['selected']['kappa']:.3f}, interpretability {o['selected']['interpretability']:.3f}, "
            "chemically admissible, and the smallest K inside the admissible band.")
    ui.rule()
    st.markdown("##### Full sweep")
    st.dataframe(sw[["method", "K", "top1", "null_top1", "kappa", "macro_f1", "balanced_acc",
                     "ece", "interpretability", "score_kappa_x_interp",
                     "chemically_admissible", "pareto"]],
                 use_container_width=True, hide_index=True, height=340)


# ── 5 ────────────────────────────────────────────────────────────────────────
def p05_performance():
    e = D.js("p7_evaluation.json")
    th = D.tb("p6_theme_reference.csv")
    ui.header("Theme performance", "Every Raman grounding analyte, scored",
              "No hand-picked examples: the full corpus, per theme, with confusion and calibration.")
    ui.stats([(ui.fmt(e["theme_top1"]), "theme top-1"), (ui.fmt(e["theme_top3"]), "theme top-3"),
              (ui.fmt(e["motif_top1"]), "motif top-1"), (ui.fmt(e["motif_top3"]), "motif top-3"),
              (ui.fmt(e["ece"]), "calibration error")])
    ui.figure(D.fig("f06_evaluation.png"))
    ui.rule()
    ui.question("Which themes work, and which do not?")
    st.dataframe(th[["theme", "n_motifs", "n_analytes", "top1", "top3", "median_rank",
                     "mean_confidence", "most_confused_with", "failure_cases"]],
                 use_container_width=True, hide_index=True)
    ui.good("<b>Fixed by V6:</b> sterol 0.00 → 0.58 · sulfur 0.00 → 0.11 · flavin/redox 0.00 → 0.88 · "
            "carotenoid n/a → 1.00. Top-3 rose 0.805 → 0.890.")
    ui.warn("<b>Cost of V6:</b> protein backbone fell 0.45 → 0.09. Free amino acids were split out "
            "into their own motif, and the amide-III / CH₂ region overlaps the saccharide modes — "
            "so the protein motif now loses to Acyl lipid and Monosaccharide.")


# ── 6 ────────────────────────────────────────────────────────────────────────
def p06_comp_to_mss():
    ui.header("Component → MSS", "How 24 latent patterns become 17 motifs",
              "M is a sparse, non-negative map. Each column sums to 1 over at most six components.")
    ui.question("Does removing the theme term damage the motif definitions?")
    ui.figure(D.fig("f03_mss_v1_vs_v6.png"))
    ui.good("Band fidelity improved for <b>12 of 13</b> motifs; stability is unchanged; the "
            "component-weight cosine between V1 and V6 is 0.96. V6 is a purification, not a "
            "replacement — but it is now derivable without any theme input.")
    ui.rule()
    ui.figure(D.fig("f10_maps.png"),
              "Left: M, component → motif. Right: T, motif → theme. Neither takes a theme label as input.")
    st.dataframe(D.tb("p1_mss_v1_vs_v6.csv")[
        ["motif", "component_weight_cosine", "activation_spearman", "components_shared",
         "components_added", "components_dropped", "confidence_v1", "confidence_v6"]],
        use_container_width=True, hide_index=True)


# ── 7 ────────────────────────────────────────────────────────────────────────
def p07_mss_to_theme():
    o = D.js("p4_theme_optimisation.json")
    ui.header("MSS → theme", "Themes are groupings of motifs",
              "T is a hard partition: every motif belongs to exactly one chemical theme.")
    ui.question("What is in each theme, and why?")
    for t in o["selected_partition"]["themes"]:
        with st.expander(f"**{t['name']}**  ·  {len(t['motifs'])} motif(s)"):
            st.markdown("  \n".join(f"- `{m}`" for m in t["motifs"]))
    ui.rule()
    st.markdown("##### The five generation methods compared")
    st.dataframe(pd.DataFrame(o["best_per_method"]), use_container_width=True, hide_index=True)
    st.markdown(
        "- **A manual** — an expert chemical hierarchy, fixed before scoring.\n"
        "- **B activation** — agglomerative on motif co-activation across the corpus.\n"
        "- **C spectral** — agglomerative on the cosine between motifs' implied Raman spectra.\n"
        "- **D ontology** — chemical class plus shared exemplar analytes.\n"
        "- **E hybrid** — the mean of the B/C/D distance matrices. **Selected.**")
    ui.take("The hybrid distance wins because it is the only one that sees all three kinds of "
            "similarity at once — how motifs co-activate, what they look like spectrally, and what "
            "chemistry they name. Its K=13 partition is also chemically admissible, which the "
            "raw-score optimum is not.")


# ── 8 ────────────────────────────────────────────────────────────────────────
def p08_representatives():
    e = D.js("p7_evaluation.json")
    per = D.tb("p7_per_analyte.csv")
    ui.header("Representative analytes", "Excellent, good, moderate, poor, failure",
              "Chosen by rule from the full evaluation, not by hand.")
    st.dataframe(pd.DataFrame(e["representatives"]), use_container_width=True, hide_index=True)
    ui.rule()
    ui.figure(D.fig("f08_pathway.png"))
    ui.rule()
    ui.question("Look up any analyte")
    a = st.selectbox("Analyte", sorted(per.analyte), index=int(np.where(
        np.array(sorted(per.analyte)) == "adenine")[0][0]) if "adenine" in set(per.analyte) else 0)
    r = per[per.analyte == a].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("expected theme", (r.expected_themes or "—").split("|")[0])
    c2.metric("predicted theme", r.predicted_theme)
    c3.metric("rank of expected", int(r.theme_rank))
    st.markdown(f"**family** `{r.family}` · **top component** `c{int(r.top_component)}` "
                f"({r.top_component_share:.3f}) · **predicted motif** `{r.predicted_motif}` "
                f"(rank {int(r.motif_rank)}) · **confidence** {r.theme_confidence:.3f}")
    st.caption(f"Nearest reference analytes in theme space (evidence, never identification): "
               f"{r.nearest_analytes}")


# ── 9 ────────────────────────────────────────────────────────────────────────
def p09_radars():
    ui.header("Radar explorer", "One radar per level of abstraction",
              "The V1 radar showed themes only. V6 shows the whole ladder, so a reader can see "
              "where an interpretation was won or lost.")
    ui.question("Why three radars instead of one?")
    st.markdown(
        "Because they answer different questions. The **component radar** shows what the frozen "
        "atlas actually saw. The **motif radar** shows what spectroscopy says about it. The "
        "**theme radar** shows the chemistry that follows. When a call goes wrong, the level at "
        "which it went wrong is the diagnosis.")
    ui.figure(D.fig("f07_radars.png"))
    ui.take("Read left to right. A clean component radar with a wrong theme radar is an "
            "<b>interpretation</b> failure. A diffuse component radar is a <b>representation</b> "
            "failure. They need different fixes.")


# ── 10 ───────────────────────────────────────────────────────────────────────
def p10_pipeline():
    ui.header("End-to-end pipeline", "Run the frozen V6 stack on a spectrum",
              "The same code path the evaluation used. Nothing is refitted.")
    ui.question("What happens, step by step?")
    st.code("coord  = atlas.coordinates(preprocess(wn, y))   # NNLS, H held fixed\n"
            "motifs = Mᵀ · coord                             # 17 MSS motifs\n"
            "themes = Tᵀ · motifs                            # 13 chemical themes",
            language="text")
    per = D.tb("p7_per_analyte.csv")
    V = D.vectors()
    names = list(V["analytes"])
    a = st.selectbox("Reference analyte", sorted(names), index=sorted(names).index("adenine")
                     if "adenine" in names else 0)
    i = names.index(a)
    import plotly.graph_objects as go
    zA, A, Th = V["zA"][i], V["A_bio"][i], V["Th"][i]
    c1, c2, c3 = st.columns(3)
    with c1:
        f = go.Figure(go.Bar(x=[f"c{j}" for j in range(24)], y=zA, marker_color="#0072B2"))
        f.update_layout(height=260, margin=dict(l=8, r=8, t=30, b=8), title="24 components",
                        yaxis_title="share", showlegend=False)
        st.plotly_chart(f, use_container_width=True)
    with c2:
        f = go.Figure(go.Bar(y=[m.replace("_", " ") for m in V["motif_ids"]], x=A,
                             orientation="h", marker_color="#D55E00"))
        f.update_layout(height=260, margin=dict(l=8, r=8, t=30, b=8), title="17 MSS motifs")
        st.plotly_chart(f, use_container_width=True)
    with c3:
        f = go.Figure(go.Bar(y=list(V["theme_names"]), x=Th, orientation="h",
                             marker_color="#009E73"))
        f.update_layout(height=260, margin=dict(l=8, r=8, t=30, b=8), title="13 chemical themes")
        st.plotly_chart(f, use_container_width=True)
    r = per[per.analyte == a].iloc[0]
    (ui.good if r.theme_rank == 1 else ui.warn)(
        f"Expected <b>{(r.expected_themes or '—').split('|')[0]}</b> · predicted "
        f"<b>{r.predicted_theme}</b> · rank <b>{int(r.theme_rank)}</b> of 13.")


# ── 11 ───────────────────────────────────────────────────────────────────────
def p11_comparison():
    ui.header("Hierarchy comparison", "V1 versus V6, on the same corpus",
              "Both stacks read the identical frozen 24 coordinates.")
    rows = [
        ["MSS independent of themes", "✗ 25 % of each weight is theme", "✓ zero theme input"],
        ["MSS motifs", "13", "17"],
        ["Motif exemplar coverage of the corpus", "35.9 %", "98.8 %"],
        ["MSS confidence", "stability × 1/3 (degenerate)", "0.475 – 0.723"],
        ["Mean band fidelity", "0.594", "0.633"],
        ["Themes derived from", "components (directly)", "MSS motifs"],
        ["Themes", "11 biochemical + 2 non-biochemical", "13 chemical classes"],
        ["Theme top-1", "0.629", "0.613"],
        ["Theme top-3", "0.805", "0.890"],
        ["Sterol theme top-1", "0.00", "0.58"],
        ["Sulfur theme top-1", "0.00", "0.11"],
        ["Flavin / redox theme top-1", "0.00", "0.88"],
        ["Protein backbone top-1", "0.45", "0.09"],
        ["Number of themes justified by", "assertion", "Pareto study over 70 partitions"],
    ]
    st.dataframe(pd.DataFrame(rows, columns=["property", "V1", "V6"]),
                 use_container_width=True, hide_index=True)
    ui.take("V6 trades a little top-1 accuracy for a hierarchy that is <b>non-circular</b>, "
            "covers the corpus, has a calibrated and discriminating confidence, recovers four "
            "chemistries V1 could never reach, and has a justified number of levels. Top-3 rose "
            "from 0.805 to 0.890.")
    ui.warn("The honest cost: protein backbone recovery fell from 0.45 to 0.09. Splitting free "
            "amino acids out of protein is chemically right but leaves the amide backbone motif "
            "competing with saccharide modes in the same spectral region.")


# ── 12 ───────────────────────────────────────────────────────────────────────
def p12_engine():
    h = D.headline()
    reg = D.mss_registry()
    ui.header("Engine summary", "What V6 ships, and what it does not",
              "Provenance, reproduction and limits.")
    ui.stats([(h["fingerprint"][:10] + "…", "atlas fingerprint"),
              ("unchanged", "assets/foundation"),
              (f"{h['n_motifs']}", "motifs"), (f"{h['n_themes']}", "themes"),
              (ui.fmt(h["theme_top1"]), "theme top-1")])
    ui.rule()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Reproduce")
        st.code("python results/v6_rebuild/code/p01_audit_and_mss_rebuild.py\n"
                "python results/v6_rebuild/code/p02_motif_audit.py\n"
                "python results/v6_rebuild/code/p03_theme_optimisation.py\n"
                "python results/v6_rebuild/code/p05_evaluation.py\n"
                "python results/v6_rebuild/code/p06_figures.py", language="bash")
        st.markdown("##### Derivation parameters")
        st.json(reg["derivation"])
    with c2:
        st.markdown("##### Limitations")
        st.markdown(
            "- **Protein backbone regressed** (0.45 → 0.09) — the honest cost of splitting free "
            "amino acids out.\n"
            "- **Sterol still weak** (0.58): no component in the frozen atlas isolates the "
            "steroid ring system, so re-banding the motif can only do so much.\n"
            "- **Porphyrin and flavin remain low-coverage** — the corpus holds no pure reference "
            "for either; both borrow protein components.\n"
            "- **Carotenoid rests on two analytes.**\n"
            "- **Calibration is loose** (ECE 0.28): theme confidence over-states accuracy at the "
            "top of the range.\n"
            "- **In-domain only.** Everything here is pure Raman. No Ag-SERS work is in V6.")
        ui.warn("Biological-state themes are deliberately absent. They require functional evidence "
                "(dose, perturbation, time) that a single static Raman spectrum does not carry.")
    ui.rule()
    st.markdown("##### Next phase — Ag-SERS adaptation (not implemented in V6)")
    st.markdown(
        "The V6 hierarchy is a **Raman** hierarchy. Earlier work showed that on silver the theme "
        "ranking collapses to chance and 95 % of analytes are pulled onto purine. The next phase "
        "should build an **observation model** on top of this frozen hierarchy — a learned "
        "Raman→SERS transfer for the ~11 detectable, representation-limited analytes — rather "
        "than re-fitting any of the layers below it.")


PAGES = [
    ("GAIRA V6 Overview", p01_overview),
    ("Semantic Hierarchy", p02_hierarchy),
    ("MSS Explorer", p03_mss_explorer),
    ("Theme Optimisation", p04_optimisation),
    ("Theme Performance", p05_performance),
    ("Component → MSS", p06_comp_to_mss),
    ("MSS → Theme", p07_mss_to_theme),
    ("Representative Analytes", p08_representatives),
    ("Radar Explorer", p09_radars),
    ("End-to-end Pipeline", p10_pipeline),
    ("Hierarchy Comparison", p11_comparison),
    ("Engine Summary", p12_engine),
]
