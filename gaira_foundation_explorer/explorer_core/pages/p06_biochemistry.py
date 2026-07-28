"""Page 6 — From Components to Biochemistry."""
from __future__ import annotations
import streamlit as st

from .. import data as D, ui, charts as C, theme as T


def render():
    reg = D.mss_registry()
    motifs = reg.get("motifs", [])
    ui.page_header("Mathematics → chemistry", "From Components to Biochemistry",
                   "How 24 latent numbers become an eleven-axis biochemical reading with confidence "
                   "— a semantic state, not another opaque embedding.")
    ui.question("How does mathematics become chemistry — and why is the output a biochemical state "
                "rather than just another latent vector?")

    ui.rule()
    ui.section("6.1", "The interpretive stack")
    ui.flow([("Spectrum", "676 bins"), ("Components", "24 coords"), ("MSS motifs", "13 patterns"),
             ("Themes", "11 biochemical"), ("BSV + radar", "state + confidence")],
            highlight={2, 3})
    st.markdown(
        "Two layers sit between the latent coordinates and the reading. Each is **deterministic and "
        "documented** — nothing here is learned or fit:")
    c1, c2 = st.columns(2)
    with c1:
        ui.card("MSS — Molecular Spectral Signatures",
                "Curated spectral **motifs** (e.g. *oxopurine carbonyl*, *lipid acyl chain*). Each "
                "motif's definition (bands + exemplar chemistries + parent theme) is curated from "
                "Raman spectroscopy; **which components express it** is *derived* from the frozen "
                "atlas. MSS answers: *which spectral patterns explain this state?*")
    with c2:
        ui.card("Ontology — component → theme weights",
                "A frozen 24×13 weight matrix **W** (each component's row sums to 1) maps components "
                "onto **11 biochemical themes** + 2 non-biochemical (background, unknown). "
                "Many-to-many: a component feeds several themes; a theme draws from several "
                "components.")

    ui.rule()
    ui.section("6.2", "The equations (nothing learned here)")
    st.markdown("From a query's non-negative activations $a\\in\\mathbb{R}^{24}$:")
    st.latex(r"\text{coord}_j = \frac{a_j}{\sum_k a_k}\quad\text{(evidence share)}\qquad "
             r"z_j = \frac{\text{coord}_j - \text{center}_j}{\text{spread}_j}\quad\text{(vs pure refs)}")
    st.latex(r"\underbrace{\text{composition}_t = \sum_j W_{jt}\,\text{coord}_j}_{\text{theme share (}\ge0,\ \Sigma\approx1\text{)}}"
             r"\qquad \underbrace{\text{elevation}_t = \sum_j W_{jt}\,z_j}_{\text{how elevated vs refs}}")
    st.latex(r"\text{confidence}_t = \text{stability}_t \cdot \text{evidence}_t \cdot (1-\text{OOD})")
    ui.note("info",
            "The two frozen matrices — component→theme <b>W</b> and the reference frame "
            "(center/spread) — are the whole transform. Given the same coordinates they always "
            "produce the same biochemical state.")

    ui.rule()
    ui.section("6.3", "Explore the Molecular Spectral Signatures")
    st.markdown(f"The {len(motifs)} derived motifs. Select one to see which components express it, "
                "its characteristic bands, exemplar chemistries, parent theme, and the perturbation "
                "evidence that links it back to the calibration experiments (Page 7).")
    if motifs:
        names = {m["id"]: f'{m["name"]} → {m["parent_theme"]}' for m in motifs}
        mid = st.selectbox("Motif", list(names), format_func=lambda k: names[k])
        m = next(x for x in motifs if x["id"] == mid)
        left, right = st.columns([1, 1])
        with left:
            contribs = m.get("contributors", [])
            if contribs:
                labels = [f"c{c['component']}" for c in contribs]
                vals = [c["weight"] for c in contribs]
                st.plotly_chart(C.hbar(labels, vals, xtitle="component weight in motif",
                                       height=max(200, 40 * len(labels)), valfmt=".2f"),
                                width="stretch", config={"displayModeBar": False})
        with right:
            st.markdown(f"**{m['name']}**")
            st.markdown(m["description"].strip())
            st.markdown(f"- **Parent theme:** `{m['parent_theme']}`\n"
                        f"- **Characteristic bands:** {', '.join(str(b) for b in m['bands_cm'])} cm⁻¹\n"
                        f"- **Exemplars:** {', '.join(m['exemplars'][:6])}\n"
                        f"- **Confidence:** {ui.fmt(m['confidence'],2)} "
                        f"(stability {ui.fmt(m['stability'],2)} × breadth {ui.fmt(m['evidence_breadth'],2)})")
            pert = m.get("perturbation", {})
            nd = len(pert.get("dose_responsive_components", []))
            ns = len(pert.get("serum_spike_matches", []))
            npd = len(pert.get("depletion_matches", []))
            ui.stat_row([(nd, "dose-responsive"), (ns, "serum-spike"), (npd, "depletion")])

    ui.rule()
    ui.section("6.4", "Why MSS exists — and why the BSV is semantic")
    a, b = st.columns(2)
    with a:
        ui.card("Why a motif layer at all?",
                "Themes are coarse (\"purine system\"). Molecules move *motifs* (\"oxopurine "
                "carbonyl\"). The MSS layer localises a change to a specific spectral pattern that a "
                "spectroscopist can point to — and, as Page 7 shows, it resolves perturbations that "
                "the coarse theme radar smears out.")
    with b:
        ui.card("Why the BSV is not just an embedding",
                "An autoencoder embedding is 24 opaque numbers. The BSV is **11 named biochemical "
                "axes**, each with a confidence and an out-of-distribution flag, each traceable "
                "down through motifs → components → the exact reference chemistries that define it. "
                "It is a *semantic* state you can read, question, and falsify — not a black box.")
    ui.note("take",
            "Component → MSS → theme → BSV is a fully deterministic, fully traceable path from a "
            "latent number to a biochemical claim with its own uncertainty attached.")
    ui.report_expander("MSS_AUDIT.md", "Read the MSS audit (Part 7)")
    ui.report_expander("BSV_AUDIT.md", "Read the BSV audit (Part 8)")
