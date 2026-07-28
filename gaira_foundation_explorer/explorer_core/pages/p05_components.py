"""Page 5 — Understanding the Components."""
from __future__ import annotations
import pandas as pd
import streamlit as st

from .. import data as D, ui, charts as C, theme as T


def render():
    df = D.components_table()
    cls = D.component_classification()
    ui.page_header("The coordinate axes", "Understanding the Components",
                   "Each of the 24 latent components is a spectral basis function — a Raman motif "
                   "with a few characteristic bands, the reference chemistries that activate it, and "
                   "the biochemical themes it feeds.")
    ui.question("What are the latent components, chemically — and are they stable, clean, and "
                "non-redundant?")

    ui.stat_row([
        (len(df), "components"),
        (ui.fmt(cls.get("mean_stability"), 2), "mean stability"),
        (ui.fmt(cls.get("min_stability"), 2), "min stability"),
        (ui.fmt(cls.get("max_pairwise_basis_cosine"), 2), "max pairwise cosine"),
        (len(cls.get("chemically_clean", [])), "clean anchors"),
    ])
    ui.note("take",
            f"Every component is stable (bootstrap ≥ {ui.fmt(cls.get('min_stability'),2)}) and "
            f"<b>none is redundant</b> — the largest similarity between any two basis spectra is only "
            f"{ui.fmt(cls.get('max_pairwise_basis_cosine'),2)}. The 24 axes are genuinely distinct.")

    ui.rule()
    ui.section("5.1", "The component map")
    if len(df):
        st.plotly_chart(C.component_map(df), width="stretch",
                        config={"displayModeBar": False})
        st.markdown('<div class="small">Each point is a component: variance share (x) vs bootstrap '
                    'stability (y), sized by its top-theme weight, coloured by chemical purity. '
                    'Top-right and darker = a large, stable, chemically clean axis. Hover for the '
                    'chemistry.</div>', unsafe_allow_html=True)

    ui.rule()
    ui.section("5.2", "What the components represent")
    groups = [
        ("Purine-dominated", cls.get("purine_dominated", []),
         "The nucleic-acid/purine system — GAIRA's best-validated axis. c0/c3 are near-twin adenine "
         "motifs (722/1334 cm⁻¹); c15 is the cleanest purine loading."),
        ("Protein-dominated", cls.get("protein_dominated", []),
         "Amide backbone + phenylalanine. c2 is the standout clean protein component (purity 0.80)."),
        ("Lipid / sterol", cls.get("lipid_sterol_dominated", []),
         "Acyl-chain CH₂ and ester modes — the corpus's largest-variance chemical family."),
        ("Saccharide / glycan", cls.get("saccharide_dominated", []),
         "Sugar C–O–C ring modes (800–1150 cm⁻¹); c10/c12 are the cleanest."),
    ]
    cols = st.columns(2)
    for i, (name, comps, desc) in enumerate(groups):
        with cols[i % 2]:
            chips = " ".join(f"c{c}" for c in comps)
            ui.card(f"{name} · {len(comps)}", f"`{chips}`\n\n{desc}")

    c1, c2 = st.columns(2)
    with c1:
        ui.card("Chemically clean anchors",
                f"Components with high theme weight, high purity, one dominant family and sharp "
                f"bands: `{' '.join('c'+str(c) for c in cls.get('chemically_clean', []))}`. "
                f"These anchor their themes.")
    with c2:
        ui.card("Honestly mixed",
                f"Low-purity or multi-family components: "
                f"`{' '.join('c'+str(c) for c in cls.get('mixed_ambiguous', []))}`. Not errors — "
                f"they encode real spectral overlap (shared nucleic-acid backbone, shared acyl CH₂).")

    ui.rule()
    ui.section("5.3", "Component explorer")
    st.markdown("Select any component to see its basis spectrum, top peaks, top analytes, linked "
                "themes and motifs, and documented collisions.")
    labels = {int(r.component): f"c{int(r.component)} · {r.audit_label} → {r.top_theme}"
              for _, r in df.iterrows()}
    j = st.selectbox("Component", sorted(labels), format_func=lambda k: labels[k])
    row = df[df.component == j].iloc[0]
    left, right = st.columns([1.15, 1])
    with left:
        fig = D.component_figure(int(j))
        if fig:
            st.image(str(fig), width="stretch")
            st.markdown('<div class="small">Basis spectrum with its dominant Raman bands marked '
                        '(dotted).</div>', unsafe_allow_html=True)
    with right:
        ui.stat_row([
            (ui.fmt(row.stability, 2), "stability"),
            (ui.fmt(row.purity, 2), "purity"),
            (ui.fmt(row.variance_share, 3), "variance share"),
        ])
        st.markdown(f"**Audit label:** `{row.audit_label}`  \n"
                    f"**Top theme:** {row.top_theme} (weight {ui.fmt(row.top_theme_w,2)})  \n"
                    f"**Top analyte:** {row.top_analyte}  \n"
                    f"**Nearest component:** c{int(row.nearest_comp)} (cos "
                    f"{ui.fmt(row.nearest_cos,2)})  \n"
                    f"**Chemical families in top-8 loadings:** {int(row.n_families_top8)}")
        tag = "clean" if "clean" in str(row.get("tags", "")) else "mixed"
        ui.pills([("clean anchor" if tag == "clean" else "mixed / ambiguous", tag)])
    with st.expander("Full component audit page", expanded=True):
        page = D.load_component_page(int(j))
        # drop the local image link (already shown above) and render the rest
        page = "\n".join(l for l in page.splitlines() if not l.strip().startswith("!["))
        st.markdown(page)

    ui.report_expander("COMPONENT_AUDIT.md", "Read the global component audit (Part 6)")
