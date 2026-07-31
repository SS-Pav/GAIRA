# GAIRA Foundation Explorer V3 — The Representation Hierarchy

An interactive walkthrough of Raman → Ag-SERS transfer organised as a **five-level hierarchy of
representations** — from latent coordinates, through motifs and biochemical themes, to dynamic
perturbation and matrix robustness.

```bash
streamlit run gaira_foundation_explorer_v3/app.py
```

## Additive — V1 and V2 are untouched

This is the third, additive generation. The original **Foundation Explorer**
(`gaira_foundation_explorer/`) and **Explorer V2** (`gaira_foundation_explorer_v2/`) both remain
unchanged and runnable. V3 reorganises the interpretation around the hierarchy and adds new
first-class metrics, **keeping every earlier metric for transparency**.

It reads **only** committed artifacts from
`results/v5_rebuild/representation_hierarchy_v3/` (tables, the eight audited figures, the
9-layer per-analyte cards, and the spec/story/guide/changelog docs) plus the frozen atlas
fingerprint `09ed804a…`, verified at load. Every V2 number is reproduced bit-for-bit. Nothing is
retrained.

## What is new in V3

- **Theme RANK preservation (Spearman ρ)** — ordering of all 11 themes, with a null control that
  shows raw ρ is baseline-inflated (honest identity signal = the small rank separation).
- **Top-k theme overlap** promoted to a first-class metric (the interpretable middle-ground).
- **The Representation Hierarchy** as the central conceptual figure, with per-level distributions.
- **The purine attractor, quantified** — interactive Sankey (Raman→Ag dominant theme), per-analyte
  ΔPurine, and ΔPurine vs latent fidelity (r=−0.38, p=0.006).
- **Matrix robustness regression** — pure transfer is only a weak predictor of serum recovery
  (r=0.17, n.s.): an honest quantitative downgrade of V2's categorical claim.
- **9-layer per-analyte cards** in physics-aware language.

## The 15 pages

Overview · **Representation Hierarchy ★** · L1 Latent fingerprint · L2 MSS motifs · L3 Theme
(raw + identity) · **L3 Theme rank ρ (NEW)** · **L3 Top-k overlap (NEW)** · L3 Argmax agreement ·
**The Purine Attractor (Sankey + ΔPurine)** · Cross-Modal Transfer · L4 Perturbation · L5 Matrix
Robustness · Per-Analyte Cards · Framework & Methods · Verdict.

## The one idea

Transfer is not one number — it is a hierarchy, and the honest question is *how far up it
agreement survives*. Raw agreement rises with abstraction (0.42 → 0.74 → 0.92); identity-specific
agreement falls at the top (0.11, argmax 35%), because silver homogenises most analytes toward a
purine attractor. A rare minority — adenine foremost — redistribute their latent fingerprint yet
keep a dose-responsive theme. V3 shows the whole ladder, honestly.

## Requirements

`streamlit`, `plotly`, `pandas`, `numpy`, and the `gaira` package (`pip install -e .`) for the
fingerprint check. No SSD_Rad, no raw data. If the engine can't be imported (minimal CI), the
fingerprint check falls back to the recorded summary and the app still runs.

## Regenerate the underlying artifacts

```bash
python results/v5_rebuild/representation_hierarchy_v3/code/hierarchy_analysis.py
python results/v5_rebuild/representation_hierarchy_v3/code/make_cards_v3.py
python results/v5_rebuild/representation_hierarchy_v3/code/make_figures_v3.py
```
