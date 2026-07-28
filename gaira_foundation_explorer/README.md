# GAIRA Foundation Explorer

**Building and validating the frozen Raman biochemical reference space.**

An interactive review article that documents and scientifically validates the GAIRA
foundation model — the frozen, pure-Raman NMF coordinate system into which every future
spectrum is projected. It is written for spectroscopy researchers, computational
biologists, ML researchers, clinicians and reviewers: it explains *how* the model was
built, *why* each design decision was made, and *why* the resulting biochemical
coordinate system is trustworthy.

This is a **standalone** application. It does **not** modify (or depend on) the existing
GAIRA reasoning demo (`gaira_demo_reasoning_v4/`).

## Run

```bash
streamlit run gaira_foundation_explorer/app.py
```

Requires `streamlit`, `plotly`, `pandas` (already in the project environment).

## What it shows

Eleven pages, each answering one scientific question, in a single narrative arc —
**problem → grounding corpus → representation learning → frozen coordinate system →
interpretability → validation → conclusions**:

1. **The GAIRA Foundation Model** — why Raman lacks a universal coordinate system; local
   vs global coordinates; the freeze-and-project philosophy.
2. **The Grounding Corpus** — every training source (Raman) and validation source (SERS),
   class balance, coverage gaps, data-quality; why SERS is excluded from fitting.
3. **Preprocessing** — the deterministic pipeline stage by stage; the <1 %-mass
   non-negativity clip; why it was frozen.
4. **Learning the Reference Space** *(centerpiece)* — V ≈ WH; why NMF beats PCA/ICA/AE;
   the interactive benchmark; why k = 24; why ICA scored highest yet NMF is the scientific
   choice; byte-identical reproducibility.
5. **Understanding the Components** — the 24 axes as an interactive map + per-component
   explorer (basis spectrum, peaks, analytes, themes, motifs, collisions).
6. **From Components to Biochemistry** — the MSS motif layer, the ontology, the BSV
   equations; an interactive motif explorer; why the BSV is a *semantic* state.
7. **Validation** — six tabs (pure Raman, Gobbato SERS transfer, adenine, ergothioneine,
   serum spike-in, uricase depletion), each with the interactive result.
8. **Results Summary** — eleven finding cards (question / evidence / conclusion).
9. **Limitations** — corpus gaps and the representation-vs-measurement distinction.
10. **Future Architecture** — the observation-model stack; why SERS/DART extend, not
    replace, the frozen coordinate system.
11. **Scientific Takeaways** — the vision: a biochemical coordinate system, not a
    classifier.

## Where the content comes from

Everything is generated from the completed audit at
**`results/v5_rebuild/foundation_audit/`** — reports (`reports/*.md`), tables
(`tables/*.json|csv`), figures (`figures/*.png`) and the 24 per-component pages
(`components/`). No scientific number is hardcoded in the UI; the pages call
`explorer_core/data.py`, which loads directly from those artifacts (and the frozen atlas
`manifold.json`). Regenerate the audit and this app updates automatically.

## Layout

```
gaira_foundation_explorer/
  app.py                     # sidebar nav + page dispatch
  .streamlit/config.toml     # light theme
  explorer_core/
    data.py                  # cached loaders (all content lives in foundation_audit/)
    theme.py                 # review-article CSS
    ui.py                    # figure cards, stat tiles, flow diagrams, callouts
    charts.py                # interactive Plotly charts (benchmark, transfer, component map, dose)
    pages/p01…p11.py         # one module per page
```
