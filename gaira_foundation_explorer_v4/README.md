# GAIRA Foundation Model Explorer V4

**Hierarchical biochemical recovery across Raman, Ag-SERS, perturbation and biological matrix.**

```bash
streamlit run gaira_foundation_explorer_v4/app.py
```

## Launch every Explorer

```bash
streamlit run gaira_foundation_explorer/app.py        # V1 — the frozen model, end to end
streamlit run gaira_foundation_explorer_v2/app.py     # V2 — theme preservation
streamlit run gaira_foundation_explorer_v3/app.py     # V3 — the representation hierarchy
streamlit run gaira_foundation_explorer_v4/app.py     # V4 — null-calibrated recovery (current)
```

**V4 is current.** V1–V3 are retained for historical reproducibility. V4 **does not change the
frozen atlas** (`09ed804a40836f4a05a91ba10900cded`) — it changes analysis and interpretation only,
and reproduces every V3 matched value bit-for-bit.

## What V4 adds

Transfer is not one score — and not one *level*. V4 calibrates **every** representation metric
against an **analyte-mismatched null**, so "recovery" means the analyte's own Ag-SERS is uniquely
nearest (retrieval rank-1) and jackknife-stable — **never** a raw cosine above a threshold.

Headline result: analyte-specific cross-modal recovery is **rare at every level** — latent
**7/51**, MSS **3/51**, theme **4/51** — and **MSS is not the primary metric** (its null separation
is smaller than the latent coordinates', and its recovered set is a strict subset). The strongest
evidence is functional **perturbation** (3 analytes). The purine attractor appears in the
**unspiked-serum blank** (purine share 0.27) *before any analyte*.

## The 15 pages

Overview · Foundation Dataset · Latent Representation · How GAIRA Interprets a Spectrum ·
Cross-Modal Validation · MSS Motif Recovery · Biochemical Theme Interpretation ·
**Recoverable Analytes ★** · The Purine Attractor · Perturbation Validation · Matrix Recoverability ·
Biological Studies · Limitations · Future DART · Methods & Provenance.

## Scientific philosophy (V4)

- **Latent cosine** — substrate / fingerprint fidelity (and the best cross-modal identity cosine).
- **MSS** — an intermediate motif-preservation *candidate* that the null rejects as primary.
- **Raw theme BSV** — broad biochemical interpretation, **not** analyte identity.
- **Residual / null-adjusted theme metrics** — analyte-specific diagnostic.
- **Perturbation** — functional validation (the strongest evidence).
- **Matrix recovery** — mixture visibility (a separate property).

Never call an analyte "detectable" from a raw cosine above an arbitrary threshold.

## Requirements

`streamlit`, `plotly`, `pandas`, `numpy`, and the `gaira` package for the fingerprint check. No
SSD_Rad, no raw data. Regenerate the underlying artifacts:

```bash
python results/v5_rebuild/hierarchical_recoverability_v4/code/recoverability_analysis.py
python results/v5_rebuild/hierarchical_recoverability_v4/code/make_figures_v4.py
python results/v5_rebuild/hierarchical_recoverability_v4/code/make_cards_v4.py
python results/v5_rebuild/hierarchical_recoverability_v4/code/make_report_v4_pdf.py   # needs reportlab
```
