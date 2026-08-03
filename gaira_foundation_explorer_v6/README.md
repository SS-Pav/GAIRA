# GAIRA Foundation Model Explorer V6

**A detection gate before recovery — can we *see* it before we ask whether we can recover it?**

```bash
streamlit run gaira_foundation_explorer_v6/app.py
```

## Launch every Explorer

```bash
streamlit run gaira_foundation_explorer/app.py        # V1 — the frozen model, end to end
streamlit run gaira_foundation_explorer_v2/app.py     # V2 — theme preservation
streamlit run gaira_foundation_explorer_v3/app.py     # V3 — the representation hierarchy
streamlit run gaira_foundation_explorer_v4/app.py     # V4 — null-calibrated recovery
streamlit run gaira_foundation_explorer_v5/app.py     # V5 — abstraction recovery
streamlit run gaira_foundation_explorer_v6/app.py     # V6 — detection gate (current)
```

**V6 is current.** V1–V5 are retained for historical reproducibility. V6 **does not change the
frozen atlas** (`09ed804a40836f4a05a91ba10900cded`) — it adds a Stage-0 detection gate and reuses
V5's recovery flags unchanged.

## What V6 adds

V5 evaluated all 51 analytes equally, including analytes whose Ag-SERS is essentially blank. That
conflates **measurement failure** (invisible on silver) with **representation failure** (measured
but chemistry not recovered). V6 inserts a **detection gate** (Stage 0): *does this Ag-SERS spectrum
contain reproducible analyte information above noise/background?*

- A deterministic, **no-ML** Detection Confidence (replicate Pearson + peak SNR + variance
  concentration + reproducible peaks), **validated before freezing** (`code/validate_detection.ipynb`):
  adenine/ergothioneine/urate/xanthine pass; glucose/tyrosine/oleate fail — for adsorption reasons.
- **22/51 detectable; 29 undetectable.**
- Restricting the V5 hierarchy to detectable analytes ~**doubles** exact identity (14%→23%) and
  lifts presence (MSS 40%→55%), but analyte-**specific** recovery stays low — the residual failure is
  representational.
- A **transfer-model decision** (measurement- vs representation-limited) and a **roadmap**: ~11
  detectable, representation-limited analytes are the concrete target for a future learned
  Raman→SERS model; the rest need a better substrate.

## The 13 pages

Overview · Detection Gate · Detection Metrics · Representative Spectra · Detection Confidence ·
Detectable vs Undetectable · Recovery Hierarchy · Recoverable Analytes · Transfer Function
Assessment · Roadmap · Individual Analytes · Limitations · Final Conclusions.

## Scientific philosophy (V6)

- Confirm the analyte gives **reproducible signal** on the substrate before interpreting it.
- An invisible analyte is a **substrate** problem, not a GAIRA problem — never "recovered."
- Replicate **cosine** is baseline-inflated; mean-centred **Pearson** is the honest reproducibility.
- A learned Raman→SERS transfer model is only justified for **detectable, representation-limited** analytes.

## Requirements & regeneration

`streamlit`, `plotly`, `pandas`, `numpy`, `scipy`, and the `gaira` package. No SSD_Rad at runtime.

```bash
python results/v5_rebuild/detection_gate_v6/code/detection_gate.py
python results/v5_rebuild/detection_gate_v6/code/restricted_hierarchy.py
python results/v5_rebuild/detection_gate_v6/code/make_figures_v6.py
python results/v5_rebuild/detection_gate_v6/code/make_report_v6_pdf.py   # needs reportlab
# threshold validation: results/v5_rebuild/detection_gate_v6/code/validate_detection.ipynb
```
