# GAIRA Foundation Model Explorer V5

**From exact molecular identity to recoverable biochemical abstraction.**

```bash
streamlit run gaira_foundation_explorer_v5/app.py
```

## Launch every Explorer

```bash
streamlit run gaira_foundation_explorer/app.py        # V1 — the frozen model, end to end
streamlit run gaira_foundation_explorer_v2/app.py     # V2 — theme preservation
streamlit run gaira_foundation_explorer_v3/app.py     # V3 — the representation hierarchy
streamlit run gaira_foundation_explorer_v4/app.py     # V4 — null-calibrated recovery
streamlit run gaira_foundation_explorer_v5/app.py     # V5 — abstraction recovery (current)
```

**V5 is current.** V1–V4 are retained for historical reproducibility. V5 **does not change the
frozen atlas** (`09ed804a40836f4a05a91ba10900cded`) — it changes analysis and interpretation only,
reproduces V4's exact-identity counts, and the **molecular subclass is an evaluation overlay, not a
new learned layer**. MSS and themes remain the canonical GAIRA interpretive layers.

## The question V5 asks

V4 established that *exact* Ag-SERS identity recovery is rare (latent 7/51). V5 asks: **when exact
identity is lost, does the correct BROADER chemistry survive?** — component, MSS motif, molecular
subclass, or broad theme.

**Answer:** the expected motif/theme is often *present* in the Ag-SERS top-3 (MSS 40%, theme 49%),
but that presence is **not analyte-specific** — specific recovery is rare at every level (≤2/48), and
cross-modal subclass/family classification is **at chance**. A Raman→Raman control proves the
taxonomy is separable and that abstraction *helps within Raman* (0.23 → 0.42) — the **Ag-SERS
modality gap collapses it**. **Presence ≠ recovery.** Only functional perturbation (3 analytes)
recovers class chemistry beyond exact identity.

## The 18 pages

Overview · Foundation Dataset · Latent NMF Atlas · How GAIRA Interprets a Spectrum · Exact Analyte
Recovery · Component Evidence · MSS Motif Recovery · Molecular Subclass Recovery · Biochemical Theme
Recovery · **Recovery by Abstraction Level ★** · The Purine Attractor · Perturbation Validation ·
Matrix Recoverability · Individual Analytes · Biological Studies · Limitations · Future DART ·
Methods & Provenance.

## Scientific philosophy (V5)

- A high raw theme/motif is **presence**, not identity; presence ≠ recovery.
- **Never** claim a molecule is *identified* when only its MSS, subclass or broad theme is recovered.
- Molecular subclass is an **evaluation overlay**, never a frozen GAIRA axis.
- The Ag-SERS modality gap — not the taxonomy — is what collapses class recovery (Raman control).
- Functional perturbation (DART) is the route to class-specific recovery.

## Requirements & regeneration

`streamlit`, `plotly`, `pandas`, `numpy`, `scipy`, and the `gaira` package for the fingerprint check.
No SSD_Rad, no raw data at runtime. Regenerate:

```bash
python results/v5_rebuild/abstraction_recovery_v5/code/build_overlay.py
python results/v5_rebuild/abstraction_recovery_v5/code/abstraction_analysis.py
python results/v5_rebuild/abstraction_recovery_v5/code/make_figures_v5.py
python results/v5_rebuild/abstraction_recovery_v5/code/make_cards_v5.py
python results/v5_rebuild/abstraction_recovery_v5/code/make_report_v5_pdf.py   # needs reportlab
```
