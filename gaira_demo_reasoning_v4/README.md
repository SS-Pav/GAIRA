# GAIRA V6 — Scientific Reasoning Demo

The definitive public-facing demonstration of the **V6 Converged Reasoning Engine**,
presented as an interactive scientific paper. It tells one story end to end: *how
GAIRA transforms a raw Raman / SERS spectrum into biochemical reasoning.*

```bash
cd gaira_demo_reasoning_v4
streamlit run app.py        # or:  ./run_demo.sh
```

## Scientific purpose

GAIRA is not a classifier. A **frozen Raman Reference Atlas** (NMF k=24, fingerprint
`09ed804a…`) defines a fixed biochemical coordinate system. Every spectrum is
projected into it, explained through **Molecular Spectral Signatures (MSS)**,
aggregated into a **Biochemical State Vector (BSV)**, interpreted for its sample
domain, and reported with confidence, an out-of-distribution (OOD) score, and
provenance. The demo is **presentation only** — it drives the frozen engine
(`gaira.engine` + `gaira.engine.mss`) and modifies no science.

## Working modality position (stated honestly)

The biochemical coordinate system is **Raman-derived**. Current Ag-SERS calibration
supports *provisional* use of Raman-derived motifs **when an analyte is effectively
recovered** through adsorption + SERS enhancement. This does **not** establish
universal Raman→SERS invariance; recoverability is analyte- and matrix-dependent
(Page 5 maps the boundary). Future Au-SERS / DART will add an explicit *observation*
layer on top of — never inside — the frozen biochemistry. No modality-correction model
exists today.

## Interpretation hierarchy (used on every page)

```
Radar (11 themes)        →  which biochemical systems?      (summarizes)
Molecular Spectral Sig.  →  which spectral motifs?          (explains)      ← centerpiece
Latent components (24)   →  what numerical evidence?        (provenance)
Reference analytes (167) →  which reference chemistries?    (support)
```

## Page guide

| # | Page | Question | Status |
|---|------|----------|--------|
| 1 | Overview | What is GAIRA? | complete |
| 2 | Reference Atlas | What evidence does GAIRA know; how is the atlas built? | complete |
| 3 | How GAIRA Reasons | What does the engine conclude, traced to chemistry? | complete |
| 4 | Calibration | Do controlled perturbations move the right motif? | complete (signature cascade) |
| 5 | Serum Spike Stress Test | When is a serum analyte recoverable — and when not? | complete |
| 6 | Biological Studies | What differences does V6 find in real cohorts? | complete (3 genuine-V6 cohorts) |
| 7 | Future DART | How do dynamics extend the frozen framework? | conceptual (interface-level) |
| 8 | Methods & Provenance | Can every claim be traced and reproduced? | complete |

Highlights: the **signature reasoning cascade** (Page 4) — Spectrum → Components →
MSS → BSV → Radar, driven by a concentration slider; the **recoverability boundary**
(Page 5) — 6 strong / 8 partial / 39 poor of 53 serum analytes, and the honest finding
that confidence does not yet track adsorber strength; **genuine V6 biological cohorts**
(Page 6) — COVID serum-Raman (465 spectra), HCC serum-SERS (144), diabetes plasma-EV
(63 **patients**, patient-level stats), with a robust patient-level purine/sulfur
contrast in diabetes.

## Data modes

- **REAL** — raw data projected live through the V6 engine (committed sanitized
  artifacts for the biological cohorts; committed frozen-atlas projections for
  calibration/serum).
- **UNAVAILABLE** — present on disk but not yet wired for V6 (EV small2023, SHINE,
  liver): shown in the registry with a reason; **no output is fabricated**.

The demo runs on a **fresh checkout without the private data volume** — it reads only
committed artifacts. The volume is needed only to *regenerate* biological artifacts.

## Commands

```bash
streamlit run app.py                         # launch
python selfcheck.py                          # headless: render every figure, verify fingerprint
python tools/build_biological_v6.py          # regenerate biological artifacts (needs volume)
python -m pytest tests/test_v6_demo_v4.py -q  # (from repo root) demo guard tests
```

## Architecture

```
gaira_demo_reasoning_v4/
  app.py                     # Streamlit shell + 8-page nav + cross-page jumps
  selfcheck.py               # headless engine + figure audit
  demo_core/
    engine_bridge.py         # loads + drives the FROZEN engine and MSS layer
    theme.py                 # publication palette + matplotlib/CSS styling
    figures.py               # matplotlib publication figures (auditable)
    interactive.py           # plotly reference-PCA + Sankey
    components.py            # shared chrome (header/caption/caveats/related nav)
    data.py / calibration.py / serum.py / biological.py   # per-page analysis
    pages/                   # one module per page
  tools/build_biological_v6.py   # SSD → sanitized committed V6 artifacts
  biological_artifacts/          # committed genuine-V6 cohort outputs
  DATA_PROVENANCE_AUDIT.md · SCIENTIFIC_CLAIMS_AUDIT.md · VISUAL_AUDIT.md
  GAIRA_V6_DEMO_BUILD_REPORT.md
```

## Frozen scientific dependencies (never modified by the demo)

Raman Reference Atlas + `manifold_components.npz` basis · preprocessing (asls+savgol+
L2) · NMF k=24 · Component Registry v1 · Biochemical Ontology v2 · component→theme
weights · MSS registry v1 · BSV v2 equations · confidence engine · Evidence Engine ·
domain context. The atlas fingerprint is verified on every load.

## Limitations

- SERS / serum / EV are out-of-domain for a Raman atlas — read group differences with
  OOD and confidence, never as absolute quantitation.
- Confidence tracks domain/spectrum quality, not analyte recoverability (Page 5).
- Biological findings are association-level ("consistent with"), never diagnoses.
- Page 7 (DART) and the Au-SERS layer are conceptual; no such data or model exists yet.
