# GAIRA V6 — Scientific Reasoning Demo

The public-facing demonstration of the **V6 Converged Reasoning Engine**, presented
as an interactive scientific paper. It tells one story: *how GAIRA turns a raw
Raman / SERS spectrum into biochemical reasoning.*

```
streamlit run app.py      # or:  ./run_demo.sh
```

## What this demo is (and is not)

- It is **presentation only**. Every number comes from the frozen engine in
  `gaira.engine` + the derived MSS layer in `gaira.engine.mss`. The atlas,
  preprocessing, NMF, ontology, component registry, theme weights, BSV equations,
  confidence engine and MSS layer are **frozen** and are never modified here.
- It is built **page by page** on that frozen engine. Pages 1, 3, and the live
  panels of 2/4/8 are wired to the engine; pages 5–7 render their scientific frame
  and are being wired next (each page states its status honestly).

## The reasoning hierarchy (this demo's spine)

```
Radar (13 themes)   →  which biochemical systems changed?
MSS (13 motifs)     →  which biochemical spectral motifs explain it?   ← centerpiece
Components (24)      →  what latent evidence supports those motifs?
Reference (167)      →  which reference chemistries contributed?
```

The **Molecular Spectral Signatures (MSS)** layer is the interpretive middle: the
place where mathematics (latent components) meets chemistry (biochemical themes).
Motif *definitions* (bands, exemplars, parent theme) are curated; motif
*contributors, confidence and perturbation evidence* are **derived** from the frozen
artifacts — nothing is asserted.

## Pages

| # | Page | Question it answers |
|---|------|---------------------|
| 1 | Overview | What is GAIRA, in one figure and a few numbers? |
| 2 | Reference Atlas | What chemistry did the frozen atlas learn? |
| 3 | How GAIRA Reasons | What does the engine conclude, traced to reference chemistry? |
| 4 | Calibration | Do controlled perturbations move the right motif monotonically? |
| 5 | Serum Spike Stress Test | Where does Ag-SERS succeed and fail? |
| 6 | Biological Studies | What separates biological cohorts, and how much to trust it? |
| 7 | Future DART | How do dynamics extend the frozen framework? |
| 8 | Methods & Provenance | Versions, fingerprints, equations — nothing hidden. |

## Auditing the visualizations

```
python selfcheck.py       # loads the engine, renders every figure to _selfcheck/
```

`selfcheck.py` runs headless (no browser), verifies the frozen atlas fingerprint is
untouched, and renders every publication figure to PNG so the visuals can be
reviewed without Streamlit.

## Architecture

```
gaira_demo_reasoning_v4/
  app.py                    # Streamlit shell + 8-page navigation
  selfcheck.py              # headless engine + figure audit
  demo_core/
    engine_bridge.py        # loads + drives the FROZEN engine and MSS layer
    theme.py                # publication palette + matplotlib/CSS styling
    figures.py              # matplotlib publication figures (auditable)
    components.py           # shared Streamlit chrome (header/caption/caveats/provenance)
    data.py                 # dataset catalog + cached-projection loader
    pages/                  # one module per page, each exposing render(bridge)
```

No science is re-implemented here. The demo is additive and does not touch the
prior demos (`gaira_demo_reasoning_v1..v3_1`) or the frozen engine.
