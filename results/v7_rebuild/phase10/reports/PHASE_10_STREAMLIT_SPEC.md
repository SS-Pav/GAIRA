# GAIRA V7 — Streamlit Client Specification

`streamlit_apps/gaira_v7_console.py` · `gaira streamlit` · 717 lines, **zero scientific
computation**

The landing line: *"Upload a Raman spectrum to project it into a frozen biochemical motif atlas,
retrieve grounded reference evidence, and generate an interpretable Chemistry Evidence profile."*

---

## It is a client, and this is enforced

`tests/test_v7_phase10_parity.py` parses the app with `ast` and fails the build if it references
any of `nnls`, `NMF`, `PCA`, `savgol_filter`, `cosine_similarity`, `find_peaks`, … or imports
`scipy.*`, `sklearn.*`, or any `gaira.v7` scientific module — **including `gaira.v7.canonical`**,
because the UI must go through the runtime rather than reach past it.

Parsed rather than grepped: the app's own docstring lists what it excludes, and a substring
search would flag the documentation stating the exclusion.

## Backend selection is configuration only

```bash
streamlit run streamlit_apps/gaira_v7_console.py                    # local engine
GAIRA_API_URL=http://localhost:8000 streamlit run …                 # deployed API
```
The request and result schemas are identical either way, and so are the numbers.

## Pages

| page | shows |
|---|---|
| **Analyze Spectrum** | upload, raw preview, parse diagnostics, metadata, scope banner, run; then five tabs — Chemistry Evidence, Grounded Evidence Retrieval, Preprocessing, CSM representation, Interpretation |
| **Scientific Audit** | confidence, explained variance, margins, reconciliation, warnings, and the open-set limitation stated in full |
| **Evidence & Provenance** | the four-layer chain with per-CSM diagnostic bands, band assignments and contributing LSMs; atlas fingerprints |
| **Compare Spectra** | two independent runs, per-axis difference chart, side-by-side retrieval |
| **Engine Information** | fingerprints, corpus, validated performance, limitations, and two methodology expanders |

## Language discipline

**Used** — *Grounded Evidence Retrieval*, *reference analogues*, *relative chemistry evidence*,
*retrieved reference spectra*.
**Never used** — *AI identifies your molecule*, *foundation model detects disease*, *clinical
diagnosis*, *concentration*, *abundance*, *composition*.

## Required displays

1. **Ordered bars are the default**; the radar is complementary. A bar chart is readable at a
   glance and precise; a radar is good for pattern recognition and bad for reading values.
2. Every chemistry view carries *"Relative biochemical evidence. Not a concentration, not an
   abundance, not a mixture fraction."*
3. The retrieval tab opens with *"Candidates are retrieved reference analogues, not definitive
   molecular identifications."*
4. Score reconciliation is shown per candidate — contributions summed against the similarity.
5. The audit page states the open-set limitation with its measured numbers (white noise
   reconstructs at CSM EV ≈ 0.61, above the 0.50 floor; the flag fires on 1 of 20).
6. Selecting a non-Raman modality shows a red block and **prevents** inference.
7. A non-pure sample type shows an amber scope warning and proceeds with unchanged arithmetic.

## Preprocessing display

The processed spectrum and the CSM reconstruction are **returned by the engine** via
`include_reconstruction=True`, which calls `GAIRAEngine.prepare()` — the same routine `infer()`
used. The UI never reproduces a transformation, so what is drawn is exactly what the projection
consumed.

## UX

Clean typography, generous whitespace, five metric tiles, tabbed detail, expandable expert
sections, no debug widgets, no default mega-tables. Large tables sit behind expanders; the
16-axis table is collapsed by default with the top five always visible.
