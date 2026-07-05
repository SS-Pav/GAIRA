# BUILD REPORT — v2 Tab 2 REBUILD (concept-first)

**Date:** 2026-04-26
**Phase:** GAIRA_COMMAND_CENTER_STREAMLIT_VERSION_REPAIR_AND_V2_REBUILD
**Decision:** SHIPPED

---

## What this is

`streamlit_apps/gaira_command_center_v2/` is the **redesigned Tab 2** with a
concept-first storyline. The broken family-first hull plot is removed; the
saliency map and ambiguity story are now the primary visual evidence.

## What changed (broken intermediate → v2 redesign)

| issue (broken intermediate) | v2 fix |
|---|---|
| Family-first motif map at the top, with huge ellipses + overlapping labels | Removed from default. Now an opt-in `experimental_family_overlay` toggle on the UMAP, default OFF |
| "Top 10 bands driving G05" table on first load | Hidden under saliency map; collapsed `Inspect one BSV axis` expander, axis selectable |
| BSV taxonomy as a single huge dataframe | Replaced by a 2-column compact card grid — each axis as `G05 · glycan_carbohydrate` with chemistry / canonical bands / caveat |
| Axis overlap network with random force-layout + overlapping labels | Manual chemistry-grouped circular layout; short G-id inside nodes; full family name OUTSIDE on a leader-line position; only top-60% edges drawn; reduced edge alpha |
| MSS / motif UMAP labelling every cluster | Only ~9 known major classes annotated (`MAJOR_CLASS_LABELS`); class legend collapsed below figure (toggle in advanced); experimental family overlay opt-in |
| Saliency canonical band labels always shown, illegible | Hidden by default (subtle dashed guidelines remain); `Show canonical band labels` toggle reveals 8 stagger-positioned labels |
| Dendrogram with no annotation | Side-by-side images + numbered callouts (4-tier interpretation) + per-cluster summary table with `caveat` column derived from purity |
| Hybrid flow cluttered | 7-node clean diagram (added Pre-processing + Confidence/ambiguity nodes) |

## Tab 2 v2 sections

```
A · Representation hierarchy           — 5-step concept diagram
B · 11-axis BSV taxonomy               — compact card grid (no huge dataframe)
C · BSV saliency · band ⇒ axis        — heatmap, canonical labels OFF default,
                                         per-axis inspector collapsed
D · Shared bands & ambiguity           — green/orange/red traffic-light
E · Axis overlap network               — manual chemistry-grouped layout
F · MSS / motif UMAP (clean)           — only major clusters labelled, legend collapsed
G · Annotated dendrograms              — image + callouts + cluster summary table
H · Hybrid BSV evidence flow           — polished 7-node Plotly diagram
↓
Tab 3 link card
```

## Sidebar (default vs advanced)

**Default (always visible):**
- Saliency band bin (5 / 10 / 20 / 25 / 50 cm⁻¹)
- Ambiguity / network threshold

**Advanced (collapsed expander):**
- Saliency canonical band labels toggle
- Dendrogram cluster-summary mode (MOTIF / MSS)
- UMAP marker opacity, label toggle, legend toggle
- Experimental: family ellipse overlay on UMAP

## Files changed / added

| file | change |
|---|---|
| `components/motif_mss_bsv_tab.py` | full rewrite — 8 sections in v2 order |
| `README.md` | replaced with v2-specific copy |
| `BUILD_REPORT_v2_TAB2_REBUILD.md` | NEW (this file) |
| All other files | inherited unchanged from the prior state |

GAIRA core untouched.

## Acceptance check

| criterion | result |
|---|---|
| All modules import cleanly | ✅ utils.* + components.* + app |
| Live `streamlit run` boots | ✅ HTTP 200 on `/` and `/_stcore/health` (port 8766); no errors / tracebacks |
| No unreadable family-first hull plot | ✅ removed from default; behind experimental toggle on UMAP only |
| Saliency map prominent and clean | ✅ default heatmap with subtle dashed guidelines; canonical labels OFF; per-axis table collapsed |
| Overlap network readable | ✅ manual layout; G-id inside, family name outside; top-60% edges only |
| MSS / motif UMAP clearer than broken intermediate | ✅ only ~9 major classes labelled; legend collapsed by default |
| Dendrogram has useful annotations | ✅ numbered callouts + cluster summary table with caveats |
| No random G05 table shown by default | ✅ collapsed inside `Inspect one BSV axis` expander |
| Tab 1 unchanged | ✅ overview_tab.py untouched |

## Tab-2 30-second test

Targeted answers a new viewer should get within 30 seconds:

1. **What are the 11 GAIRA biochemical axes?** → Section B card grid (`G01 · purine_nucleotide` … `G11 · metabolic_small_molecule`).
2. **Where do they live in spectral space?** → Section C heatmap rows.
3. **Which axes overlap?** → Section E network thickness + side table.
4. **Why does MSS help disambiguate?** → Section D traffic-light bars (red bars = collision).
5. **Why are molecule calls caveated in biofluids?** → Card-level caveats in Section B + interpretation panels in D / G + Tab 3 link.

## How to run

```bash
streamlit run streamlit_apps/gaira_command_center_v2/app.py
```

Default port 8501. Run alongside v1 by picking different ports.

## Remaining experimental / not-yet-done

- Family ellipse overlay on the UMAP is still imperfect (chemistry naturally overlaps in 2-D embedding); kept as opt-in advanced toggle.
- Dendrograms are still pre-rendered PNGs with text annotations rather than a native interactive Plotly dendrogram.
- Axis-overlap network uses curated `EDGE_INTERPRETATION` strings; a few pairs are flagged `(uncurated)` in the side table.

## Strict invariants preserved

- GAIRA core unchanged.
- No GAIRA scoring rerun.
- All loads cached via `@st.cache_data`.
- Missing-artifact tolerance: every loader returns `None` and the renderer surfaces a soft `gaira-warn` card.
- All visualisation-only computations (ellipses, network edges, traffic-light counts) explicitly marked.
