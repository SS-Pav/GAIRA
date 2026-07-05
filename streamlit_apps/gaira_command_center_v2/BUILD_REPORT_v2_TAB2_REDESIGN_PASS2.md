# BUILD REPORT — v2 Tab 2 REDESIGN · pass 2 (family-first storyline)

**Date:** 2026-04-26
**Phase:** GAIRA_COMMAND_CENTER_STREAMLIT_V2_TAB2_REDESIGN_PASS2
**Decision:** SHIPPED

---

## What this is

Pass 2 of the v2 Tab 2 redesign. The previous pass (`pass 1` /
`BUILD_REPORT_v2_TAB2_REBUILD.md`) had `B · 11-axis BSV taxonomy` cards as
the primary visual but no actual scientific *figure* leading the story.
This pass replaces that with a **family-first cluster constellation map**
that uses chemistry-grouped centroid positions (not UMAP-derived), keeps
MSS analyte structure as the secondary view, adds a per-family drilldown,
and demotes the axis network behind a heatmap matrix.

v1 is **unchanged**. Only v2 was edited.

## New Tab 2 section order

```
A · Representation hierarchy           — concept diagram + thesis line
B · Motif / 11-family cluster map      — PRIMARY scientific figure
C · MSS analyte cluster map            — SECONDARY · single full-width
D · MSS-within-family drilldown        — dropdown G01–G11
E · BSV saliency map                   — band ⇒ axis (clean defaults)
F · Shared-band ambiguity map          — green/orange/red traffic-light
G · Axis overlap matrix                — heatmap default; network in expander
H · Hierarchical clustering support    — expander
I · Hybrid evidence flow               — expander
↓
Tab 3 link card
```

## What was removed / hidden by default

- ✗ The single split MSS+Motif UMAP as the only main plot — replaced by the family-first map (B).
- ✗ Giant family ellipses across the embedding — gone. Centroids are placed on a chemistry-grouped layout; member analytes scatter around them with a small Fibonacci-sunflower jitter for stable visual density.
- ✗ Random "Top 10 bands driving G05" table on first load — only appears when the user opens `Inspect axis details` *and* picks an axis (default G01).
- ✗ Messy axis-overlap network as the default — replaced by a clean 11×11 heatmap matrix with cell annotations on top-overlap cells. The network is still available behind an expander with manual layout + only top-8 edges drawn.
- ✗ Dense saliency canonical band labels by default — only subtle dashed guidelines; toggle in advanced controls.
- ✗ Dense legends consuming half the page — MSS analyte map legend is hidden by default; toggle in sidebar to show as horizontal bar below the figure.

## What was redesigned

### B · Motif / 11-family cluster map (new primary figure)

`render_motif_family_cluster_map()` — replaces the old "first scientific figure".

- **Centroid positions** come from `utils/layout_constants.AXIS_POSITIONS`: chemistry-grouped quadrants (nucleic upper-left, glycan/phosphate centre-top, protein/aromatic upper-right, lipid/sterol right/lower-right, sulfur/metabolic bottom).
- **Member analytes** are scattered on a deterministic Fibonacci sunflower around their family centroid — same family always renders the same shape.
- **Centroid markers** scale with member count (G11 = 82, G06 = 31, G09 = 27, G05 = 24 …); short G-id label inside the marker; full family name on a leader-line label *outside* the cluster, biased upward when the centroid is near horizontal centre to avoid label collisions.
- **Optional faint hull** (default OFF) only drawn when the family has ≥ 4 members.
- **Hover** per analyte: id · biochemical class · BSV family · top anchor bands.
- **Below the figure**: a 2-column compact card legend repeating the 11 axes with `chemistry / canonical bands / caveat / # analytes`. This is the readable replacement for the huge taxonomy dataframe.

### C · MSS analyte cluster map (single full-width)

`render_mss_analyte_cluster_map()` — replaces the cramped split MSS+Motif view.

- Single full-width Plotly Scattergl over the 236 MSS-embedding points, coloured by `broad_class`.
- Annotation budget: only the 10 names in `MAJOR_CLASS_LABELS` (`protein_polypeptide`, `free_amino_acid`, `free_fatty_acid`, `triglyceride`, `sugar`, `tryptophan_indole`, `organic_acid_metabolite`, `sterol`, `purine_nucleobase`, `sulfur_amino_acid`). Anything else is in the legend (collapsed by default).
- Manual de-collision: if a candidate label sits within 0.7 / 0.5 of an already-placed one, it is nudged 0.6 in y.
- Interpretation panel beneath: *"MSS is analyte-level. Nearby points share anchor / support evidence; overlap means shared vibrational chemistry."*

### D · MSS-within-family drilldown (NEW)

`render_mss_within_family_drilldown()` — **this is where molecule-level detail belongs in Tab 2**, replacing any hard-coded G05 table.

- Sidebar dropdown: `BSV family for the within-family drilldown` (G01–G11; default G05).
- For the selected family:
  1. **Subset MSS UMAP** — dim grey background = other families; selected family points are full-colour and larger.
  2. **Analyte table** with columns: `analyte | biochemical class | top anchors | support bands | main competitors`. Anchors come from `top_anchors_for_class()`; support bands and competitors are parsed from the MSS-signature row matching the analyte's broad class.
  3. **Most-frequent bands bar chart** — built by `family_band_frequencies()` (new helper) from the MSS-signature anchors+supports of all broad classes that map to this family.

### G · Axis overlap matrix (NEW default)

`render_axis_overlap_matrix()` — replaces the network as the default.

- 11×11 symmetric heatmap from `axis_overlap_matrix()` (new helper). z = # bands ≥ threshold shared between axis pairs; diagonal zeroed.
- Cell text annotations only above an inner 60-th-percentile threshold to keep it readable.
- Side table of top 10 axis-pair overlaps with curated `interpretation` (G01-G02 purine, G04-G05 phosphate/glycan, G06-G07 Phe, G06-G08 amide/lipid, G08-G09 lipid/sterol, …) and `risk` (HIGH / MODERATE / LOW).
- The network view is still available — but now lives inside `Show network view (manual chemistry-grouped layout)` expander, default closed, with only the top-8 strongest edges drawn.

### H · Hierarchical clustering support (expander, unchanged structure)

- Collapsed by default. Pre-rendered MSS + Motif dendrograms inside, plus a numbered-callout interpretation panel and the per-cluster summary table with a `caveat` column derived from purity (`high purity` / `moderate — within-family chemistry mixed` / `low purity — shared chemistry across families`).

## Files changed / created (v2 only)

| file | change |
|---|---|
| `streamlit_apps/gaira_command_center_v2/components/motif_mss_bsv_tab.py` | full rewrite — pass-2 section order with new render_* functions |
| `streamlit_apps/gaira_command_center_v2/utils/layout_constants.py` | NEW — `AXIS_POSITIONS`, `AXIS_INFO`, `MAJOR_CLASS_LABELS`, `CANONICAL_BANDS` |
| `streamlit_apps/gaira_command_center_v2/utils/bsv_saliency_utils.py` | extended — `axis_overlap_matrix()`, `family_band_frequencies()`, internal `_parse_band_field()` |
| `streamlit_apps/gaira_command_center_v2/BUILD_REPORT_v2_TAB2_REDESIGN_PASS2.md` | NEW (this file) |

`v1/` was not touched. GAIRA core was not touched.

## Why family-first map is now clearer

The previous pass tried to derive family separation from the UMAP itself, which produced messy overlapping ellipses (236 analytes × 11 families on a 2-D embedding will always overlap — that is biochemical reality). Pass 2 inverts the strategy:

- **Family layout is curated, not learned.** The 11 family centroids are placed in chemistry-grouped quadrants. Reviewers see the taxonomy first, before any embedding evidence.
- **Member analytes scatter around their own centroid.** They visibly belong to that family. There is no fake overlap, no fake separation — the figure is honest about what is being shown ("centroid positions are chemistry-grouped, not UMAP-derived").
- **Hover preserves analyte truth.** Each member point still carries its analyte id, biochemical class, BSV family, and top anchor bands.
- **The actual MSS UMAP is one section down (C)** — that is where the viewer sees the *unsupervised* structure with full overlap honesty. The family map answers *"what are the 11 axes?"*; the MSS map answers *"how do real analytes arrange themselves under MSS?"*.

## What remains approximate

- The family-map centroid positions are a curated layout, not data-driven. They prioritise readability over geometric truth (and the figure label says so explicitly).
- The MSS-within-family drilldown's "main competitors" come from the MSS-signature row matched on broad class; analytes within the same broad class share the same competitor list. A truly per-analyte competitor list would need analyte-resolved MSS scoring (not in current artifacts).
- The dendrograms are still pre-rendered PNGs.
- `family_band_frequencies()` aggregates anchor + support across MSS-signature rows that map to a family by broad class. It does not weight by per-analyte presence (each broad-class signature contributes its bands once regardless of how many analytes share that class).

## Acceptance check

| criterion | result |
|---|---|
| v1 unchanged | ✅ no edits in `gaira_command_center_v1/` |
| v2 launches | ✅ `streamlit run` returns HTTP 200 on `/` and `/_stcore/health` (port 8766); no errors / tracebacks in log |
| Tab 2 starts with motif/11-family map | ✅ section B renders the family-first cluster constellation immediately after section A's hierarchy diagram |
| MSS cluster map secondary and clearer | ✅ section C, single full-width Plotly figure, only ~10 major classes labelled, legend collapsed by default |
| MSS-within-family dropdown works | ✅ sidebar dropdown `BSV family for the within-family drilldown` (default G05); section D shows the subset UMAP + analyte table + bands bar chart |
| Saliency map still good | ✅ section E unchanged from pass 1 (canonical labels OFF default; per-axis table collapsed) |
| Overlap matrix default; network optional | ✅ section G renders the 11×11 heatmap by default; network is in `Show network view` expander |
| No unreadable family hull plot | ✅ removed entirely; faint hull is opt-in per-family in the family-first map |
| No random G05 table unless selected | ✅ removed from default; only appears as the user-selected drilldown family |

## How to run (both versions)

```bash
streamlit run streamlit_apps/gaira_command_center_v1/app.py    # stable
streamlit run streamlit_apps/gaira_command_center_v2/app.py    # pass-2 redesign
```

## Next step

**Tab 3 — Grounding molecule explorer.** The Tab 3 link card already calls
this out: per-analyte spectrum trace + anchor / support / anti-evidence overlays
+ BSV radar + MSS row + per-analyte calibration check. The `D · MSS-within-family
drilldown` from this pass is a useful preview of the data scope Tab 3 will own.

## Strict invariants preserved

- GAIRA core unchanged.
- No GAIRA scoring rerun.
- All loads cached via `@st.cache_data`.
- Missing-artifact tolerance: every loader returns `None` and the renderer surfaces a soft `gaira-warn` card.
- All visualisation-only computations (centroid layout, member jitter, overlap matrix, band frequencies) are explicitly marked in the code and the figure titles.
