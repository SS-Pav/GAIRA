# GAIRA V4 — Architecture Truth Audit

**Date:** 2026-07-18 · Read-only code trace (src/gaira + demo + docs + UMAP scripts).

## The claim under test
> pure grounding spectra → emergent UMAP clusters → learned biochemical axes → project new spectra → BSV → MSS confidence

**Verdict: FALSE.** GAIRA's axes are **hard-coded curated constants**; UMAP only visualizes precomputed vectors; BSV is **deterministic band/motif scoring**; the learned encoder is **non-load-bearing**. This matches the design docs, which explicitly reject an embedding-first / emergent-cluster foundation.

## Concept-by-concept
| Concept | Intended (docs) | Demo | Production (src/gaira) | Scientific status | Correction |
| --- | --- | --- | --- | --- | --- |
| 11 axes | curated candidate pool, pruned by evidence; NOT embedding labels | `BSV_AXES` hard-coded (`config.py:73`) | `BIOLOGY_AXES_V11` hard-coded (`base2/schema.py:18`, "from axis design doc §2") | **curated** | stop describing as emergent |
| UMAP → axes | rejected as foundation (core-concepts §13) | "themes painted onto clusters afterward" (`demo/ev_latent_map.py:59`) | **not in inference path at all** | **visualization/clustering only** | UMAP defines nothing |
| 8→11 | evolving curated list | `LEGACY8_TO_V11` display-only remap | `PROJECTION_V11_TO_V8` locked hand-authored map (`schema.py:48`) | **inherited/curated regrouping** (6 of 11 are split children) | not a data split |
| BSV | grounded band evidence, not latent | `project_to_bsv` motif noisy-OR + small MSS push | `score_spectrum` motif→axis noisy-OR (band window-max, fixed weights) | **deterministic** | not a learned projection |
| MSS | interpretable band signatures | runs BEFORE BSV; +0.25·f·w numeric push | separate base3 stack; NOT in base2 BSV | **data-derived but deterministic** | in demo MSS is a minor contributor, motif-dominant |
| Retrieval (grounding_search) | supporting evidence only | not a BSV input | retrieval+rerank+themes; **emits no `bsv`** | **deterministic support layer** | retrieval never sets BSV |
| Disease → axes | forbidden (`gaira-base.md:160`) | context/routing only | context/routing only | **none found** | axes are disease-independent |
| Learned CNN encoder | "later layer, after scaffold" | offline embeddings for plots | trained but **unused at inference** | **learned but non-load-bearing** | not part of BSV |

## Answers to the 10 explicit questions
1. Axes learned from UMAP clusters? **No.**
2. Manually curated? **Yes** (hard-coded tuples from the axis design doc).
3. Inherited from an 8-axis ontology? **Yes** (curated 8-axis + hand-authored 11↔8 map).
4. Created by proportional splitting? **Yes for 6 of 11** (purine→nuc/met, lipid→acyl/sterol, redox→thiol/metabolite).
5. Does UMAP define axes or only visualize? **Only visualize** precomputed BSV/embeddings.
6. Does production project into a learned latent space? **No** — deterministic motif scoring.
7. MSS before/during/after BSV? **Before** (demo, small numeric push); **absent** from production base2 BSV.
8. Does retrieval influence BSV? **No.**
9. Does disease data influence axis definitions? **No** (forbidden by design).
10. Learned vs deterministic? Learned = the CNN encoder (unused at inference) + UMAP (viz only). Everything defining axes/BSV/MSS/retrieval is **deterministic/curated**.

## Consequence
GAIRA is a **curated, deterministic, grounding-scored biochemical evidence system**, not an emergent latent model. Any prior text implying axes "emerged from UMAP" is incorrect and is superseded (see `GAIRA_CURRENT_STATE_AND_ARCHITECTURE_V4.md`).
