# GAIRA V5 — Phase 1.5 Grounding Completion Report (785 nm)

**Date:** 2026-07-18 · Branch `gaira-v5-rebuild-plan` · Governs: `GAIRA_V5_REBUILD_PLAN.md` §5.5 (Phase 1.5). Notebook: `results/v5_rebuild/phase1_5/`. Hypotheses: `GAIRA_V5_HYPOTHESIS_REGISTER.md`.

> **Decision: Phase 1.5 succeeded. Phase 2 (Canonical Representation Discovery) is now scientifically justified. STOP here; do not begin representation discovery until instructed.**

> **[2026-07-18 correction, applied during Phase 2 §4 input audit]** The 6 adenine bAgNPs Ag-SERS spectra were re-examined and found to be a **controlled concentration series** (10 pg → 10 µg/mL), i.e. controlled-perturbation *evaluation* data, not independent molecular grounding. They are therefore **excluded from representation fitting** (data-role separation, Phase-2 §3). Adenine remains a matched grounding analyte via the Gobbato Raman + Ag-SERS references, so **the matched analyte count (51) and unique analyte count (87) are unchanged**. Corrected corpus totals for representation: **485 → 479 spectra; 271 → 265 Ag-SERS spectra** (214 Raman unchanged). The corpus-composition and exclusion tables below reflect the original Phase 1.5 assembly; the authoritative Phase-2 input is `results/v5_rebuild/phase2_stage_a/tables/phase2_input_manifest.csv`.

## What was built (canonical, in `src/gaira`)
- `src/gaira/data/gobbato.py` — loader for the Gobbato pure-metabolite corpus (B&WTek 785 nm; parses Raman-Shift + Dark-Subtracted columns; extracts the zip to a temp cache, never into source).
- `src/gaira/data/synonyms.py` — abbreviation/synonym/salt/hydrate reconciler → canonical analyte names.
- Phase 1.5 registries: `canonical_analyte_registry_v5.csv`, `grounding_spectrum_registry_785.csv`, `phase1_5_grounding_summary.json` (`results/v5_rebuild/phase1_5/tables/`).
No PCA / clustering / NMF / embeddings / ontology / observation model / BSV / MSS (correctly deferred).

## The seven gate questions
1. **How many unique analytes now exist?** **87** (785 nm, canonical, de-duplicated across synonyms/salts).
2. **How many 785 Raman spectra?** **214** (RamanBioLib 785-nm subset ≈61 + Gobbato pure Raman powders 153).
3. **How many 785 Ag-SERS spectra?** **271** (adenine bAgNPs 6 + Gobbato pure Ag-SERS 265).
4. **How many analytes exist in both?** **51** matched (785 Raman ∩ 785 Ag-SERS).
5. **What % of the corpus is now matched?** **58.6%** of the 87 analytes (up from 7/… in Phase 1 — a ~7× increase).
6. **Is this sufficient to begin representation discovery?** **Yes.** 51 chemically-broad, replicated (Raman ~3, Ag-SERS ~5 per analyte) matched analytes + 485 spectra across 87 analytes is a defensible basis to test whether stable biochemical structure emerges. The Phase-1 blocker (7 matched) is resolved.
7. **If not, what data are still missing?** (Sufficient to proceed, but note remaining gaps for later multi-substrate work): **zero Au-SERS grounding**; the Ag-SERS side has only two sources (Gobbato colloid + adenine bAgNPs); matched pairs are strongest within Gobbato (single instrument); non-785 RamanBioLib (141) and 633-nm metabolite-63 (63) are indexed-but-excluded and could support future multi-excitation observation models.

**8. Should the next phase be Canonical Representation Discovery or another data-acquisition phase?** → **Canonical Representation Discovery** (Phase 2), Stage A (direct spectra). Treat H1 (shared representation exists) as a hypothesis to test, not assume; keep modality-stratified representation as the fallback.

## Corpus composition & exclusions
| Included (785 nm, → representation) | spectra | analytes |
| --- | --- | --- |
| RamanBioLib 785 (Raman) | ~61 | ~61 |
| Gobbato pure Raman powders | 153 | 51 |
| Gobbato pure Ag-SERS | 265 | 53 |
| adenine bAgNPs (Ag-SERS) | 6 | 1 |
| **Total** | **485** | **87 unique** |

| Excluded (228) | reason |
| --- | --- |
| RamanBioLib 532/1064/488/514.5/… (141) | non-785 (indexed, kept for future multi-excitation work) |
| metabolite-63 (63) | 633 nm, not 785 |
| ORC-Ag (24) | peak-only (kept for MSS, not representation) |

## Hypothesis outcomes
- **H1a** (enough matched analytes to attempt cross-mode analysis): **Supported** (51 matched, 58.6%).
- **H1** (shared Raman/Ag-SERS representation exists): **now testable** (was untestable at 7); to be tested — not assumed — in Phase 2.

## Recommendation
Proceed to **Phase 2 — Canonical Representation Discovery** on the completed 785 nm corpus, **Stage A (direct spectra) first**. Do not build an observation model until Stage A/B evidence shows whether a shared space is warranted or modality-stratified representations are required. Await instruction before beginning.
