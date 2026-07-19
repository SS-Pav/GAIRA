# GAIRA V4 — Substrate Layer Validation

**Date:** 2026-07-18 · Tests: `audits/v4/substrate_atlas_tests.py` → `data_audit/v4_substrate_atlas_test_results.csv`. Ground truth: European multi-instrument adenine (cAg/cAu/sAg/sAu × 532/785).

## Two layers audited
- Demo 5-rule layer (wired): 2 heuristic multipliers (purine ×0.65, thiol ×1.20) + 3 caveats. Only knows "Ag colloid SERS" vs "Raman".
- Production 42-effect engine (`src/gaira/substrate`, dormant): source-backed, bounded [0.40,1.15], conflict-aware; families Ag-colloid 28 / Au-colloid 7. Evidence-backed but **imported by nothing**.

## Experiment A — cross-substrate purine (G01) identification (adenine)
| Engine | G01 by substrate (cAg/cAu/sAg/sAu) | cross-substrate CV | purine top-1 |
| --- | --- | --- | --- |
| demo + rules | 0.196 / 0.163 / 0.139 / 0.198 | **0.141** | 1.00 |
| demo − rules | 0.290 / 0.242 / 0.203 / 0.295 | 0.146 | 1.00 |
| production | 1.00 / 1.00 / 1.00 / 1.00 (saturated) | 0.00 | 1.00 |

**Finding:** every engine identifies purine as top-1 for adenine on ALL four substrates — the **analyte signal, not the substrate layer, drives correct identification**. The demo substrate rules change magnitude but give **no cross-substrate CV benefit** (0.141 vs 0.146). None of the engines has an Au/planar/excitation model, so cross-substrate *magnitude* cannot be corrected.

## Experiment C — rule ablation (target rank / off-target / cross-substrate)
- Target family rank: unchanged (purine stays top-1) with rules, without rules, and in production.
- Off-target suppression: substrate rules touch only the purine motif; off-target axes unchanged (prior ablation).
- Cross-substrate consistency: not improved by any rule layer.

## Per-layer verdict
| Layer | Utility |
| --- | --- |
| Demo purine dampen (×0.65) | **no detectable utility** (magnitude only) |
| Demo thiol boost (×1.20) | suggestive on ergothioneine; unvalidated |
| Demo caveats (carotenoid/amide) | caveat-only, 0 numeric effect |
| Production 42-effect engine | **not testable in-place** (dormant; no importer) — but it is the only source-backed, bounded, Au-aware layer and is the right target to activate |
| Metadata-only stratification | **recommended** — carry substrate+excitation and compare within-stratum rather than applying universal multipliers |

## Recommendation
Do **not** apply universal substrate multipliers without paired evidence. Adopt **metadata-only substrate/modality stratification** now (Raman / Ag-colloid / Au-colloid / ORC-Ag / unknown-SERS), and **activate the production 42-effect engine** (bounded, source-backed) as the substrate observation model once paired cross-substrate references exist. No demo substrate rule has demonstrated validated utility.
