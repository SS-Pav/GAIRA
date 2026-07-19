# GAIRA Substrate & Modality Layer Audit

**Date:** 2026-07 · Full table: `data_audit/substrate_physics_rules.csv`.

## Two parallel, non-overlapping stacks
### A. Demo stack (WIRED into V3.1, live) — heuristic, unvalidated
`gaira_demo_reasoning_v3_1/gaira_core/substrate_physics.py` — **5 hard-coded rules**:
| rule_id | substrate | target | op | mult | classification |
| --- | --- | --- | --- | --- | --- |
| ag_sers_purine_amplify | Ag SERS | motif 720–735 / G01 | multiply | **0.65** | heuristic multiplier |
| ag_sers_thiol_amplify | Ag SERS | motif 490–500 / G10 | multiply | **1.20** | heuristic multiplier |
| ag_sers_carotenoid_overlap | Ag SERS | G02 | caveat | 0.85 | caveat generator |
| raman_amide_i_lipid_overlap | Raman | G08 | caveat | 0.92 | caveat generator |
| raman_amide_protein | Raman | G06 | none | 1.0 | metadata-only no-op |

Only the two motif-level rules change BSV numbers; the axis-level rules surface as caveats. **None are empirically calibrated** — the "scientific_basis" is prose.
- **Diabetes override** (`analysis/_diabetes_overrides.py`, opt-in): G10 window tighten 480–510→490–505 (spectral mask) + **co-band-gated thiol boost** (×1.20 only if 720 cm⁻¹ imidazole ≥0.010). The most defensible substrate rule in the codebase (evidence-gating), but only in the diabetes reproduction path.
- **domain_context.py** (EV/serum): ranking weights only (G08/G09→0.7 EV; G02→0.7 serum) + caveats; explicitly never alters coordinates.

### B. Production stack (`src/gaira/substrate/`, `src/gaira/atlas/`) — rigorous but DORMANT
- **Substrate Engine v1.1.1**: 8 effect types, **42 seed effects / 47-row evidence registry** (on `/Volumes/SSD_Rad/GAIRA_BUILD`), multipliers bounded **[0.40, 1.15]**, conflict-aware (CONVERGED 22 / EMERGING 20 / CONFLICTING 2 / INSUFFICIENT 3), families Ag-colloid 28 / Au-colloid 7 / others. Provenance + `merge_registries` + conflict reports.
- **Atlas loader**: band constraints / companion bands / ambiguity rules (GAIRA_BUILD phase4 YAMLs).
- **`config/spectral_anchor_windows_v1.csv`**: 64 source-backed anchor windows (anchor 13 / secondary 25 / ambiguous 26).
- **Critical: NO file under `src/gaira/` imports `gaira.substrate` or `gaira.atlas`.** These engines are loadable but **imported by nothing** — dormant, unused by both the runtime inference and the demo.

## Precise terminology (per the demo stack)
- ag_sers_purine_amplify / ag_sers_thiol_amplify → **substrate-aware heuristic multipliers** (not validated corrections).
- carotenoid / amide-I overlap → **substrate sensitivity flags** (caveat generators; no numeric effect).
- domain_context weights → **substrate/domain-specific evidence weighting** (ranking only).
- The production engine → **conflict-aware evidence-reranking engine** (the only source-backed, bounded, provenance-tracked layer — but unused).

## Modality (Raman vs SERS) awareness
Distinguished ONLY by the `substrate` string ("Ag colloid SERS" vs "Raman") threaded through `build_report`. **No excitation-wavelength awareness; no Au vs Ag distinction; no colloid vs planar distinction.** The European 4-substrate/2-laser adenine dataset exposes this blindness directly (see ablation report). Modality handling is a **coarse binary flag**, not a modality-specific observation model.
