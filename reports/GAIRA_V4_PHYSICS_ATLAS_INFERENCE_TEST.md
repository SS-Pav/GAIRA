# GAIRA V4 — Physics Atlas Inference Test

**Date:** 2026-07-18 · Current atlas = 8 curated literature prose regions, **UI captions only** (confirmed: caveats generated but BSV numerically unchanged).

## Can atlas entries become testable inference rules?
| Atlas function | Testable? | Current status | V4 assessment |
| --- | --- | --- | --- |
| Band ambiguity flags | yes | prose only | **useful for confidence/caveats** — the Ag-flake `*` exclusive-peak flag + shared-band map could drive a real ambiguity signal |
| Substrate-specific band reliability | yes, but needs paired data | prose only | **not yet testable** (no paired multi-substrate reference panel) |
| Collision penalties (down-weight ambiguous molecular calls) | yes | not implemented | **useful for confidence** — reduces false molecule-level calls; does NOT need to change BSV |
| Mode-specific evidence weighting | yes | not implemented | **not yet testable** (needs modality-stratified references) |
| Unsupported-molecular-call suppression | yes | partial (MSS class-level default) | **useful for caveats** |
| Out-of-distribution warning | yes | not implemented | **useful for confidence** (flag Raman spectra scored on an Ag-SERS scale) |

## Test result (collision / numeric)
- Baseline vs +collision vs +substrate-reliability vs +full atlas: the current atlas is prose, so all four are numerically identical (0 BSV change). A collision layer built from the **Ag-flake exclusive-peak flags + shared-band map** is the concrete next step — it would operate on **confidence/caveats**, not BSV values.

## Per-component assignment
| Component | Verdict |
| --- | --- |
| 8 prose atlas regions | **useful for confidence/caveats** (as-is: not numerically useful) |
| Collision / shared-band handling | **useful for confidence** (prototype from Ag-flake exclusive flags) |
| Substrate band reliability | **not yet testable** (needs paired data) |
| OOD / cross-modality warning | **useful for confidence** (implementable now from substrate metadata) |
| Direct BSV modification | **not useful / not recommended** — the atlas should never invent BSV values |

## Recommendation
Keep the atlas **inference-safe**: it should drive **confidence, caveats, collision down-weighting, and OOD warnings** — never numeric BSV. The most promising testable upgrade is a collision/ambiguity layer seeded by the Ag-flake exclusive-characteristic peak flags and the known shared-band map (720 purine, 1440 lipid, 1003 aromatic, 1517 carotenoid/UA).
