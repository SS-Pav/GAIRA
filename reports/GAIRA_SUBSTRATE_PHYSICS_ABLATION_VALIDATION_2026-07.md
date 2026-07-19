# GAIRA Substrate & Physics Ablation Validation

**Date:** 2026-07 · Harness: `audits/corpus_audit/run_ablation_tests.py` → `data_audit/ablation_results.csv`. Uses the unmodified V3.1 demo engine.

## Results
| Test | Layer | Metric | With layer | Without | Verdict |
| --- | --- | --- | --- | --- | --- |
| adenine dose | substrate_weighting | G01 dose Spearman | 0.829 | 0.829 | **no_detectable_utility** (ordering unaffected) |
| adenine dose | substrate_weighting | G01 max | 0.168 | 0.249 | magnitude effect (dampens, keeps class-level) — not validated |
| adenine dose | substrate_weighting | off-target spillover | 0.0247 | 0.0246 | **no_detectable_utility** (only touches purine motif) |
| European adenine | substrate_weighting | cross-substrate G01 CV | 0.131 | n/a | **not_testable** — demo maps cAg/cAu/sAg/sAu to ONE rule (blind to Au/planar) |
| European adenine | modality/excitation | G01 532 vs 785 | 0.105 | 0.181 | **not_testable** — no excitation awareness |
| adenine high-conc | physics_caveats/collision | caveats change BSV? | no | no | **no_detectable_utility** (caveat generator only) |
| ergothioneine high-dose | substrate_weighting | G10 thiol boost | 0.130 | 0.110 | **suggestive_utility** (heuristic raises redox; not validated vs ground truth) |

## Per-layer verdict
| Layer | Verdict |
| --- | --- |
| Ag-SERS purine dampen (×0.65) | **no_detectable_utility** for ordering/spillover; a magnitude effect only |
| Ag-SERS thiol boost (×1.20) | **suggestive_utility** (raises G10 on ergothioneine) — heuristic, unvalidated |
| Cross-substrate correction | **not_testable_with_current_data** — the demo has no Au/planar model; the European 4-substrate dataset exists but the layer cannot use it |
| Modality/excitation correction | **not_testable** — no 532-vs-785 awareness |
| Physics atlas | **no_detectable_utility** on numbers (UI caveats only) |
| Collision handling | **no_detectable_utility** on numbers (caveats only) |
| Diabetes co-band thiol gate | **suggestive/partially validated** (audit-reasoned; the most defensible rule) |

## Conclusion
The demo substrate layer is a **coarse Ag-colloid-only heuristic**: it changes band magnitudes but does not improve dose ordering, reduce off-target spillover, or provide any cross-substrate/cross-modality stability (it is blind to Au, planar surfaces, and excitation wavelength). Physics caveats and collision handling are **caveat generators with zero numerical effect** on the BSV. **No substrate/physics layer has demonstrated (validated) utility with the current single-substrate calibration data**; the ergothioneine thiol boost and the diabetes co-band gate are *suggestive*. The rich European multi-instrument dataset (4 substrates × 2 lasers × 15 labs) is the obvious validation vehicle but requires a substrate/modality-aware model the demo does not have.
