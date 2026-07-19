# GAIRA V4 — Demo ↔ Production Reconciliation

**Date:** 2026-07-18 · Adapter: `audits/v4/reconcile_demo_production.py` → `data_audit/v4_demo_production_comparison.csv`. Both engines run standalone and deterministically.

## Two engines
| | Demo (`gaira_demo_reasoning_v3_1/gaira_core`) | Production (`src/gaira/base2`) |
| --- | --- | --- |
| Entry | `report_builder.build_report` | `base2.score_spectrum(y, master_x, *load_engine())` |
| Motifs | 11 curated | **50** (`MOTIF_REGISTRY_V1_2`), 39 axis mappings, 53 dual-status |
| Axes | G01–G11 (`BSV_AXES`) | `BIOLOGY_AXES_V11` (same 11 chemistry names) + `axis8_projection` |
| Substrate | 5 heuristic rules | regime/dual-status (`AG_COLLOID_SERUM`), bounded status weights |
| Data deps | small bundled CSV/parquet | none to run; DuckDB for retrieval |
| MSS in BSV | yes (small push) | no (base3 separate) |
| Learned encoder | no | trained but unused |

## Comparison packet (representative spectra)
| Case | Demo top axis | Prod top axis | Agree | Pearson | Cosine(shape) | Demo/Prod motifs |
| --- | --- | --- | --- | --- | --- | --- |
| adenine 10 pg/mL | G01 purine | G01 purine | ✅ | — | 0.94 | 11 / 35 |
| adenine 100 pg/mL | G01 | G01 | ✅ | — | 0.87 | 9 / 36 |
| adenine 10 µg/mL | G01 | G01 | ✅ | — | 0.60 | 9 / 36 |
| serum liver CCA | G07 aromatic | G07 aromatic | ✅ | 0.94 | 0.96 | 7 / 13 |
| EV Impact | G10 redox | G08 lipid | ❌ | 0.48 | 0.72 | 6 / 14 |
| synthetic Phe+purine+lipid | G07 | G07 | ✅ | 0.93 | 0.95 | 2 / 4 |

**Agreement:** top-axis matches 5/6 (all reference/serum/synthetic); diverges on the EV mixture. Shape cosine 0.60–0.96. The production engine fires ~3–4× more motifs (richer registry) and saturates purine on strong adenine.

## Recommendation (no refactor this pass)
- **Canonical future inference engine = production `src/gaira/base2` (+ base3/mss_engine).** It is richer (50 motifs, dual-status regime handling, explicit 11↔8 projection), source-backed, and already deterministic — the demo's 11-motif engine is a teaching subset.
- The demo's value is UX + the frozen global-coordinate layer (V3) + the honest provenance framing (V2/V3.1). Keep the demo as the presentation layer; **retire the duplicate demo scoring logic later** by wiring the demo UI onto the base2/base3 engine behind an adapter.
- Do NOT merge engines in this pass. First stabilize: (a) substrate/modality metadata, (b) MSS-as-support (not BSV contributor), (c) a shared axis constant module.
