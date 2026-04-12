# Phase C1.4 — Comparator Resolution + Absolute Support + Coverage-Aware Fixes

## What Changed

### 1. Absolute Support Display
Theme tables now show:
- `Q direct` / `Q src` — query-side evidence count and source count
- `C direct` / `C src` — comparator-side evidence count and source count
- `Enrich` — enrichment ratio with absolute basis visible

Users can now tell whether "3.2x enriched" means "32 vs 10" or "3 vs 1".

### 2. Comparator Summary Block
New UI section for comparative queries showing:
- Query condition + evidence count + source count
- Comparator condition + evidence count + source count
- Inferred sample type
- Comparator adequacy (ADEQUATE / SPARSE / VERY SPARSE)

### 3. Coverage-Aware Confidence
- Themes with `low_comparator` flag get smaller enrichment boost (1.1x vs 1.3x)
- Themes with `low_query_support` (< 3 direct) cannot claim "enriched"
- High confidence is capped at medium when comparator is sparse
- Explicit caveats for sparse comparator and unstable enrichment

### 4. Sample Type Inference
- Dominant sample type inferred from retrieved evidence
- Reported in comparator summary and query understanding
- Not yet filterable at query time (future enhancement)

### 5. One-vs-Rest Background Query
- Proper Cypher query that excludes the query condition from the background pool
- Retrieves up to 1000 background evidence rows for enrichment comparison

## Files Updated (5)
| File | Changes |
|---|---|
| `graph/phaseC1_query_engine.py` | Sample type in Cypher, comparator stats, OVR background query |
| `graph/phaseC1_scoring.py` | Coverage thresholds, absolute counts, adequacy assessment, coverage flags |
| `graph/phaseC1_templates.py` | Absolute support columns, comparator summary section, coverage flags |
| `app/gaira_query_demo.py` | Comparator summary block, absolute support columns, adequacy metric |
| `app/graph_preview.py` | (unchanged structurally; contrast is in tables not graph) |

## Files Created (5)
| File | Content |
|---|---|
| `docs/phaseC14_comparator_policy.md` | How comparators are resolved |
| `docs/phaseC14_sample_type_resolution.md` | Sample type inference logic |
| `docs/phaseC14_summary.md` | This summary |
| `graph/phaseC14_coverage_confidence_rules.md` | Thresholds and coverage flags |
| `graph/phaseC14_graph_contrast_design.md` | Why graph preview shows query-side only |

## Preserved
- All deterministic reasoning (no LLM)
- Direct vs inferred edge distinction
- Scientific caution in caveats
- Backward compatibility with single-condition queries
