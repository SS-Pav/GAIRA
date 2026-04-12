# Phase C1.6 — Comparator Scope Policy

## Scope Modes

| Mode | What It Means | When Used |
|---|---|---|
| `broad` | All evidence, no filtering | Default; no sample/domain specified |
| `same_matrix` | Filter to same sample type | "vs healthy serum", "in serum" |
| `same_matrix_domain` | Filter to same sample type AND domain sources | "within liver sources" |
| `all_sources_same_matrix` | Same sample type, all source domains | "across all serum sources" |
| `same_domain` | Domain-filtered only (no sample) | "within liver sources" (no sample) |

## Implementation
- Sample-type filtering is done via Cypher: `MATCH (e)-[:FROM_SAMPLE_TYPE]->(st:SampleType) WHERE st.name = $sample`
- Domain filtering is done via source name matching: `WHERE src.name CONTAINS $domain`
- Both are injected into the condition query builder at graph-query time

## Fallback
If scoped query returns < 3 evidence rows:
1. Scope is broadened (filters removed)
2. `scope_fallback` field records what happened
3. UI displays a warning
4. Results are from the broader scope, not the original narrow scope

## Comparator Gets Same Scope
When a pairwise comparison specifies a scope (e.g., "serum within liver"), the same scope is applied to BOTH the query condition AND the comparator condition. This ensures like-for-like comparison.

## What Is NOT Done
- No explicit sample-type matching validation (e.g., verifying both conditions have serum evidence)
- No cross-matrix comparison warning (comparing serum vs tissue)
- These are future enhancements
