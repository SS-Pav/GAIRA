# Phase C1.6 — Summary

## What Changed

### Query Router (v1.6)
- **New**: `sample_scope` field (serum, plasma, EV, tissue, saliva, urine)
- **New**: `domain_scope` field (liver, cancer, infectious, etc.)
- **New**: `scope_mode` field (broad, same_matrix, same_matrix_domain, all_sources_same_matrix)
- Scope modifiers extracted from query text before condition matching
- Scope mode determined from combination of "within", "across all", sample, domain keywords

### Query Engine (v1.6)
- **New**: `_build_condition_cypher()` generates Cypher with optional sample-type and domain filters
- **New**: Automatic fallback when scoped query returns < 3 rows
- **New**: `scope_fallback` field in GraphResult records fallback decisions
- **New**: Same scope applied to both query and comparator in pairwise mode
- **New**: `scope_mode`, `sample_scope`, `domain_scope` passed through to GraphResult

### Scoring (v1.5 + scope passthrough)
- ComparatorSummary now includes scope_mode, sample_scope, domain_scope, scope_fallback
- No scoring logic changes — scope affects what evidence is retrieved, not how it's scored

### Streamlit App (C1.6)
- **New**: Version header (VERSION = "C1.6")
- **New**: Scope display in query info bar (matrix, domain)
- **New**: Scope details in comparator summary (scope mode, sample, domain, fallback)
- **New**: Scope fallback warning when triggered
- **New**: Scoped query examples in sidebar

### Versioned Apps
- `app/gaira_query_demo_C1_5.py` — preserved C1.5 with VERSION header
- `app/gaira_query_demo_C1_6.py` — new C1.6 with scope-aware comparator

## Files Updated (3 core)
- `graph/phaseC1_query_router.py` — v1.6 with scope parsing
- `graph/phaseC1_query_engine.py` — v1.6 with scoped Cypher + fallback
- `graph/phaseC1_scoring.py` — ComparatorSummary scope fields

## Files Created (6)
- `app/gaira_query_demo_C1_5.py` — preserved snapshot
- `app/gaira_query_demo_C1_6.py` — new versioned app
- `docs/phaseC16_query_scope_grammar.md`
- `docs/phaseC16_comparator_scope_policy.md`
- `docs/phaseC16_summary.md`
- `reports/phaseC16_scope_test_matrix.csv`
- `reports/phaseC16_scope_validation_summary.md`

## Scope Modes Supported

| Query | Scope |
|---|---|
| `Compare HCC vs healthy control` | broad |
| `Compare HCC vs healthy serum` | same_matrix (serum) |
| `Compare HCC vs healthy serum within liver sources` | same_matrix_domain (serum + liver) |
| `Compare HCC vs healthy serum across all serum sources` | all_sources_same_matrix (serum, all domains) |
| `Compare HCC vs NAFLD in serum` | same_matrix (serum) |

## Main Demo Pointer
`app/gaira_query_demo.py` — currently points to C1.5. Update to C1.6 after live validation if desired.
