# Phase C1.5 — Summary

## What Changed

### Backend Fixes (scoring.py)
1. **Zero-comparator handling**: Themes/motifs with 0 comparator support are labeled "associated" with `comparator_absent` flag, NOT "enriched". Confidence forced LOW.
2. **Insufficient comparator**: Themes with < 2 comparator direct support cannot be labeled "enriched" — flagged as `insufficient_comparator`.
3. **Minimum query support**: Themes with < 3 query direct support cannot claim enrichment regardless of ratio.
4. **Normalized enrichment**: `norm_enrichment_ratio = (query_direct/total_query) / (comp_direct/total_comp)`. Both raw and normalized shown.
5. **Dual threshold for enrichment**: Requires BOTH `norm_enrich >= 2.0` AND `raw_enrich >= 1.5`.
6. **Evidence balance metric**: `2*min(Q,C)/(Q+C)`. Triggers caveat when < 0.3.
7. **Motif coverage flags**: Motifs with `comparator_absent` or `insufficient_comparator` are flagged.

### Templates (templates.py)
- Normalized enrichment column in comparative tables
- Evidence balance column
- Coverage flag display
- Comparator summary includes balance metric

### Versioned Apps
- `app/gaira_query_demo_C14.py` — preserved C1.4 snapshot
- `app/gaira_query_demo_C15.py` — new C1.5 demo
- `app/gaira_query_demo.py` — main pointer (copy of C1.5)

### Test Suite
14 backend tests, all passing. Validates zero-comparator fix, normalization, balance, sparse comparator warnings, and adequate comparator enrichment.

## Files Updated (4)
- `graph/phaseC1_scoring.py` — v1.5 with all fixes
- `graph/phaseC1_templates.py` — v1.5 formatting
- `app/gaira_query_demo.py` — main pointer to C1.5
- `app/gaira_query_demo_C14.py` — preserved prior version

## Files Created (8)
- `app/gaira_query_demo_C15.py` — versioned C1.5 demo
- `reports/phaseC15_backend_test_matrix.csv` — 14-test results
- `reports/phaseC15_backend_validation_summary.md` — test narrative
- `docs/phaseC15_version_lineage.md` — app version documentation
- `docs/phaseC15_summary.md` — this file
- `graph/phaseC15_normalization_logic.md` — normalization formula + rationale
- `graph/phaseC15_comparator_warning_rules.md` — coverage flags + caveats

## C1.4 vs C1.5 Key Differences

| Behavior | C1.4 | C1.5 |
|---|---|---|
| Theme with 20 query / 0 comparator | "enriched" | "associated" [comparator_absent] |
| Enrichment ratio 10x from 2 vs 0.2 | Shown as "enriched" | Flagged as insufficient_comparator |
| Normalized enrichment | Not computed | Computed and displayed |
| Evidence balance | Not tracked | Computed, triggers caveat < 0.3 |
| Backend tests | None | 14 tests, all passing |
| Versioned apps | Single file | C14 + C15 preserved |
