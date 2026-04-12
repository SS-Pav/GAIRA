# BSV v1 — Output Format

## BSVComparison (top-level)
- `query_bsv`: BSVVector for the query condition
- `comparator_bsv`: BSVVector for the comparator (None for single queries)
- `delta_components`: list of per-component delta dicts

## BSVVector
- `query_condition`: condition name
- `comparator_condition`: comparator name
- `components`: list of BSVComponent
- `total_motifs_used`: sum of motifs across components

## BSVComponent
- `name`: component identifier (e.g., "membrane_lipid")
- `raw_score`: unnormalized accumulation
- `weighted_score`: same as raw (weights applied per-motif)
- `normalized_score`: 0-1 scale (max component = 1.0)
- `contributing_motifs`: top motif subfamilies
- `motif_count`: number of contributing motifs
- `dominant_stability`: most common stability label among contributing motifs
- `coverage_note`: adequate / sparse / absent

## Delta Dict
- `component`: name
- `query_score`: normalized query score
- `comparator_score`: normalized comparator score
- `delta`: query - comparator
- `direction`: "up" / "down" / "flat"
- `query_stability`: dominant stability from query side
