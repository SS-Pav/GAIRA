# D1 — Integration Notes

## Current State
The contrast evidence pilot produced 19 entries stored in `data/contrast/d1_contrast_evidence_pilot.csv`. These are NOT yet integrated into the GAIRA graph or Streamlit demo.

## Integration Hooks

### Loading the Contrast Data
```python
import csv
with open('data/contrast/d1_contrast_evidence_pilot.csv') as f:
    contrast = list(csv.DictReader(f))
```

### Join Keys to Existing Layers
- `matched_peak_node_id` → links to existing `evidence_item_id` in the active evidence CSV
- `source_id` → links to GAIRA source provenance
- `query_condition` / `comparator_condition` → links to condition ontology
- `matched_peak_cm` → links to Peak nodes in the Neo4j graph

### How Directionality Would Complement Motif Differential
The current motif differential layer (C1.7) computes direction from **cross-source evidence count ratios**. Contrast evidence provides **within-source explicitly stated direction**. These complement each other:

| Layer | Source of Truth | Granularity |
|---|---|---|
| Motif differential | Cross-source count ratios | Motif/subfamily level |
| Contrast evidence | Within-source explicit text | Individual peak level |

When both agree (e.g., motif shows "enriched" AND contrast evidence says "increased"), confidence should be boosted. When they disagree, the within-source explicit evidence may be more reliable for that specific source.

### Future Graph Integration
New edges:
- `ContrastEvidence -[SUPPORTS_DIRECTION]-> Peak`
- `ContrastEvidence -[FROM_SOURCE]-> Source`
- `ContrastEvidence -[IN_CONDITION]-> Condition`

These would be `evidence_type = "contrast"` to distinguish from assignment and inferred edges.

### Demo Integration (C1.8 or later)
Add a "Directional Evidence" panel that shows:
- Which peaks have explicit directional support
- Whether the direction agrees with the motif differential
- Confidence level of the directional claim
- Raw source text for transparency

Do NOT replace the motif differential display. Add contrast evidence as a supplementary layer.
