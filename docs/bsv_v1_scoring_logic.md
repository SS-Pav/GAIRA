# BSV v1 — Scoring Logic

## Per-Motif Contribution
```
base = log(1 + direct_count)         # log-scaled to prevent single-source dominance
source_bonus = min(source_count * 0.3, 2.0)  # diversity reward
stability_weight = {STABLE: 1.0, MIXED: 0.6, UNSTABLE: 0.3, INSUFFICIENT: 0.15}
broad_penalty = 0.5 if broadly-shared else 1.0
comparator_absent_penalty = 0.2 if comparator_absent else 1.0

contribution = (base + source_bonus) * stability_weight * broad_penalty * comp_penalty
```

## Per-Component Accumulation
```
raw_score = sum(contribution * component_weight for each matching motif)
```

## Normalization
```
normalized_score = raw_score / max_raw_score_across_all_components
```
This ensures the strongest component = 1.0 and others are relative.

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Log-scaling of counts | Prevents a single 100-row source from dominating |
| Stability weighting | STABLE motifs count more than MIXED or UNSTABLE |
| Broad penalty (0.5x) | Broadly-shared motifs are downweighted — they don't distinguish conditions |
| Comparator-absent penalty (0.2x) | Motifs without comparator evidence contribute minimally |
| No component for "unresolved" | Unresolved evidence would add noise without biological meaning |

## Delta Computation
For comparative queries:
```
delta = query_normalized - comparator_normalized
direction = "up" if delta > 0.05 else "down" if delta < -0.05 else "flat"
```
