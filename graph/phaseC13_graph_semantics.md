# Phase C1.3 — Graph Preview Semantics

## Layout Hierarchy
1. **Anchor** (largest, center): queried condition or peak
2. **Themes** (inner ring): biochemical themes, sized by evidence count
3. **Motifs** (middle ring): spectral motifs, sized by member count
4. **Biomolecules + FGs** (outer concepts): specific molecules and chemistry
5. **Evidence rows** (periphery): sampled evidence dots

## Comparative Mode
When a comparator is present, the graph preview still shows the query-side subgraph. The comparator enrichment is reflected in the scoring tables, not the graph layout (which would require a dual-anchor layout not yet implemented).

## Explainer Note
The graph preview is a simplified visual of the query traversal. It may emphasize locally coherent clusters (e.g., adenine/guanine co-occurring) that look prominent but may not dominate the global score. The score tables are authoritative; the graph is for structural intuition.

## Edge Semantics
- **Solid purple** (DIRECTLY_SUPPORTS_THEME): direct evidence linking assignment to theme
- **Dashed purple** (INFERRED_SUPPORTS_THEME): mapping-table inference
- **Thick red** (LINKED_TO): condition linkage from motif/neighborhood
- **Green** (PART_OF_MOTIF): evidence membership in a motif
- **Orange** (HAS_FUNCTIONAL_GROUP): chemistry decomposition
