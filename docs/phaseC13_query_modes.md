# Phase C1.3 — Query Modes

## Supported Query Types

### 1. Single Condition (associative)
Returns everything associated with a condition in the GAIRA graph.
- Example: `What does GAIRA know about HCC?`
- Mode: `single`
- Output: top themes, motifs, biomolecules ranked by support + specificity

### 2. Pairwise Comparison (comparative)
Compares two conditions, showing enriched/shared/depleted themes and motifs.
- Example: `Compare HCC vs healthy control`
- Example: `Compare HCC vs NAFLD`
- Mode: `pairwise`
- Output: themes split into enriched / associated / shared / depleted

### 3. One-vs-Rest (enrichment)
Shows what is enriched in a condition compared to all other conditions.
- Example: `What is enriched in HCC vs rest?`
- Example: `What distinguishes fibrosis from others`
- Mode: `one_vs_rest`
- Output: same as pairwise but comparator is "all other conditions"

### 4. Peak / Theme / Chemistry (unchanged)
- `What does peak 1005 mean?`
- `What supports lipid signal?`
- `What links amide I to biology?`

## Routing Logic
1. Check for `vs` keyword -> pairwise if two conditions found
2. Check for "enriched in" / "distinguishes" -> one_vs_rest
3. Otherwise single condition / peak / theme / chemistry
4. Longest condition match first to avoid substring conflicts
