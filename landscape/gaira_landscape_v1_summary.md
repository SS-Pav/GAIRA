# GAIRA Landscape v1 — Summary

## Coverage Overview

| Metric | Value |
|---|---|
| Total conditions | 59 |
| Conditions with evidence | 23 (39%) |
| Conditions with zero evidence | 36 (61%) |
| Low coverage (<20 rows) | 52 (88%) |
| Well-supported (>=20 rows) | 7 |

### Well-Supported Conditions (>=20 evidence rows)
1. liver_cancer_unspecified (95 rows, 10 sources)
2. reference_method (83 rows, 9 sources)
3. hepatitis (70 rows, 2 sources)
4. cancer_stem_cell (67 rows, 1 source)
5. HCC (42 rows, 3 sources)
6. environmental_toxicology (35 rows, 1 source)
7. NAFLD_NASH (24 rows, 2 sources)

### Critical Observation
Only **HCC** and **NAFLD_NASH** have both sufficient evidence AND multiple independent sources for disease-focused liver analysis. All other liver conditions (fibrosis, DILI, hepatitis) are either single-source or very sparse.

## Cirrhosis Audit
**Cirrhosis is NOT directly present as a condition.** The closest match is "fibrosis" with only 2 evidence rows. The word "cirrhosis" does not appear in any evidence text. This is a significant gap for a liver-focused system.

## Condition Diversity
The condition space is heavily fragmented:
- 27 communities detected (cosine > 0.5 threshold)
- Most conditions are ISOLATED (16 singleton communities)
- Only 4 communities have 3+ conditions sharing motif profiles

### Nearest Neighbor Pairs
| Condition | Nearest Neighbor | Similarity |
|---|---|---|
| acetaminophen_hepatotoxicity | untreated_control | 1.000 (same source) |
| cancer_exosome_diagnosis | multicancer_exosome_diagnosis | 1.000 (same source) |
| HCC | NAFLD_NASH | 0.431 |
| bacterial_identification | cancer_stem_cell | 0.586 |

The HCC-NAFLD similarity (0.43) confirms they share substantial biochemistry — consistent with both being liver diseases affecting similar metabolic pathways.

## Key Clustering Observations

### Disease separation IS partially visible
- HCC, NAFLD, and liver_cancer_unspecified cluster near each other (expected — liver disease family)
- Bacterial_identification is distinct (separate Raman fingerprint)
- Cancer conditions share a broad biochemical signature dominated by protein+lipid

### Disease separation is NOT strong enough for confident discrimination
- Most conditions are singletons in the community structure
- The dominant signal is "serum baseline" (phenylalanine + protein + lipid), which all serum conditions share
- Condition-specific signals (purine, glycan, redox) are present but sparse

## Major Biochemical Axes

| Axis | Motifs | Sources | Assessment |
|---|---|---|---|
| Aromatic amino acid | Phe, Tyr, Trp | 31 | Biology-driven, ubiquitous |
| Protein / amide | protein, collagen, amide I | 28 | Biology-driven, ubiquitous |
| Lipid / membrane | lipid, cholesterol | 31 | Biology-driven, weak differentiator |
| Purine nucleotide | adenine, guanine, AMP | 15 | Biology-driven, potentially discriminative |
| Glycan carbohydrate | glucose, galactose | 10 | Biology-driven, sparse but important |
| Redox metabolite | glutathione, ergothioneine | 8 | Biology-driven, sparse |

## Limitations

1. **Single-source bias**: 8 of 23 evidenced conditions come from a single source. Cross-study validation is limited.
2. **Evidence is descriptive, not differential**: Most evidence says "peak X means molecule Y" — NOT "peak X goes up in disease Z."
3. **Serum dominance**: 70% of evidence is from serum. Non-serum conditions are severely underrepresented.
4. **Reference/method contamination**: 83 evidence rows (4.4%) are from reference methods with no disease context, inflating some motif profiles.
5. **Cirrhosis gap**: A major liver condition is essentially absent.
6. **No spectral intensity data**: BSV is built from literature assignment counts, not from actual spectral measurements. This limits discriminative power.

## What GAIRA Actually Knows
GAIRA knows what **biochemical features are associated with which conditions** based on literature peak assignments. It knows the shared serum baseline (aromatic AA + protein + lipid) and can detect condition-specific enrichments in purine, glycan, and redox components.

## What GAIRA Does NOT Know
- **Direction of change** (increased/decreased) for most peaks
- **Effect sizes** or statistical significance
- **Cirrhosis/fibrosis staging**
- **Non-serum biochemistry** (sparse)
- **Whether associations are causal or correlative**

## Whether Meaningful Structure Exists
**Yes, partially.** The aromatic AA hub, purine axis, and glycan axis are biologically real. The liver disease cluster (HCC + NAFLD + liver_cancer) is detectable. But the signal-to-noise ratio is low, and most conditions are too sparse for confident per-condition claims.
