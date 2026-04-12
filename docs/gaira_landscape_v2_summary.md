# GAIRA Landscape v2 — Summary

## 1. What is well-supported enough for meaningful interpretation?

| Condition | Evidence | Sources | Tier | Usable? |
|---|---|---|---|---|
| liver_cancer_unspecified | 95 | 10 | strong | Yes |
| reference_method | 83 | 9 | strong | Yes (baseline only) |
| hepatitis | 70 | 2 | moderate | With caution (2 sources) |
| cancer_stem_cell | 67 | 1 | weak (single-source) | High bias risk |
| HCC | 42 | 3 | strong | Yes |
| NAFLD_NASH | 24 | 2 | moderate | With caution |

**Only HCC and liver_cancer_unspecified have both sufficient evidence AND multi-source validation.**

## 2. Which conditions are trustworthy for comparison?

| Comparison | Feasible? | Confidence |
|---|---|---|
| HCC vs healthy_control | Yes | Moderate (limited healthy-liver overlap) |
| NAFLD vs healthy_control | Marginal | Low (24 rows, 2 sources) |
| HCC vs NAFLD | Marginal | Low (cosine 0.43 — similar but sparse comparator) |
| bacterial vs healthy | Yes | Moderate (distinct biochemistry) |

## 3. Most robust biochemical neighborhoods

| Neighborhood | Support | Biology-Driven? |
|---|---|---|
| Aromatic amino acid (Phe/Tyr/Trp) | Strong (31+ sources) | Yes — fundamental SERS markers |
| Protein / amide backbone | Strong (28+ sources) | Yes — ubiquitous |
| Lipid / membrane | Strong (31+ sources) | Yes — but weak differentiator |
| Purine nucleotide | Moderate (15 sources) | Yes — potentially discriminative |
| Glycan carbohydrate | Sparse (10 sources) | Yes — liver-relevant but thin |
| Redox metabolite | Sparse (8 sources) | Yes — but exploratory |

## 4. Does BSV reveal more structure than raw motifs?

**Partially.** The BSV heatmap shows that most conditions share high protein_backbone and membrane_lipid scores (the serum baseline). The differentiating components (purine, glycan, redox) are sparser and noisier. BSV does compress the 181-motif space into 8 interpretable dimensions, but the dominant signal is still shared background.

## 5. Is disease separation meaningfully visible?

**Barely.** The dendrogram shows:
- Bacterial identification IS separated (distinct chemistry)
- HCC and NAFLD ARE neighbors (shared liver biology)
- Most other conditions are too sparse to cluster confidently
- The serum baseline (Phe + protein + lipid) dominates, making condition-specific signals hard to isolate

## 6. What next move is most justified?

**Ranked by expected impact:**

1. **Targeted cirrhosis/fibrosis literature ingestion** — the biggest gap for liver-focused analysis
2. **Dataset integration** — actual spectral intensity data would transform BSV from count-based to measurement-based
3. **Source-bias correction** — 14 single-source conditions need cross-validation before claims
4. **BSV refinement** — component weights could be tuned using spectral data as ground truth
5. **Condition-family expansion** — more HCC/NAFLD/fibrosis sources would strengthen the most useful comparisons

## Source Bias Assessment
- **14 conditions from single sources** — these cannot be trusted for cross-study claims
- **Source-driven clustering IS present** — some condition neighborhoods reflect which papers happened to be processed, not pure biology
- **Mitigation**: The signal stability layer (C1.7) already flags single-source instability. But landscape-level claims should still be cautious.
