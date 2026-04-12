# Phase C1.3 — Example Outputs

## Example 1: Single — "What does GAIRA know about HCC?"

**Mode**: ASSOCIATIVE

### Top Themes
| Theme | Score | Conf. | Direct | Sources |
|---|---|---|---|---|
| protein (broad) | 12.3 | MED | 40 | 8 |
| lipid (broad) | 10.8 | MED | 30 | 6 |
| nucleic acid | 8.5 | HIGH | 25 | 5 |

### Top Motifs
| Subfamily | Members | Status |
|---|---|---|
| tryptophan | 15 | query associated |
| lipid | 42 | broadly shared |

---

## Example 2: Pairwise — "Compare HCC vs healthy control"

**Mode**: COMPARATIVE (HCC vs healthy control)

### Enriched in HCC
| Theme | Score | Enrich. | Direct | Sources |
|---|---|---|---|---|
| nucleic acid | 9.2 | 3.2x | 25 | 5 |
| carbohydrate | 5.8 | 2.5x | 12 | 3 |

### Associated (not clearly enriched)
| Theme | Score | Enrich. | Direct |
|---|---|---|---|
| lipid (broad) | 7.5 | 1.4x | 30 |

### Shared with healthy control
| Theme | Score | Enrich. |
|---|---|---|
| protein (broad) | 11.0 | 1.1x |

### Interpretation
Nucleic acid and carbohydrate themes show enrichment in HCC vs healthy control (3.2x and 2.5x respectively). Protein is shared — present in both conditions at similar levels. This is consistent with known HCC biochemistry where nucleic acid changes (DNA/RNA metabolism) and glycan alterations are more disease-specific than protein backbone changes.

---

## Example 3: One-vs-Rest — "What is enriched in HCC vs rest?"

**Mode**: ENRICHMENT (HCC vs all other conditions)

Similar output to pairwise, but comparator is all non-HCC conditions aggregated. Enrichment ratios will be lower because the comparator pool is larger and more diverse.

---

## Example 4: Peak — "What does peak 1005 mean?"

**Mode**: ASSOCIATIVE (peak query — no comparative mode)

### Top Themes
| Theme | Score | Conf. |
|---|---|---|
| protein | 8.5 | HIGH |

### Top Biomolecules
phenylalanine (high evidence, multi-source)

Peak 1005 cm-1 is phenylalanine ring breathing. No comparative mode for peak queries.
