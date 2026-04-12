# GAIRA Phase C1 — Example Query Outputs

## Example 1: "What does GAIRA know about HCC?"

### A. Query Understanding
- **Type**: Disease/Condition Query
- **Entity**: HCC
- **Evidence rows**: ~100+ from 10+ sources

### B. Grounding
- Motifs matched: ~23
- Themes: protein, lipid, nucleic acid, carbohydrate
- Biomolecules: tryptophan, phenylalanine, tyrosine, adenine, guanine, collagen

### D. Ranked Interpretation
| Theme | Score | Confidence | Direct | Sources |
|---|---|---|---|---|
| protein | ~150 | high | 40+ | 8+ |
| lipid | ~120 | high | 30+ | 6+ |
| nucleic acid | ~80 | medium | 20+ | 4+ |
| carbohydrate | ~40 | medium | 10+ | 3+ |

### E. Sample Evidence
- 1005 cm-1: phenylalanine ring breathing [chemistry_plus_biomolecule]
- 1656 cm-1: Amide I protein band [chemistry_only]
- 1445 cm-1: CH2 bending lipids/proteins [chemistry_plus_biomolecule]

### F. Caveats
- Some functional groups (e.g., "ring") are generic — multiple biochemical origins possible

---

## Example 2: "What does peak 1005 mean?"

### A. Query Understanding
- **Type**: Peak Interpretation Query
- **Entity**: 1005 cm-1

### D. Ranked Interpretation
| Theme | Score | Confidence |
|---|---|---|
| protein | high | high |
| amino acid | medium | medium |

### Key biomolecule: **phenylalanine** (ring breathing mode at ~1005 cm-1)

This is one of the most reproducible and widely reported Raman/SERS peaks. Multi-source agreement is very strong.

---

## Example 3: "What links amide I to biology?"

### A. Query Understanding
- **Type**: Chemistry-to-Biology Query
- **Entity**: amide I

### D. Ranked Interpretation
| Target Theme | Weight | Confidence | Evidence Type |
|---|---|---|---|
| protein | 0.85 | high | inferred (mapping) + direct |
| lipid | 0.10 | low | inferred (C=C overlap region) |

### Explanation
Amide I (~1650 cm-1) is the strongest protein marker in vibrational spectroscopy. It arises from C=O stretching of the peptide backbone. The inferred mapping correctly identifies protein as the dominant target with high weight. The minor lipid link comes from spectral overlap with C=C stretching in unsaturated lipids near 1660 cm-1.
