# GAIRA Landscape — Gap Analysis

## 1. Missing Conditions (expected but absent or very weak)

| Condition | Status | Expected In Literature |
|---|---|---|
| **Cirrhosis** | Present as "fibrosis" (2 rows only) | Yes — major liver condition |
| **MELD/Child-Pugh staging** | Absent | Yes — liver severity staging |
| **Colorectal cancer** | Absent | Yes — common cancer SERS target |
| **Gastric cancer** | Absent | Moderate |
| **Sepsis** | Absent (bacterial_infection exists but not sepsis) | Yes — biofluid SERS target |
| **Alzheimer's** | Absent | Moderate — CSF/serum studies exist |
| **Thyroid disease** | Absent | Low |

## 2. Weakly Supported Conditions

| Condition | Evidence | Sources | Issue |
|---|---|---|---|
| fibrosis | 2 | 2 | Far too sparse for staging |
| DILI_hepatotoxicity | 9 | 1 | Single-source — no cross-validation |
| breast_cancer | 8 | 1 | Single-source |
| leukemia | 7 | 1 | Single-source |
| depression | 5 | 1 | Single-source |
| prostate_cancer | 4 | 1 | Single-source |

## 3. Overrepresented Motifs/Themes
- **Phenylalanine** (88 rows) — present in nearly every source
- **Protein** theme (239 rows) — ubiquitous in serum
- **Lipid** theme (201 rows) — ubiquitous in serum
- These are real biology but poor differentiators

## 4. Underrepresented Biochemical Classes
- **Sphingolipids/ceramides** — absent despite relevance to cancer
- **Steroid hormones** — absent
- **Bile acids** — absent (critical for liver disease)
- **Specific phospholipid species** — only bulk phospholipid present
- **Metabolic flux markers** — only static assignments, no kinetic evidence

## 5. Sample-Type Imbalance
- **Serum dominates**: ~70% of evidence
- **Tissue**: only 3 sources
- **EV/exosome**: 7 sources (growing but still sparse)
- **Urine**: 2 sources
- **Saliva**: 3 sources
- **CSF**: 0 sources
- **Whole blood**: 0 sources

## 6. Source Family Imbalance
- **Liver corpus**: 40 sources (largest single contribution)
- **Critical_A/B**: 55 sources combined
- **Pre-critical_A**: 39 sources (reference-heavy)
- The liver focus is by design but limits generalizability to non-liver conditions
