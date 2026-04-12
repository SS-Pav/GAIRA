# GAIRA Landscape v5.1 — Inference Hardening Summary

## What Was Fixed

### 1. Confidence System
- **Bounded to [0,1]** — was unbounded (HCC was 1.39, now 0.73)
- 5 explicit factors: evidence depth, source diversity, BSV mapping, trust tier, single-source penalty
- Geometric-mean blend with exponent weighting
- Component-level confidence = aggregate × signal presence

### 2. Protein Correction Audit
- 6 overcorrection risks found out of 176 entries (3.4%) — within acceptable bounds
- Correction is classified as: none, mild, major, overcorrection_risk, large
- Status: **exploratory but justified** — keeps flag system for honest reporting

### 3. Support-Robust Delta Framework
Four delta variants, from diagnostic to inference-grade:
| Variant | Purpose | When to Use |
|---|---|---|
| raw | Diagnostic | Audit, debugging |
| support_weighted | Attenuated by confidence | Better than raw for comparison |
| corrected | Protein-corrected | Mechanistic exploration |
| corrected_support_weighted | **Best inference candidate** | Primary for HCC/NAFLD analysis |

Support attenuation: sparse conditions went from mean |delta|=0.44 to 0.05 (10x reduction).

### 4. Inference Readiness Gate
| Label | Criteria | Count |
|---|---|---|
| inference_grade | ev>=20, src>=3, bio, mapped>=0.3 | **2** |
| provisional | ev>=8, src>=2, bio, mapped>=0.2 | **2** |
| exploratory | ev>=3, bio | 12 |
| insufficient | everything else | 43 |

**Inference-grade conditions:**
- HCC (ev=42, src=3, conf=0.73)
- liver_cancer_unspecified (ev=95, src=10, conf=0.86)

**Provisional:**
- hepatitis (ev=70, src=2)
- NAFLD_NASH (ev=24, src=2)

### 5. Filter Hardening
Single source of truth: `filter_membership_audit.csv`
- 0 non-biological conditions in inference core (PASS)
- Serum-only, trusted-only, biological-only all explicitly tracked

### 6. Latent Axes
PCA on 4 inference-grade/provisional conditions using support-weighted corrected delta:
- PC1: 50.9% variance
- No single condition dominates (max contribution 0.33)

## Validation Results

| Test | Result |
|---|---|
| V1: Confidence [0,1] | **PASS** (0.00-0.86) |
| V2: Monotonic (HCC > weak) | **PASS** |
| V3: Correction stability | **PASS** (6/176 = 3.4% overcorrection) |
| V4: Support attenuation | **PASS** (0.44 → 0.05) |
| V5: Filter integrity | **PASS** (0 non-bio in core) |
| V6: Readiness strict | **PASS** (2 inference-grade) |
| V7: PCA robustness | **PASS** (max single contribution 0.33) |

## Remaining Weaknesses

1. **Only 2 inference-grade conditions** — HCC and liver_cancer_unspecified
2. **NAFLD is only provisional** — 2 sources is marginal
3. **Protein correction is exploratory** — correlation-based removal may overcorrect in some components
4. **51/95 motifs unmapped to BSV** — chemistry-only evidence without biological theme
5. **No spectral intensity data** — BSV is still literature-count-based

## Final Judgment

**READY WITH CAUTION for HCC holdout.**

HCC has:
- 42 evidence rows from 3 sources
- Confidence 0.73 (bounded, transparent)
- Inference-grade readiness
- 8 nonzero BSV components
- Support-weighted corrected delta is the best available signal

The system is honest about its limitations. The HCC holdout should be treated as an exploratory validation, not a clinical claim.
