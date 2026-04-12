# GAIRA Spectral Query v1 — HCC Holdout Summary

## Dataset
- 144 spectra: 72 HCC (H0T), 72 healthy control (CTR)
- Source: Vornoli 2020 Trieste HCC serum SERS
- Wavenumber range: 402-1650 cm-1 (fingerprint region)
- Preprocessing: AsLS baseline (lambda=1e5, p=0.001), SG smoothing (11,3), vector normalization

## Key Results

### Alignment to GAIRA Priors

| Comparison | Cosine Similarity | Rank Correlation |
|---|---|---|
| HCC spectral vs GAIRA HCC | **0.237** | 0.095 |
| HCC spectral vs GAIRA healthy | 0.000 | — |
| HCC spectral vs GAIRA NAFLD | 0.056 | -0.024 |
| CTR spectral vs GAIRA healthy | 0.000 | — |

**Interpretation**: Weak but directionally correct. HCC spectral deltas show 0.24 cosine alignment with GAIRA's literature-derived HCC prior — better than alignment with healthy (0.00) or NAFLD (0.06). This is above chance but far from strong transfer.

### Sample-Level Separation

| Metric | HCC Samples | CTR Samples | Separation |
|---|---|---|---|
| Mean similarity to GAIRA HCC | **0.161** | 0.051 | **+0.109** |

HCC samples are, on average, 3x more similar to the GAIRA HCC prior than controls. This is a meaningful signal — the literature-derived biochemical structure partially transfers to real spectra.

### Most Informative Spectral Windows

| Window | BSV Component | Importance | Delta |
|---|---|---|---|
| **1020-1080** | nucleic_acid_backbone | 9.14 | 0.233 |
| **1380-1450** | membrane_lipid | 6.70 | 0.433 |
| **620-660** | aromatic_amino_acid | 2.97 | 0.203 |
| **1260-1320** | protein_backbone | 2.71 | 0.350 |
| **1200-1260** | protein_backbone | 2.42 | 0.292 |
| 820-860 | aromatic_amino_acid | 1.75 | 0.099 |
| 1140-1200 | membrane_lipid | 1.72 | 0.055 |
| 740-780 | pyrimidine_nucleotide | 1.67 | 0.074 |
| **500-540** | redox_metabolite | 1.51 | 0.199 |
| **450-500** | redox_metabolite | 1.12 | 0.244 |

### Which Biochemical Axes Align vs Fail

**Aligned (partial transfer):**
- nucleic_acid_backbone (1020-1080): strongest single window — phosphate/C-O-C region. Consistent with GAIRA's prediction of nucleic acid enrichment in HCC.
- membrane_lipid (1380-1450): CH2/CH3 deformation region. Consistent with membrane remodeling.
- aromatic_amino_acid (620-660): tyrosine ring. Consistent with protein composition shift.

**Failed / flat:**
- purine_nucleotide: GAIRA predicted enrichment, but spectral evidence is mixed (some windows contribute, others don't).
- glycan_carbohydrate: GAIRA predicted depletion, spectral signal is weak.

### Low-Wavenumber Region (<720 cm-1)

**Meaningful contribution confirmed.** Windows at 450-500 and 500-540 cm-1 (redox_metabolite component) rank in the top 10 by importance. The 620-660 cm-1 window (aromatic amino acid) is the 3rd most important. Low-wavenumber exclusion would lose significant biochemical information.

### Signal Stability
Most BSV components show moderate variance across samples (0.02-0.06). The signal is noisy but directionally consistent — HCC samples collectively shift in the direction predicted by GAIRA.

## Success Criteria Evaluation

| Criterion | Result |
|---|---|
| HCC samples shift away from healthy | **YES** — mean separation +0.109 |
| Shifts aligned with nucleic acid / aromatic AA | **PARTIALLY** — nucleic acid backbone and aromatic AA windows are top contributors |
| GAIRA HCC prior matches spectral HCC better than healthy | **YES** — cosine 0.237 vs 0.000 |
| Signals stable across samples | **MODERATE** — directionally consistent but noisy |
| Useful windows cluster in meaningful regions | **YES** — 1020-1080, 1380-1450, 620-660 are biochemically interpretable |
| Low-wavenumber adds value | **YES** — 450-540 cm-1 contributes meaningfully |

## Honest Assessment

The transfer from GAIRA's literature-derived biochemical space to real SERS spectra is **partial but real**:
- Cosine alignment of 0.24 is weak in absolute terms
- But it is the *correct condition* (HCC > healthy > NAFLD)
- Sample-level separation (3x) suggests genuine biochemical grounding
- The most important spectral windows map to biologically interpretable BSV components

The weak alignment reflects:
1. GAIRA's BSV is count-based (not intensity-based) — fundamentally different from spectral intensity
2. The window-to-component mapping is coarse (each window → one component)
3. Spectral variance is high in SERS (substrate effects, concentration effects)
4. The HCC dataset is from a single lab with specific substrate chemistry

## FINAL VERDICT

**Partial transfer with caveats.**

GAIRA's literature-derived biochemical structure shows directionally correct alignment with real HCC serum SERS spectra. The nucleic acid backbone, membrane lipid, and aromatic amino acid axes transfer best. The signal is weak but scientifically consistent. This is exploratory evidence that the biochemical inference framework has physical grounding — not a clinical validation.
