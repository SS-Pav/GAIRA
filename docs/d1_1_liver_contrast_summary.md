# D1.1 — Liver Raw-PDF Directional Extraction Pilot Summary

## Results

| Metric | Value |
|---|---|
| Liver sources processed | 20 |
| Sources with directional evidence | **3 (15%)** |
| Total matched directional entries | **6** |
| Unmatched entries | 0 |
| Direction: up | 5 |
| Direction: down | 1 |
| With numeric effect size | 0 |
| With significance info | 0 |
| With magnitude qualifier | 0 |
| With disease condition | 6 |
| Confidence: high | 0 |
| Confidence: medium | 5 |
| Confidence: low | 1 |

## Overall Viability: WEAKLY VIABLE

Going directly to raw PDFs yielded only marginally more than the D1 pilot on already-extracted text (6 vs 5 liver-relevant entries from D1). The raw documents DO contain more contextual information, but the directional content is mostly:
- In figure comparisons (not text-extractable)
- In vague narrative discussion (not pattern-matchable)
- In classification results without per-peak direction

## Critical Answers

### 1. Does the raw liver corpus contain substantially more directional info than the extracted evidence layer suggested?
**Marginally, yes — but not enough to change the picture.** The raw PDFs contain more disease-comparison context, but most of it is classification-oriented rather than per-peak directional. The gap between "the document mentions HCC vs healthy" and "the document says peak X went up in HCC" is larger than expected.

### 2. Is scaling across all liver sources justified?
**No, not at this time.** 6 entries from 20 sources (0.3 entries/source) is too low for a productive scaling effort. Processing the remaining 18 liver sources would likely yield ~3-5 additional entries.

### 3. Are HCC vs healthy and NAFLD vs healthy likely to benefit materially?
**Not from text extraction alone.** The strongest entries came from hepatitis (2 entries with protein content increase) and a liver cancer animal model (3 entries with intensity time course). HCC-specific and NAFLD-specific per-peak directionality is essentially absent from the extractable text.

### 4. What should come next?

**Recommended path**: Defer contrast evidence scaling. Instead:

1. **Proceed with BSV** using the existing motif differential + stability layers, which work at cross-source level and don't require within-source directionality
2. **Mark contrast evidence as a future enhancement** that requires either:
   - Manual expert curation of key papers
   - Direct spectral data integration (computing contrasts from raw spectra)
   - NLP-enhanced extraction targeting discussion paragraphs
3. **Keep the D1 schema and 6+19=25 pilot entries** as seed data for when the above becomes available

**What NOT to do**: Do not invest heavily in expanding text-based directional extraction across critical_A/B. The fundamental issue — that SERS literature discusses classification accuracy more than per-peak biochemical direction — is not corpus-specific.
