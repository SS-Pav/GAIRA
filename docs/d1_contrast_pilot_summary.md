# D1 — Contrast Evidence Pilot Summary

## Results

| Metric | Value |
|---|---|
| Total directional entries | **19** |
| Unique sources | 15 |
| Matched to existing peaks | 19 (100%) |
| Unmatched | 0 |
| Direction: up | 12 |
| Direction: down | 7 |
| With significance info | 2 (10.5%) |
| With magnitude qualifier | 6 (31.6%) |
| With disease condition | 5 (26.3%) |
| Confidence: high | 2 |
| Confidence: medium | 5 |
| Confidence: low | 12 |

## Key Answers

### 1. Is directional extraction practically viable?
**Marginally, from the current extracted text.** 19 entries from 137 sources is very thin. However, the extraction worked — patterns matched, peaks linked, and the schema populated correctly. The bottleneck is input data, not methodology.

### 2. Are text/table sources sufficient for a useful initial layer?
**Not from the current extraction.** The assignment extractor captures "what a peak means" but not "how it changes." A targeted re-extraction pass on the source PDFs, focusing on discussion/results sections, would likely recover 5-10x more directional statements.

### 3. How often do we get direction-only vs direction+effect vs direction+significance?

| Combination | Count | % |
|---|---|---|
| Direction only | 11 | 57.9% |
| Direction + magnitude | 6 | 31.6% |
| Direction + significance | 2 | 10.5% |
| Direction + effect size | 0 | 0% |
| Direction + effect + significance | 0 | 0% |

No numeric effect sizes were found. All directionality is qualitative.

### 4. Is scaling beyond the pilot justified?
**Not from the current corpus alone.** The yield is too low (19 entries) to build a meaningful contrast layer. Two paths forward:
- **Re-extraction pilot**: Go back to 10-20 liver-disease PDFs with differential-focused regex patterns targeting discussion sections. Expected yield: 50-200 additional directional entries.
- **Dataset integration**: If raw spectral data is available, compute actual peak intensity differences directly. This is more reliable than text extraction.

## What Worked
- Schema is clean and implementable
- Peak matching (±5 cm-1) achieved 100% match rate
- Magnitude/significance harmonization works for the few cases that have it
- Provenance is fully preserved

## What Didn't Work
- The current evidence corpus is assignment-focused, not differential-focused
- Discussion paragraphs (where directionality lives) were not captured by the assignment extractor
- No numeric effect sizes exist in the extracted text

## Recommendation
**Proceed cautiously.** The schema and infrastructure are ready. The data is not. Next steps should be:
1. **Keep the D1 schema and pilot data** as the foundation
2. **Run a targeted differential re-extraction** on top 20 liver-disease PDFs
3. **Integrate contrast evidence into C1.8** once the re-extraction yields meaningful volume
4. **Do NOT replace the motif differential layer** — contrast evidence should supplement it, not replace it
