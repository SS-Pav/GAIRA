# D1 — Harmonization Rules

## IMPORTANT
All harmonized scores are **derived metadata**, not raw truth. The raw reported text is always preserved in `magnitude_qualifier`, `significance_value`, and `extracted_text_or_note`.

## Magnitude Score

| Qualifier | Score | Examples |
|---|---|---|
| slight / mildly / small | 1 | "slightly elevated", "mildly increased" |
| moderate / moderately / consistently | 2 | "moderately reduced", "consistently elevated" |
| marked / strong / pronounced / dramatic / substantial | 3 | "markedly upregulated", "dramatically decreased" |
| absent / not reported | null | No magnitude qualifier present |

### Special Case: "significant" / "significantly"
- **NOT** treated as magnitude
- Mapped to significance support instead
- "significantly decreased" → direction=down, significance_score>=1, magnitude_score=null

## Significance Score

| Level | Score | Criteria |
|---|---|---|
| not reported | 0 | No significance information |
| text only | 1 | "significantly" without p-value |
| p < 0.05 | 2 | Reported p-value < 0.05 |
| p < 0.01 | 3 | Reported p-value < 0.01 |
| p < 0.001 | 4 | Reported p-value < 0.001 |

### P-value Extraction
Regex: `p\s*[<>=]\s*([\d.]+)`
Applied to both `cleaned_meaning` and `original_meaning` fields.

## Pilot Statistics
- 6/19 entries had magnitude qualifiers (score distribution: 2x score=2, 4x score=3)
- 2/19 entries had significance info (both score=1, text-only "significantly")
- 0/19 entries had numeric p-values
- 0/19 entries had numeric effect sizes
