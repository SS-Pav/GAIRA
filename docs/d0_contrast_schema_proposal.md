# D0 — Contrast Evidence Schema Proposal

## Status: DEFERRED
This schema is defined for future use. The current corpus does not contain sufficient directionality to populate it. It should be implemented when a targeted differential extraction pass is executed.

## Proposed Schema

### Fields

| Field | Type | Description |
|---|---|---|
| contrast_id | string | Unique identifier |
| source_id | string | Link to GAIRA source provenance |
| study_id | string | Paper/study-level grouping |
| query_condition | string | Disease/condition being studied |
| comparator_condition | string | Healthy control / other comparator |
| sample_type | string | serum / plasma / EV / tissue |
| domain | string | liver / cancer / etc. |
| peak_cm | float | Center wavenumber |
| peak_window | string | e.g., "1000-1010" for range |
| direction | enum | up / down / shared / unclear |
| effect_size_value | float | Optional numeric (fold change, ratio, etc.) |
| effect_size_type | enum | fold_change / ratio / intensity_difference / vip_score / none |
| significance | string | p-value or significance description |
| evidence_mode | enum | text / table / figure / dataset |
| extracted_text_or_note | string | The original phrase or table cell content |
| provenance_location | string | Page, section, table number |
| extraction_confidence | enum | high / medium / low |

### Allowed Enums

**direction**: `up`, `down`, `shared`, `unclear`
- `up`: peak intensity/contribution increased in query condition vs comparator
- `down`: decreased in query condition
- `shared`: no meaningful difference
- `unclear`: mentioned but direction ambiguous

**evidence_mode**: `text`, `table`, `figure`, `dataset`
- `text`: extracted from body/discussion text
- `table`: from a structured peak table with comparison
- `figure`: digitized from a spectral figure
- `dataset`: derived from spectral data analysis

**effect_size_type**: `fold_change`, `ratio`, `intensity_difference`, `vip_score`, `none`

## Integration with GAIRA Graph
When populated, contrast evidence would create new edges:
- `ContrastEvidence -[SUPPORTS_DIRECTION]-> MotifDifferential`
- `ContrastEvidence -[FROM_SOURCE]-> Source`
- `ContrastEvidence -[IN_CONDITION]-> Condition`

This would strengthen the motif differential layer by providing within-source directional support for the cross-source stability analysis.
