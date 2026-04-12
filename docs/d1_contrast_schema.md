# D1 — Contrast Evidence Schema

## Purpose
Attach explicit directionality and reported contrast metadata to existing GAIRA evidence rows. This layer captures "peak X is increased/decreased in disease vs control" — information that the assignment extraction pipeline does not capture.

## Schema Fields

| Field | Type | Description |
|---|---|---|
| contrast_id | string | Unique identifier (e.g., contrast_0001) |
| source_id | string | GAIRA source provenance |
| study_id | string | Study-level grouping (typically = source_id) |
| matched_peak_cm | float | Peak position referenced in the directional statement |
| matched_peak_node_id | string | Link to existing GAIRA evidence row (via peak tolerance matching) |
| peak_match_count | int | Number of existing evidence rows matched within tolerance |
| query_condition | string | Disease/condition (e.g., HCC) |
| comparator_condition | string | Baseline (e.g., healthy_control) |
| sample_type | string | serum / plasma / EV / tissue |
| domain | string | liver / cancer / etc. |
| direction | enum | up / down / shared / unclear |
| effect_size_value | float | Numeric value if reported |
| effect_size_type | enum | fold_change / percent_change / mean_difference / intensity_ratio / qualitative_only / not_reported |
| effect_size_reported_text | string | Raw reported text for effect size |
| magnitude_qualifier | string | Raw qualifier (e.g., "markedly", "slightly") |
| magnitude_score | int | Harmonized: 1=slight, 2=moderate, 3=marked, null=absent |
| significance_value | string | Raw significance text (e.g., "p < 0.05") |
| significance_type | enum | p_value / adjusted_p / text_only / not_reported |
| significance_score | int | Harmonized: 0=none, 1=text, 2=p<.05, 3=p<.01, 4=p<.001 |
| evidence_mode | enum | text / table / figure / dataset |
| extracted_text_or_note | string | Raw source text (up to 200 chars) |
| provenance_location | string | Evidence item ID or page reference |
| extraction_confidence | enum | high / medium / low |

## Allowed Enums

**direction**: `up`, `down`, `shared`, `unclear`

**effect_size_type**: `fold_change`, `percent_change`, `mean_difference`, `intensity_ratio`, `qualitative_only`, `not_reported`

**evidence_mode**: `text`, `table`, `figure`, `dataset`

**significance_type**: `p_value`, `adjusted_p`, `text_only`, `not_reported`
