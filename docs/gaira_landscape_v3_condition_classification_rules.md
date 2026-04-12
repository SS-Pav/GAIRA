# Landscape v3 — Condition Classification Rules

## Categories

| Category | Definition | Count |
|---|---|---|
| `biological_condition` | True disease/clinical condition | 23 |
| `healthy_or_control` | Healthy/normal/untreated baseline | 2 |
| `method_or_review` | Reference methods, reviews | 4 |
| `context_or_task` | Detection targets, ML contexts, monitoring tasks | 15 |
| `unknown_or_misc` | Unclassifiable | 15 |

## Rules (deterministic, keyword-based)

### biological_condition
Includes: HCC, NAFLD_NASH, fibrosis, cirrhosis, hepatitis, cholangiocarcinoma, ovarian_cancer, prostate_cancer, breast_cancer, lung_cancer, leukemia, depression, bacterial_identification, covid19, etc.

### healthy_or_control
Includes: healthy_control, untreated_control

### method_or_review
Includes: reference_method, review_or_method, reference_compound, reference_biomolecule

### context_or_task
Includes: albumin_detection, bilirubin_detection, glucose_monitoring, miRNA_biomarker, metabolite_detection, drug_monitoring, environmental_toxicology, etc.

### unknown_or_misc
Everything not captured by the above rules.

## Impact on Landscape
- **Default view**: biological_condition + healthy_or_control only
- **Method/review/context labels excluded** from main clustering to avoid polluting disease similarity structure
- User can toggle to "all" to see everything
