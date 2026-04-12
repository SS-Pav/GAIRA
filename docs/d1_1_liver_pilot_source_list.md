# D1.1 — Liver Pilot Source List

## Pilot Scope
20 liver-disease comparative papers from liver_SERS_DB, selected for:
- HCC / NAFLD / hepatitis / DILI / fibrosis conditions
- Serum/plasma sample type
- Direct assignment or spectral analysis paper types
- PDFs accessible on disk

## Sources Yielding Directional Evidence (3/20)

| Source | Condition | Entries | Mode | Why Useful |
|---|---|---|---|---|
| liverDB_elsevier_S1386142520307083 | hepatitis | 2 | text | HCV infection study; explicit protein content increase at amide III |
| liverDB_gen_boe_15_11_6469 | liver staging | 3 | text | In-situ liver cancer mouse model; peak intensity changes over time |
| liverDB_gen_119000U__1 | liver staging | 1 | text | Substance content classification at 1580 cm-1 |

## Sources With No Extractable Directionality (17/20)

Most papers contain:
- Peak assignment tables without directional comparison
- ML classifier results (PCA, SVM accuracy) without per-peak direction
- Spectral difference mentions in context too vague for confident extraction
- Discussion of "differences" without explicit up/down/increased/decreased language tied to specific peaks

## Key Observation
The liver SERS literature frequently discusses classification performance (sensitivity, specificity, accuracy) rather than per-peak biochemical directionality. Many papers jump from "we detected differences" to "our classifier achieved X% accuracy" without the intermediate step of "peak Y went up/down in disease."
