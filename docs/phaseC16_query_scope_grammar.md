# Phase C1.6 — Query Scope Grammar

## Supported Scope Modifiers

### Sample Type Scope
Keywords: `serum`, `plasma`, `ev`, `exosome`, `tissue`, `saliva`, `urine`, `pathogen`

Examples:
- `Compare HCC vs healthy **serum**` -> `sample_scope = serum`
- `Compare HCC vs NAFLD in **serum**` -> `sample_scope = serum`

### Domain Scope
Keywords: `liver`, `hepatic`, `cancer`, `infectious`, `neurological`, `cardiovascular`

Examples:
- `Compare HCC vs healthy serum within **liver** sources` -> `domain_scope = liver`

### Scope Mode Resolution

| Query Pattern | Scope Mode |
|---|---|
| No sample/domain specified | `broad` |
| Sample type present | `same_matrix` |
| "across all" + sample type | `all_sources_same_matrix` |
| "within" + domain | `same_domain` |
| Sample type + domain | `same_matrix_domain` |
| Sample type + "within" + domain | `same_matrix_domain` |

### Fallback Behavior
If a scoped query returns < 3 evidence rows:
- The scope is broadened (sample/domain filters removed)
- A `scope_fallback` warning is generated and displayed
- The user is informed that results reflect the broader scope
