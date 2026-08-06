# `GAIRA_v7_rebuild/data_contracts/`

Machine-readable schema definitions for the artefacts specified in
`../architecture/DATA_CONTRACTS.md`. **Currently empty — schemas are written in Phase 00.**

## What belongs here

- JSON Schema / YAML schema files, one per contract (C-00 … C-11)
- schema version history and migration notes
- validation helpers that check an artefact against its schema

## What must not be stored here

- **Actual data.** Schemas only. Instances live in `../results/`.
- **Prose specifications.** Those live in `../architecture/DATA_CONTRACTS.md`; this directory
  holds the executable form of the same contracts.
- **Raw spectra.**

## Why the schemas are executable and not only prose

Every artefact crossing a phase boundary carries invariants that matter scientifically, not
just structurally:

| Invariant | Consequence if violated |
|---|---|
| non-negativity throughout | a negative "amount of lipid chemistry" is meaningless, and it breaks composition with the future SERS observation model |
| no canonical ID across CV folds | every downstream metric inflates invisibly (risk R-09) |
| `is_singleton` ⇔ `n_lsms == 1` | a weakly-supported axis silently presents as a well-supported one — the V5 flavin failure (1.2% coverage, indistinguishable output) |
| `S` rows sum to 1.0 | theme mass is not conserved and the BSV is not interpretable as a distribution over chemistry |
| `bsv` absolute, `bsv_elevation` signed | conflating them places a difference into an absolute coordinate frame — a correctness bug |

Checking these by eye across ten phases does not scale. Checking them mechanically does.

## Contract index

| ID | Artefact | Phase |
|---|---|---|
| C-00 | canonical analyte table | 00 |
| C-01 | replicate group table | 00 |
| C-02 | quality metadata | 00 |
| C-03 | CV split manifest | 00 |
| C-04 | balanced reference matrix | 01 |
| C-05 | LSM dictionary + registry | 02 |
| C-06 | LSM similarity graph | 03 |
| C-07 | CSM dictionary + registry | 03 |
| C-08 | theme registry + membership | 04 |
| C-09 | BSV specification | 05 |
| C-10 | inference output | 06 |
| C-11 | build manifest | all |
