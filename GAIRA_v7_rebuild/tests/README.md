# `GAIRA_v7_rebuild/tests/`

V7-specific tests. **Currently empty.**

The scaffold test for this documentation pass lives at the repository root:
`tests/test_v7_rebuild_scaffold.py` — placed there so it runs with the existing suite.

## What belongs here

- unit tests for V7 build modules
- data-contract validation tests (artefact ↔ schema)
- invariant tests: non-negativity, determinism, batch independence, provenance completeness
- regression tests pinning frozen V7 artefacts

## What must not be stored here

- **Tests for V5/V6 code.** Those live in the root `tests/`.
- **Scientific-model tests before the model exists.** This pass adds structure tests only.
- **Tests requiring raw data**, unless skipped when `GAIRA_DATA_ROOT` is unset. A test that
  fails on a clean clone is a broken test.

## Invariants that must eventually be tested

| Invariant | Phase |
|---|---|
| all bases, activations, memberships, BSVs non-negative | 02–05 |
| identical input → byte-identical output | 06 |
| output identical alone vs in a batch of N | 06 |
| no `fit`/`fit_transform`/`partial_fit`/RNG in the inference path | 06 |
| clean clone inference with `GAIRA_DATA_ROOT` unset | 06 |
| multi-layer atlas fingerprint verified on load | 06 |
| every CSM resolves to LSMs → classes → analytes → sources | 03, 06 |
| no canonical ID crosses a CV fold | 00 |
| `bsv` absolute; `bsv_elevation` signed; never conflated | 05, 06 |
| no domain object reachable from any pre-BSV module | 06 |
