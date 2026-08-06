# Phase 04 — Biochemical theme construction

**Status:** Not started — blocked by Phase 03

---

## Purpose

Derive soft, sparse, non-negative biochemical themes from the CSMs and select the theme count `K`.

## Why this phase exists

Themes are derived *from* CSMs rather than asserted *over* them — the direct response to limitation L-05, where a cleaned ontology imposed on a fixed representation produced one significant improvement out of four levels.

## Inputs

- `csm_dictionary_v1.npz` and `csm_registry_v1.json` from Phase 03
- frozen CV splits

## Outputs

- `theme_registry_v1.yaml`, `theme_membership_v1.npz` (C-08)
- membership entropy per CSM
- theme graph
- `K` sweep with the Pareto frontier
- theme-layer value analysis vs the CSM layer
- `reports/PHASE_04_REPORT.md`

## Gate — all must pass before the next phase begins

- [ ] Themes represent coherent chemistry
- [ ] **No disease, pathway, process, or phenotype labels** (P-07)
- [ ] No hard one-parent requirement; soft membership retained
- [ ] `K` justified on a Pareto frontier by the pre-registered rule
- [ ] Theme layer's value over the CSM layer measured and reported either way

## Primary risks

- R-11 themes decorative
- R-12 BSV axes correlated

### The mapping

`t = Sᵀc` with `S ∈ ℝ₊^{M×K}` sparse, non-negative, row-normalised. A CSM may belong to
several themes — shared biochemical structure genuinely does, and forcing one parent destroys
information.

### Precedent to heed

The V6.2 Pareto study (`results/v6_rebuild/tables/v62_pareto.csv`) found chemical
admissibility first satisfied at `K = 13`, while information retained already reached 0.796
at `K = 6`, and recoverability *fell* monotonically with `K` (0.969 at K=2 → 0.503 at K=12).
Compression and admissibility pull hard in opposite directions in this data. Expect the same
tension; resolve it by the pre-registered rule, not by whichever number looks nicer.

### The decorative-layer test

At V6.2, `theme_raw` and `theme_posterior` were numerically identical at every metric on
every ontology — the posterior machinery changed no decisions. Phase 04 must show the theme
layer adds value over the CSM layer, or record that it does not and consider shipping a
CSM-level BSV with `K = M`.

### Naming discipline

"lipid chemistry" ✓ · "membrane remodelling" ✗ · "inflammation" ✗ · "tumour metabolism" ✗

---

## What belongs in this directory

Phase-04 code, configs, notebooks, per-phase tables, and the phase report. Artefacts that
later phases consume belong in `../../results/` (tables, figures, manifests, checkpoints,
phase_outputs) so the provenance chain stays in one place.

**Do not store here:** raw spectra · large regenerable intermediates · anything with a
hard-coded absolute path · outputs from other phases.

## Reference documents

- `../../context/GAIRA_V7_CONTEXT.md` — canonical scientific context
- `../../plan/GAIRA_V7_REBUILD_PLAN.md` — Phase 04 in the full sequence
- `../../plan/VALIDATION_AND_DECISION_RULES.md` — pre-registered selection rules
- `../../plan/RISK_REGISTER.md` — full risk detail
- `../../architecture/DATA_CONTRACTS.md` — artefact schemas
