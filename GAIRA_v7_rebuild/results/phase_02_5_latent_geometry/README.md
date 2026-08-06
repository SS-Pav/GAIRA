# Phase 02.5 — Latent geometry of spectral motif space

**Status:** COMPLETE — analysis only.

> **Where this lives.** Phases 00–02 write to `results/v7_rebuild/phaseNN/` at the repository
> root. Phase 02.5 was commissioned to write here instead, so the V7 results are split across
> two trees. `results/v7_rebuild/README_phase_02_5_pointer.md` points back here so the
> provenance chain stays discoverable from the canonical location.

## Purpose

Phase 02 asked which motifs are *interchangeable* and found one pair. This phase asks how the
motifs are **related** — neighbourhoods, gradients, bridges, isolates — and turns the answer
into provisional priors for Phase 03.

## What this phase does NOT do

- refit preprocessing, balanced references, class-local NMF, LSMs, CSMs or canonical identities
- create any theme, or any object a later phase consumes as a fitted artefact
- use chemistry-class or source labels to construct the geometry (both are revealed at step 8)

## Inputs (all read-only, all fingerprint-verified)

- 50 LSMs — `results/v7_rebuild/phase01/artifacts/` · registry `208482d6f7178b5b8f16cace91be55b0`
- 49 CSMs — `results/v7_rebuild/phase02/artifacts/` · dictionary `0b4aa550ccefed3edabdbde5bae11c8d`
- frozen V5 atlas `09ed804a40836f4a05a91ba10900cded` — verified only

## Outputs

`artifacts/phase03_geometry_priors.json` (10 provisional priors) · `artifacts/geometry_v1.npz`
(all ten metric geometries, five fusions, and the two primary matrices under distinct keys) ·
11 tables · 5 validation tables · 25 figures (SVG + PNG) · `interactive/motif_geometry.html` ·
`reports/PHASE_02_5_LATENT_GEOMETRY_REPORT.md`

## Layout

`code/` run script and figures · `artifacts/` manifests, geometries, priors · `tables/` ·
`validation/` confounding and leave-one-out · `figures/` · `interactive/` · `logs/` · `reports/`

**Do not store here:** raw spectra · anything with a hard-coded absolute path · outputs of
other phases.
