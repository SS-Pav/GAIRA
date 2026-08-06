# Phase 02.5 — Latent geometry of spectral motif space

**Status:** COMPLETE — analysis only.

> **Numbering.** 02.5 is an inserted analysis phase between the Consensus Spectral Motif
> construction (02) and the biochemical themes (03). It sits in the single results tree with
> every other phase.

## Purpose

Phase 02 asked which motifs are *interchangeable* and found one pair. This phase asks how the
motifs are **related** — neighbourhoods, gradients, bridges, isolates — and turns the answer
into provisional priors for Phase 03.

## What this phase does NOT do

- refit preprocessing, balanced references, class-local NMF, LSMs, CSMs or canonical identities
- create any theme, or any object a later phase consumes as a fitted artefact
- use chemistry-class or source labels to construct the geometry (both are revealed at step 8)

## Inputs (all read-only, all fingerprint-verified)

- 50 LSMs — `../phase01/artifacts/` · registry `208482d6f7178b5b8f16cace91be55b0`
- 49 CSMs — `../phase02/artifacts/` · dictionary `0b4aa550ccefed3edabdbde5bae11c8d`
- frozen V5 atlas `09ed804a40836f4a05a91ba10900cded` — verified only

## Outputs

`artifacts/phase03_geometry_priors.json` (10 provisional priors) · `artifacts/geometry_v1.npz`
(all ten metric geometries, five fusions, and the two primary matrices under distinct keys) ·
11 tables · 4 validation tables · 25 figures (PNG, 200 dpi) ·
`interactive/motif_geometry.html` · `reports/PHASE_02_5_LATENT_GEOMETRY_REPORT.md` (source of
record) · `reports/PHASE_02_5_FIGURES.pdf` (27 pages — cover, contents, all 25 figures with
captions)

## Layout

`code/` run script, figures, PDF builder · `artifacts/` manifests, geometries, priors ·
`tables/` · `validation/` confounding and leave-one-out · `figures/` · `interactive/` ·
`logs/` · `reports/`

**Do not store here:** raw spectra · anything with a hard-coded absolute path · outputs of
other phases.
