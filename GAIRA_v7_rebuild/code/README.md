# `GAIRA_v7_rebuild/code/`

V7 implementation code. **Currently empty — no V7 code has been written.**

## What belongs here

- V7 build modules, one per phase: reference construction, LSM fitting, graph construction,
  CSM derivation, theme mapping, BSV construction
- the V7 inference engine (a **parallel** implementation — not an edit of `src/gaira/engine/`)
- shared V7 utilities: manifest writing, fingerprinting, canonicalisation, provenance traversal
- phase entry-point scripts

## What must not be stored here

- **Modifications to `src/gaira/` or any existing engine.** V7 is a parallel implementation.
  The V5 engine stays in production, unmodified, until Phase 07 authorises a replacement.
- **Raw spectra or any data.** Code only.
- **Absolute local paths.** Use `GAIRA_DATA_ROOT`; the documented default is `None` so no
  lab-specific path is ever committed.
- **Generated artefacts.** Those go to `../results/`.
- **Notebooks used as the source of truth.** A notebook may explore; the committed pipeline
  must be a script or module that a manifest can point at.

## Structural requirements

Two requirements are architectural, not stylistic, and both are Phase-06 gates.

**1. The learning/inference boundary is enforced in the module layout.** Anything that fits
lives in a build module; the inference module imports nothing that can fit. The prohibited
list — NMF fitting, PCA fitting, UMAP, clustering, community detection, ontology optimisation,
batch statistics, threshold tuning on incoming data — is checked statically for
`fit`/`fit_transform`/`partial_fit`/RNG use in the inference path.

**2. Every build entry point writes a manifest.** Inputs and hashes, config, seeds, code git
SHA, environment, outputs and hashes, gate results, decisions with their pre-registration
pointer. A manifest with `code.dirty: true` invalidates the phase, because the code that
produced it cannot be recovered.

## Reference

- `../architecture/LEARNING_MODE_ARCHITECTURE.md` — what the build code must implement
- `../architecture/INFERENCE_MODE_ARCHITECTURE.md` — the permitted/prohibited operation lists
- `../architecture/DATA_CONTRACTS.md` — artefact schemas
- `../plan/GIT_AND_VERSIONING_PLAN.md` — commit and protected-path rules
