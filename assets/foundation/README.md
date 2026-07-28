# `assets/foundation/` — the frozen GAIRA biochemical reference space

This folder is the **frozen scientific product**: everything required to project a Raman
spectrum into GAIRA's biochemical coordinate system and read it. It is self-contained —
the inference engine loads exclusively from here (with a legacy fallback), and it needs
no raw data, no SSD volume, and no recomputation.

**Atlas fingerprint:** `09ed804a40836f4a05a91ba10900cded` (SHA-256 of the NMF basis).
Verified on every engine load against the pinned version. See `MANIFEST.json` for the
per-file SHA-256s and the version of every stack layer.

## Contents

| File | What it is |
|---|---|
| `manifold.json` | Frozen NMF metadata: k=24, fingerprint, corpus card, selection, validation. |
| `manifold_components.npz` | The **NMF basis H** (24 × 676) — the coordinate axes — plus the grid. |
| `component_registry_v1.json` | Per-component provenance: bands, reference loadings, stability, perturbation evidence. |
| `component_theme_weights_v1.json` | The component→theme weight matrix **W** (24 × 13). |
| `biochemical_ontology_v2.yaml` | The 13 biochemical theme definitions (11 biochemical + 2 non-biochemical). |
| `mss_motifs_v1.yaml` | The 13 Molecular Spectral Signature motif definitions. |
| `reference_normalization_v1.json` | Reference frame (per-component center/spread) for z-scores & elevation. |
| `reference_support.npz` | Reference support vectors for the out-of-distribution (OOD) score. |
| `MANIFEST.json` | Fingerprint, all stack versions, preprocessing config, per-file SHA-256. |

## How it is loaded

`gaira.engine.paths` resolves each frozen file, preferring this folder and falling back
to the original build locations (`results/v5_rebuild/…`, `src/gaira/engine/data/`) if this
folder is ever absent — so the engine is fully backwards compatible.

## Provenance & reproduction

This bundle is a byte-identical snapshot of the frozen build under
`results/v5_rebuild/` (the reproducible provenance tree). Rebuilding the NMF from the
pure-Raman corpus reproduces `manifold_components.npz` exactly (identical fingerprint) —
see `results/v5_rebuild/foundation_audit/` (the full scientific audit) and its
`NMF_REBUILD.md`. The model is **frozen**: these files do not change; changing them would
change the fingerprint and invalidate every downstream layer.
