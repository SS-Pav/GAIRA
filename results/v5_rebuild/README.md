# `results/v5_rebuild/` — the reproducible build & audit of the frozen model

This tree is GAIRA's **provenance and reproducibility record** — how the frozen
biochemical coordinate system was built, characterised, and validated. It is *not* what
you run at inference time (that is `assets/foundation/`); it is what proves the frozen
model is correct and reproducible.

**Atlas fingerprint:** `09ed804a40836f4a05a91ba10900cded` (NMF, k = 24, 375 pure-Raman
spectra / 167 analytes, 450–1800 cm⁻¹ @ 2 cm⁻¹).

## Contents

| Path | Purpose | Runtime? |
|---|---|---|
| `foundation/` | The frozen NMF **build**: benchmark, manifold, artifacts (`manifold.json`, `manifold_components.npz`) and build logs. Source of `assets/foundation/`. | source of frozen model |
| `engine_v1/` | The interpretation-layer artifacts: component registry, component→theme weights, reference-normalization frame + support. | source of frozen model |
| `foundation_audit/` | The complete **Foundation Model audit** (see its own `README.md`). | powers the Explorer app |
| `spike_validation/` | Serum-spike / dose / uricase projection tables used by the reasoning demo. | demo data |
| `phase0/ … phase2_stage_*`, `preprocessing_autoresearch/`, `spectral_audit/`, `reference_atlas_audit/`, `bsv_validation/`, `perturbation_response/` | Historical build phases: preprocessing selection, representation strategy, audits. | provenance only |

## The frozen model vs this tree

`assets/foundation/` is a byte-identical, self-contained snapshot of the frozen inference
assets drawn from `foundation/artifacts/` and `engine_v1/artifacts/`. The engine loads
`assets/foundation/` (with a fallback to these original locations), so the two are always
consistent. **`assets/foundation/` is the product; this tree is the receipt.**

## Reproduce the frozen atlas (deterministic, seed 0)

```bash
# rebuild NMF k=24 from the pure-Raman corpus and reproduce the fingerprint byte-for-byte
python results/v5_rebuild/foundation_audit/code/repro_benchmark.py     # ~4 min
# full corpus / preprocessing / component / validation reproduction:
python results/v5_rebuild/foundation_audit/code/corpus_analysis.py
python results/v5_rebuild/foundation_audit/code/component_audit.py
python results/v5_rebuild/foundation_audit/code/run_validation.py
```

Regeneration scripts read the raw corpus from the lab volume
(`/Volumes/SSD_Rad/GAIRA_DATA/`, i.e. GAIRA_Lab). They are **not** needed to run the
engine or the demos — those use only the committed assets. Intermediate lab artifacts
(e.g. `phase2_stage_b/models/*.pt`) are git-ignored and belong in GAIRA_Lab.

## The audit, in one line

Rebuilding the representation from scratch reproduces the frozen basis exactly; the full
5-representation × 6-k benchmark reproduces to floating-point identity; the 24 components
are stable and non-redundant; and the model validates on six datasets it never trained on.
Full write-up: `foundation_audit/reports/` (11 parts) + `foundation_audit/README.md`.
