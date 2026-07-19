# GAIRA V6 Engine — Migration Notes

How the converged `src/gaira/engine/` relates to the existing codebase, what is
new, what is untouched, and how to adopt it. **Additive migration: nothing is
removed or broken.**

---

## 1. What is NEW (this build)

- `src/gaira/engine/` — the converged reasoning engine (10 modules).
- `src/gaira/engine/data/biochemical_ontology_v2.yaml` — curated 12-theme ontology.
- `results/v5_rebuild/engine_v1/artifacts/` — generated versioned artifacts
  (component registry, theme weights, reference normalization, examples, versions).
- Docs: `GAIRA_Rebuild_Plan_vNext.md`, `GAIRA_Engine_Architecture.md`.

## 2. What is UNTOUCHED (do not modify)

- The frozen Raman Reference Atlas (`results/v5_rebuild/foundation/`), its NMF,
  its preprocessing, and all completed studies.
- The historical inference stack: `src/gaira/inference.py`, `base2/`, `base3/`,
  `theme_ontology.py`, `biochemical_theme_layer.py`, `serum_context.py`,
  `ev_context.py`, `evidence/`, `evidence_v1/`, `domain_pack_registry.py`.
- The demos (`gaira_demo_reasoning_v1`, `v3`, `v3_1`, `src/gaira/demo`, `app`).

The engine imports only `gaira.foundation` (frozen atlas), `gaira.preprocessing`
(frozen), and reads the frozen study tables. It has **zero** write-back into any
historical module.

## 3. Naming — why a new package rather than editing the old one

The repository already has `theme_ontology.py`, a curated 11-axis `biochemical_theme_layer.py`,
and domain-context modules from the V2/V3 line. These encode the *old* hand-curated
radar and are used by the production `inference.py`. Editing them would break the
existing demo and conflate two ontologies. The V6 engine therefore lives in its own
namespace (`gaira.engine`) with its own versioned `biochemical_ontology_v2`. The two
can coexist; a future UI can switch between them.

## 4. Conceptual mapping old → new

| Old (V2/V3) | New (V6 engine) | Note |
| --- | --- | --- |
| 11 curated band-window axes | 12 evidence-derived themes | themes carry provenance + confidence |
| `biochemical_theme_layer.py` | `engine/ontology.py` + `data/ontology_v2.yaml` | many-to-many, not one-label |
| cohort-mean normalization | `engine/normalization.py` (frozen reference frame) | Part 10 |
| `serum_context.py` / `ev_context.py` | `engine/domain.py` | interpretation-only, does not change BSV |
| implicit scoring | `engine/bsv.py` (documented equations) | Part 6 |
| radar in demo | `engine/radar.py` (backend only) | UI deferred |
| — | `engine/evidence.py` | new: full provenance tracing |
| — | `engine/dart.py` | new: future-DART interfaces only |
| — | `engine/versioning.py` | new: per-layer versions |

## 5. How to adopt (for a future integrator)

```python
from gaira.engine import GAIRAEngine
eng = GAIRAEngine()                         # loads frozen atlas + versioned stack once
out = eng.infer(wavenumber=wn, intensity=y, domain="serum")
radar = out.radar                           # theme axes for the UI
evidence = out.evidence                     # click-through provenance
bsv = out.bsv                               # canonical biochemical representation
full = out.as_dict()                        # Part-12 nothing-hidden output
```

Or, from a cached 24-component projection:

```python
out = eng.infer(coordinates=coords24, domain="buffer")
```

## 6. Regeneration order (if artifacts are ever rebuilt)

`build_registry.py` → `build_theme_weights.py` → `build_reference_norm.py`
→ (`run_validation.py`, `emit_examples.py`). The first three are deterministic
functions of frozen inputs; running them again reproduces byte-stable artifacts.

## 7. Versioning contract (Part 13)

Bump `component_theme_weights` / `biochemical_ontology` freely to re-theme — this
never touches the frozen atlas coordinates or the reference normalization. Bump
`reference_atlas` ONLY by replacing the frozen artifact, which invalidates every
downstream version and requires re-running all builders. `engine/versioning.py`
pins the atlas fingerprint; a mismatch raises on load.

## 8. Guardrails already enforced in code

- Every module that loads the atlas verifies the fingerprint and refuses to run on
  a mismatch.
- Theme weights are asserted to sum to 1 per component at build time.
- The pipeline stamps `versions` (including the atlas fingerprint) into every output.
- OOD score, matrix share and unknown share are always reported; confidence is
  attenuated by OOD; low-purity components carry explicit caveats.
