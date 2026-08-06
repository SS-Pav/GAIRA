# GAIRA V7 — Artifact and Manifest Specification

> **REVISED SCOPE — 2026-08-06 after Phase 05.** The manifest and fingerprint rules are
> unchanged and binding. The *list* of layers the fingerprint must cover has changed: the
> theme registry, membership matrix and BSV reference frame are **ARCHIVED** (A-13, A-14) and
> the bundle now covers the reference bank, calibrator, rejection thresholds and — once built
> — the chemistry-evidence map and BSV2 programmes. See `GAIRA_V7_TARGET_ARCHITECTURE.md` §3.

What gets frozen, how it is fingerprinted, and what a version bump means.

---

## 1. The V7 Atlas bundle

The V7 Atlas is a **layered bundle**, not a single basis. In V5 the two were nearly the same
thing — a 24×676 matrix plus a thin overlay. In V7 the atlas spans preprocessing, two
dictionary levels, a mapping, a reference frame, and a declared validation boundary.

Target layout (`GAIRA_v7_rebuild/results/checkpoints/atlas_v7_<version>/`):

```
atlas_v7_<version>/
├── MANIFEST.json                  ← fingerprint, versions, per-file hashes, boundary
├── preprocessing_spec_v1.json     ← window, grid, baseline/smooth/norm + parameters
├── csm_dictionary_v1.npz          ← CSM basis (M × 676) + grid    [PROJECTION BASIS]
├── csm_registry_v1.json           ← per-CSM provenance, bands, uncertainty, flags
├── lsm_dictionary_v1.npz          ← per-class LSM dictionaries    [EVIDENCE LAYER]
├── lsm_registry_v1.json           ← per-LSM stability, type, provenance
├── theme_membership_v1.npz        ← S (M × K)
├── theme_registry_v1.yaml         ← K themes, chemical definitions
├── bsv_reference_v1.json          ← per-axis reference stats, effective rank
├── bsv_ood_support_v1.npz         ← OOD support vectors
├── bsv_pca_v1.npz                 ← frozen visualisation transform (P, μ) — VIS ONLY
├── canonical_analytes_v1.csv      ← identity table (provenance root)
├── PROVENANCE.json                ← full build chain, input hashes, code SHA, environment
└── README.md                      ← what this atlas is, its boundary, how to load it
```

Design constraints, all inherited from what made the V5 bundle work:

- **Self-contained.** Inference runs from this directory alone — no raw data, no lab volume.
- **No raw spectra.** Prototypes and basis vectors only.
- **No absolute paths.** Nothing inside references a machine-specific location.
- **Immutable once frozen.** Any change is a new version, never an edit.

---

## 2. Fingerprinting

### The V5 precedent

```
atlas_fingerprint = sha256(ascontiguousarray(H).tobytes()).hexdigest()[:32]
                  = 09ed804a40836f4a05a91ba10900cded
```

This hashes **only the NMF basis**. That was adequate in V5 because the basis was the atlas
in every meaningful sense — the overlay layers were thin, curated, and separately versioned.

### The V7 requirement

It is **not** adequate for V7. A V7 atlas with an identical CSM basis but a different `S`
produces different BSVs. If the fingerprint covered only the basis, two behaviourally
different atlases would be indistinguishable — the exact silent-divergence failure the
fingerprint exists to prevent.

**The V7 fingerprint covers every behaviour-determining layer:**

```
layer_hash(name)   = sha256(canonical_bytes(layer))
atlas_fingerprint  = sha256( concat over layers in FIXED ORDER of
                             (name || ":" || layer_hash) ).hexdigest()[:32]
```

Fixed layer order (any change to this order is itself a breaking change):

1. `preprocessing_spec`
2. `csm_dictionary`
3. `lsm_dictionary`
4. `theme_membership`
5. `theme_registry`
6. `bsv_reference`
7. `bsv_ood_support`

**Excluded from the fingerprint** (they do not change inference output): registries carrying
only provenance and human-readable metadata, `PROVENANCE.json`, `README.md`,
`canonical_analytes_v1.csv`, and `bsv_pca_v1.npz` (visualisation only). These are hashed
per-file in `MANIFEST.json` for integrity, but a change to them is not a behavioural change.

**Per-layer hashes are recorded individually** in `MANIFEST.json`. This is what makes it
possible to say *which* layer changed between two atlas versions — something the V5 single
hash could not do.

### Canonicalisation rules (required for reproducible hashing)

| Artefact type | Rule |
|---|---|
| numpy arrays | `float64`, C-contiguous, fixed shape order, hashed via `.tobytes()` |
| JSON | UTF-8, sorted keys, no insignificant whitespace, fixed float repr |
| YAML | serialised to canonical JSON before hashing |
| CSV | fixed column order, fixed line ending, UTF-8, no trailing whitespace |

Without these, the same atlas hashes differently on two machines and the whole scheme is
worthless.

---

## 3. MANIFEST.json

```
{
  "schema": "gaira_v7_atlas_manifest_v1",
  "name": "GAIRA V7 hierarchical biochemical reference space",
  "atlas_version": "v7.0.0",
  "atlas_fingerprint": "<32 hex>",
  "layer_fingerprints": { "preprocessing_spec": "...", "csm_dictionary": "...",
                          "lsm_dictionary": "...", "theme_membership": "...",
                          "theme_registry": "...", "bsv_reference": "...",
                          "bsv_ood_support": "..." },
  "dimensions": { "D": 676, "M": <int>, "K": <int>,
                  "n_classes": <int>, "n_lsms": <int> },
  "preprocessing": { "window_cm": [450.0, 1800.0], "grid_step_cm": 2.0,
                     "pipeline": {"baseline": "asls", "smooth": "savgol", "norm": "l2"},
                     "parameters": {...} },
  "corpus_card": { "domain": "Raman only", "n_spectra": <int>, "n_analytes": <int>,
                   "sources": {...}, "excitations": {...},
                   "excluded_domains": ["Ag-SERS", "Au-SERS", "DART",
                                        "serum Ag-colloid", "..."] },
  "validation_boundary": { "validated_domains": ["pure Raman reference"],
                           "window_cm": [450.0, 1800.0],
                           "excitations_validated": [...],
                           "not_validated": ["SERS", "biological mixtures",
                                             "in vivo", "..."] },
  "provenance": { "build_id": "...", "code_git_sha": "...", "built_utc": "...",
                  "input_manifests": {...}, "environment": {...} },
  "files": { "<filename>": {"bytes": <int>, "sha256": "..."} },
  "supersedes": { "atlas": "v5 / 09ed804a40836f4a05a91ba10900cded",
                  "status": "candidate | replacement | not-adopted" }
}
```

The **`validation_boundary`** block is new relative to V5 and is deliberately part of the
frozen artefact. It states, in machine-readable form, where this atlas has been validated and
where it has not — so an engine can flag out-of-boundary input rather than silently returning
a confident-looking number for a SERS spectrum or an in-vivo measurement.

The **`supersedes`** block is what carries the Phase-06 replacement decision. Until Phase 06
delivers a passing evaluation, every V7 atlas is `"status": "candidate"`.

---

## 4. Per-phase manifests

Every phase writes `results/manifests/phase_<NN>_manifest_v1.json`:

```
{ "schema": "gaira_v7_phase_manifest_v1",
  "phase": "<NN>", "phase_name": "...", "build_id": "...", "built_utc": "...",
  "inputs":  [ {"artifact_id": "...", "path": "...", "sha256": "..."} ],
  "config":  { ...all parameters... },
  "seeds":   { "numpy": <int>, "sklearn": <int>, "resampling": <int> },
  "code":    { "git_sha": "...", "dirty": false, "entry_point": "..." },
  "environment": { "python": "...", "numpy": "...", "scipy": "...",
                   "sklearn": "...", "blas": "...", "platform": "..." },
  "outputs": [ {"artifact_id": "...", "path": "...", "sha256": "..."} ],
  "gates":   [ {"gate": "...", "passed": true, "evidence": "...", "value": ...} ],
  "decisions": [ {"decision": "...", "rule_preregistered_in": "...",
                  "chosen": "...", "alternatives": [...], "rationale": "..."} ] }
```

Two fields carry unusual weight:

- **`code.dirty`** — a manifest produced from an uncommitted working tree is not evidence of
  anything, because the code that produced it cannot be recovered. `dirty: true` invalidates
  the phase for gate purposes.
- **`decisions[].rule_preregistered_in`** — every model-selection decision must name the
  document and section that stated the rule *before* the sweep ran (P-12). A decision with no
  pre-registration pointer is a post-hoc choice and must be labelled as one.

---

## 5. Versioning

Semantic versioning `v7.MAJOR.MINOR.PATCH` on the atlas:

| Change | Bump | Fingerprint changes |
|---|---|---|
| CSM basis, `S`, theme set, preprocessing spec, BSV reference, OOD support | **MAJOR** | yes |
| LSM dictionary (evidence layer only) | **MAJOR** | yes — it is in the fingerprint |
| Registry metadata, provenance, README, visualisation transform | **MINOR** | no |
| Typo fix in non-behavioural metadata | **PATCH** | no |

**Rules.**

- A frozen atlas is never edited in place. Ever.
- Any fingerprint change requires a new version directory and a migration note stating what
  changed, which layer, and what the downstream effect is.
- Downstream consumers pin an atlas version **and** verify its fingerprint on load — the V5
  engine already does this and V7 keeps the behaviour.
- Two atlases with the same fingerprint must produce byte-identical output on identical input.
  This is testable and must be tested.

---

## 6. Storage policy

| Artefact | In Git? | Rationale |
|---|---|---|
| Atlas bundle (basis, `S`, registries, manifests) | **yes** | it is the scientific product; small; must clone-and-run |
| Phase manifests, tables, figures, reports | **yes** | the evidence trail |
| Raw spectra | **no** | `data/`, `GAIRA_DATA/`, `/Volumes/` are gitignored |
| Large intermediates (full sweep tensors, per-run NMF outputs) | **no** | regenerable from manifests; keep the summary tables |
| PDFs | **no** | repo policy gitignores `*.pdf`; Markdown + SVG are tracked instead |

**Note on `checkpoints/`.** The root `.gitignore` ignores `checkpoints/` globally. A scoped
`GAIRA_v7_rebuild/.gitignore` re-includes `GAIRA_v7_rebuild/results/checkpoints/` so V7 atlas
bundles can be tracked, while still excluding bulk intermediates by extension.

**Size discipline.** The V5 bundle is ~400 KB and holds a complete inference-capable model.
V7 will be larger (two dictionary levels), but the same discipline applies: if a file is not
needed for inference or for the evidence trail, it does not go in the bundle.

---

## 7. What must never be modified

The V5 frozen atlas is **read-only for the entire duration of V7**:

| Asset | Fingerprint / hash |
|---|---|
| `assets/foundation/manifold_components.npz` | `ca385847146c7a9b72bd5c7ecfae85105ecf8740e43abe4a83ce894587444b9f` |
| `assets/foundation/` atlas fingerprint | **`09ed804a40836f4a05a91ba10900cded`** |
| all other `assets/foundation/` files | per `assets/foundation/MANIFEST.json` |
| `results/v5_rebuild/**` | V5 artefacts |
| `results/v6_rebuild/**` | V6, V6.2, V6.3 artefacts |
| `src/gaira/engine/**` | production inference engine |
| `src/gaira/preprocessing/**` | canonical preprocessing |
| existing Streamlit apps | production surfaces |

V7 **reads** these. V7 never writes them. The V5 atlas remains in production until Phase 06
produces evidence that a V7 candidate clears the pre-registered replacement bar — and if it
does not, the V5 atlas simply stays.
