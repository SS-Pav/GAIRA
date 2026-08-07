# GAIRA V7 — HTTP API Specification

`gaira.v7.api` · FastAPI · OpenAPI at `/docs` and `/openapi.json`

Start: `gaira serve` or `python -m gaira.v7.api` (default `127.0.0.1:8000` — loopback only).

The engine loads once at startup, verifies ten frozen asset digests, and is shared read-only. It
holds no mutable state and draws no random numbers, so concurrency needs no lock on the science:
16 concurrent requests over 8 threads reproduce the serial digests exactly.

---

## Routes

### `GET /v1/health`
```json
{"status": "ok", "engine_loaded": true, "frozen_assets_verified": true,
 "n_frozen_assets": 10, "detail": {"atlas_fingerprint": "2e43ddcc…"}}
```
`status` is `degraded` if any frozen asset fails verification.

### `GET /v1/engine`
`EngineInfo`: versions, all four frozen fingerprints, atlas shape (50 LSMs / 49 CSMs / 154
molecules / 16 axes), grid, corpus, `validated_performance`, `supported_modalities`,
`validated_sample_types`, `known_limitations`.

### `POST /v1/validate-spectrum`
```json
{"spectrum": {"wavenumber": [...], "intensity": [...]},
 "metadata": {"modality": "raman", "sample_type": "pure"}}
```
Returns `ValidationResult` — `can_run` plus three-severity diagnostics. Runs no inference.

### `POST /v1/infer`
```json
{"spectrum": {"wavenumber": [...], "intensity": [...]},
 "metadata": {"modality": "raman", "sample_type": "pure", "excitation_nm": 785,
              "sample_id": "S-001", "source_name": "run3.csv", "notes": null},
 "options": {"top_k_molecules": 10, "include_lsm": true, "include_csm": true,
             "include_provenance": true, "include_audit": true,
             "include_reconstruction": false, "already_preprocessed": false}}
```
`200` → `InferenceResult`. `422` → validation failed (`detail.code = "spectrum_rejected"`,
`detail.validation` carries every diagnostic). Unknown fields are **rejected**, not ignored.

### `POST /v1/compare`
`{"a": <InferenceRequest>, "b": <InferenceRequest>, "label_a": "…", "label_b": "…"}` →
`ComparisonResult`: both full results, CSM cosine, chemistry cosine, 16 per-axis deltas, shared
top molecules, Jaccard rank agreement, deterministic text, and a scope note stating that V7 does
not license a claim about biological state change.

### `POST /v1/report`
`{"format": "json"|"html"|"pdf", "inference": <InferenceResult>}` — or `"request":
<InferenceRequest>` to run the spectrum first. JSON and HTML return inline; PDF returns
base64 in a JSON envelope with a **derived** filename. **No route accepts a filesystem path.**

---

## Semantics that matter

| | |
|---|---|
| `chemistry.evidence_l1` | **relative biochemical evidence** — not a concentration, abundance or mixture fraction |
| `retrieval.top[]` | reference **analogues**; validated top-1 is 0.6053 |
| `confidence.unknown_warning` | an unexplained *spectrum*, not an unknown *molecule* |
| `audit.open_set_limitation` | present on every result; must be surfaced |
| `result_digest` | MD5 over the scientific fields — the cross-surface parity anchor |
| `metadata.sample_type` | recorded, warned on, and **never applied to the calculation** (tested) |
| `metadata.modality` | anything but `raman` is rejected with `422` |

## Errors

| status | meaning |
|---|---|
| 400 | `/v1/report` with neither `inference` nor `request` |
| 413 | body exceeds 32 MB |
| 422 | schema violation, unknown field, or spectrum rejected by validation |

## Performance (laptop, measured)

engine load 0.28 s · median inference 2.3 ms · **median API latency 4.0 ms, overhead 1.6 ms** ·
100 sequential spectra 0.22 s · PDF report 1.4 s · peak RSS 596 MB.
