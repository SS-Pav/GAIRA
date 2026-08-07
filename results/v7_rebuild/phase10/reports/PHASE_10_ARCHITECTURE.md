# GAIRA V7 — Phase 10 Architecture

The scientific architecture is frozen after Phase 09. Phase 10 wraps it in a runtime platform and
changes no science: cross-surface parity was measured at **max |Δ| = 0.0** and every Phase 09
number reproduces exactly through the runtime path.

---

## 1. The shape

```
                    ┌──────────────────────────────────────────────────────┐
                    │  FROZEN SCIENTIFIC ENGINE — gaira.v7.canonical       │
                    │  preprocess → LSM → CSM → retrieval → chemistry      │
                    │  → confidence → provenance          (Phase 09)       │
                    └──────────────────────┬───────────────────────────────┘
                                           │  the ONLY thing that computes
                    ┌──────────────────────┴───────────────────────────────┐
   adapters ───────▶│  RUNTIME SERVICE — gaira.v7.runtime                  │──────▶ reporting
   (CSV/TSV/TXT)    │  validate · call once · translate · template text    │        (PDF/HTML/JSON)
                    └──────────────────────┬───────────────────────────────┘
             ┌───────────┬─────────────────┼─────────────────┬───────────────┐
        Python SDK      CLI            FastAPI              MCP          Streamlit
```

## 2. The rules

| # | rule | enforcement |
|---|---|---|
| **P-19** | One implementation of every scientific quantity | AST-based static tests over every surface |
| **P-20** | A surface may narrow what it *shows*, never what it *computes* | `include_*` options change the view, not `result_digest` |
| A-09 | Raman-only scientific core | unsupported modality is an **ERROR**, not a warning |
| P-02 | Non-negativity at every layer | asserted in tests |
| P-04 | Provenance is first-class | every result carries the full chain |
| P-13 | No threshold adjustment after seeing results | Phase 05 warning thresholds untouched |

**P-19 is not theoretical.** The first script Phase 10 wrote — the freeze audit itself —
hand-rolled the leave-one-out retrieval loop, dropped every spectrum of the query molecule
instead of only the query spectrum, and reported molecule top-1 of **0.0000**. The phase
reproduced the failure mode it exists to prevent within twenty minutes of starting.

## 3. Packages

| package | responsibility | may compute? |
|---|---|---|
| `gaira.v7.canonical` | the frozen engine | **yes — and only here** |
| `gaira.v7.contracts` | typed public schemas (pydantic v2) | no |
| `gaira.v7.runtime` | freeze ledger, service, deterministic interpretation | no |
| `gaira.v7.adapters` | CSV / TSV / TXT / arrays; a protocol for more | no |
| `gaira.v7.validation` | three-severity input checks | no |
| `gaira.v7.reporting` | PDF / HTML / JSON — one implementation | no |
| `gaira.v7.api` | FastAPI transport | no |
| `gaira.v7.mcp` | 8 read-only tools | no |
| `gaira.v7.sdk` | Python client, local or remote | no |
| `gaira.v7.plugins` | modality / context / interpretation / trajectory **contracts** | never |
| `gaira.v7.cli` | command line | no |
| `streamlit_apps/gaira_v7_console.py` | thin client | no |

## 4. Two independent freeze layers

| layer | what it checks | when | failure |
|---|---|---|---|
| Phase 09 declared fingerprints | four values recorded *inside* `PHASE_STATE.json` / `csm_registry_v1.json` | engine load | `FrozenArtifactError` |
| **Phase 10 content digests** | MD5 of all **ten files** the engine opens | before the runtime serves | `FrozenAssetError` |

The second layer exists because the first answers *"did the producing phase claim this artefact?"*
and not *"is the file on disk still the file that phase wrote."* A dictionary could be edited in
place and the declared fingerprint would not move, because it lives in a different file.

## 5. The engine's one Phase 10 change

`GAIRAEngine.prepare()` was extracted from `infer()` — a pure refactor exposing the canonical
processed spectrum read-only, so a client can *display* what the projection consumed instead of
reimplementing preprocessing. The arithmetic is unchanged and the golden fixture digests are
byte-identical before and after.

## 6. Where a future agent connects

An LLM would sit **above** the MCP tool server, calling read-only tools and rephrasing numbers
the engine computed. It may never compute a similarity, an activation or a chemistry axis;
re-rank retrieval; assert a molecular identification; drop a scope warning; or infer a biological
state. Phase 10 ships no model and requires no cloud account.

## 7. Why DART belongs downstream

DART is a mass-spectrometric channel with no vibrational correspondence to Raman, so it is not a
modality *transform*. It is better modelled as a **trajectory** over an orthogonal measurement —
downstream of the spectral representation, because a trajectory of CSM activations means
something only if every activation was produced by the same frozen path.
