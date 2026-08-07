# GAIRA V7 — Phase 11: Interactive Scientific Demo

**Status: COMPLETE · 7 of 7 gates PASS · science unchanged, measured at max |Δ| = 0.0**

Phase 11 is presentation, interaction and visualisation. It adds no science, changes no number
and touches no frozen artefact. Its objective is to make GAIRA read as a scientific reasoning
engine rather than as software — and its obligation is to prove that nothing it displays differs
from what the frozen runtime returns.

| | |
|---|---|
| New app | `streamlit_apps/gaira_v7_demo.py` — 5 pages, 970 lines, **zero scientific computation** |
| Supporting modules | `theme.py` (visual system), `figures.py` (Plotly), `data.py` (display-only reference data) |
| Scientific calls | **exactly one** — `GAIRA.infer(...)`, asserted by test |
| Parity | 36 comparisons across **7 surfaces**, 0 divergent, **max |Δ| = 0.0** |
| Analysis latency | **3.52 ms** median (p95 4.24 ms) — the demo's exact call |
| Inference + all six figures | **130 ms** |
| Tests | 34 new; all 1039 V7 tests pass |
| Screenshots | 15, captured from the running app by a headless browser |

---

## 1. What was built

### The application

Five pages behind a top navigation bar, no sidebar, dark by default.

| page | contents |
|---|---|
| **Home** | Hero, four live stat tiles, three claims, scope footer |
| **Analyze** | Upload → Preprocess → Analyze → Report, as four staged screens |
| **Docs** | Fingerprints, corpus, validated performance, limitations — all pulled live from the running engine |
| **Architecture** | The seven stages, selectable, with what is deliberately absent and why DART is not a modality |
| **About** | What GAIRA is, how it differs, why CSM, why Chemistry Evidence, and why there is no LLM |

The results screen carries a verdict card, three columns (spectrum · chemistry · analogues) and
eight expandable sections: chemical evidence, CSM contributions, LSM view, reconstruction,
molecular retrieval, confidence, provenance, and downloads.

### The visual system

A single `theme.py` defines the palette, typography, glass surfaces, animations and the shared
Plotly layout, so every figure reads as one system. Streamlit's own components are darkened
through `.streamlit/config.toml` rather than fought with CSS — the base theme is what the
expanders, dataframes and selectboxes actually read.

### Every figure is Plotly

Thirteen figure constructors in `figures.py`, all built from an `InferenceResult` and nothing
else. No matplotlib anywhere in the demo — a test asserts it. (The PDF report generator still
uses matplotlib; that is the frozen Phase 10 module, not the UI.)

---

## 2. How honesty was preserved in an animated interface

This is the part of Phase 11 that required actual decisions rather than styling.

### The analysis sequence

The brief asked for a cinematic six-stage sequence, each step "tied to actual engine calls" and
never simulated. The engine's `infer()` returns everything in one call, in ~3.5 ms.

Three options were available:

1. Fake per-stage timing. **Rejected** — that is simulating outputs.
2. Call the engine's individual stage methods from the app. **Rejected** — the app would be
   orchestrating the inference path, which is the runtime's job and the exact duplication P-19
   forbids.
3. Make one sanctioned `infer()` call and reveal the real per-stage results in sequence.
   **Adopted.**

So the six stages you watch are a reveal of work that genuinely happened, with real data behind
each one. The app states the true latency rather than implying the animation is the computation.
No stage displays a number the engine did not return.

### The preprocessing animation

Only two curves exist and only two are drawn: what the user supplied, and what the **engine
returned** via `GAIRAEngine.prepare()` — the read-only accessor Phase 10 added so a client can
display what the projection consumed. No intermediate is interpolated for the animation. The raw
trace is scaled to its own maximum so both fit on one axis, and the caption says so.

### Display-only reference data

The demo needs the LSM and CSM basis *spectra* to draw a motif when a user clicks one. Those are
reference data, not results — the engine reads the same arrays to project. `data.py` loads them
from the pinned frozen artefacts and **calls `FREEZE.verify(strict=True)` before use**, so the
motif a user sees is provably the motif the engine used.

### Language

Enforced by test, not by discipline:

| required somewhere in the app | forbidden anywhere |
|---|---|
| "reference analogue" | "AI identifies your molecule" |
| "not a concentration" | "detects disease" |
| "open-set" | "clinical diagnosis" |
| "relative" | "diagnose" |

Retrieval is titled **Grounded Evidence Retrieval**, never *Molecule Identification*.

---

## 3. Validation

### Cross-surface parity — seven surfaces

The validation runs the **exact call the demo makes** —
`{"include_reconstruction": True, "top_k_molecules": 10, "already_preprocessed": True}` — and
compares it against every other surface, field by field, at a tolerance of 1e-12.

| comparison | identical |
|---|---|
| engine → runtime service | 7 / 7 |
| runtime service → Python SDK | 7 / 7 |
| Python SDK → HTTP API | 7 / 7 |
| MCP → matched SDK call | 7 / 7 |
| **SDK → Streamlit demo path** | **7 / 7** |
| **CLI subprocess → SDK** | **1 / 1** |

**36 comparisons, 0 divergent, maximum absolute difference 0.0.** Compared per case: all 49 CSM
activations, explained variance, all 16 chemistry evidence values, the L1 radar shares, all 16
calibrated probabilities, the predicted class, all ten retrieval ranks and similarities, overall
confidence, and the unknown flag.

The seven locked spectra are the seven the demo offers as built-in examples, so the validation
covers precisely what a user will run:

| example | predicted class | digest |
|---|---|---|
| cholesterol | sterol_steroid | `5ea50dbff5ac…` |
| cysteine | free_amino_acid | `51981c591bd3…` |
| (+)-glucose | mono_oligosaccharide | `73e0eccef0eb…` |
| palmitic acid | fatty_acid | `07f99bd04114…` |
| adenine | purine | `0bfa5eb2bc8c…` |
| albumin | peptide_protein | `b509fa61e0c5…` |
| pyruvate *(hard case)* | carboxylic_acid_metabolite | `d63d1eb429db…` |

The CLI arm is a **real subprocess** writing and reading a real file, not an in-process call.

### Performance

| | |
|---|---|
| engine load (once, cached) | 0.431 s |
| **demo analysis, median** | **3.52 ms** (p95 4.24, max 4.60) |
| all six figures built | 126.9 ms |
| **inference + every figure** | **130.4 ms** |
| PDF report | 2.16 s |
| JSON report | 0.004 s |

The brief's target — analysis under 200 ms after preprocessing — is met with two orders of
magnitude to spare. The figures cost 36× more than the inference, which is the honest shape of an
interactive scientific tool: the science is cheap and the drawing is not.

### Static architecture enforcement

Every demo module is parsed with `ast` and must reference no scientific primitive
(`nnls`, `NMF`, `PCA`, `savgol_filter`, `cosine_similarity`, …) and import no scientific module
(`scipy.*`, `sklearn.*`, `gaira.v7.{lsm,csm,chemistry,retrieval,…}`) — **including
`gaira.v7.canonical`**, because the demo must go through the runtime rather than reach past it.

Additional assertions: exactly one `client().infer(` call in the whole app; no LLM or cloud
import; Plotly and not matplotlib; `theme.py` imports nothing at all; `data.py` verifies the
freeze ledger; all sixteen axes have a plain-English description; the unsupported-modality block
is present.

### Gates

| gate | status |
|---|---|
| D1 every surface agrees with the demo's exact call | **PASS** |
| D2 result digests identical across service, SDK, API and demo | **PASS** |
| D3 the CLI subprocess agrees with the SDK | **PASS** |
| D4 analysis under 200 ms after preprocessing | **PASS** (3.52 ms) |
| D5 inference plus every figure under 1 s | **PASS** (130 ms) |
| D6 the demo computes no scientific quantity | **PASS** (static) |
| D7 the engine is loaded once and cached | **PASS** |

---

## 4. Defects found and fixed

All four were caught by driving the real application in a headless browser. None would have been
caught by unit tests, and three would have shipped as visible errors.

| # | defect | how it presented | fix |
|---|---|---|---|
| 1 | **`axref="paper"` is invalid in Plotly.** Used for the architecture flow arrows. | The entire Architecture page rendered a `ValueError` traceback instead of content. | Paper-anchored arrows express their tail as a **pixel** offset; `axref` omitted. |
| 2 | **The page-dispatch dict kept the old label.** The nav was shortened `Documentation` → `Docs`, but the dispatch was not. | `KeyError: 'Docs'` — the Docs page was completely inaccessible. | Dispatch key updated. A stale mapping is invisible until the exact path is walked. |
| 3 | **`width:100%;` collided with old-style `%` formatting**, producing `ValueError: unsupported format character ';'`. | Architecture page crash. | f-string instead of `%`-formatting. |
| 4 | **A Plotly `title` dict without `text` renders the literal string "undefined".** Introduced when the shared layout gained title defaults. | The word *undefined* appeared above three figures. | `text=""` in the shared layout; a test asserts it. |

### The tooling change that mattered

The first screenshot pass reported "No browser errors" while the Architecture page was displaying
a traceback. **Streamlit renders server-side Python exceptions into the DOM**, so a
`page.on("pageerror")` listener never sees them.

The capture script now reads the rendered body text and fails on `Traceback:`, `ValueError`,
`KeyError`, `AttributeError`, `TypeError`, `IndexError` and `undefined`. Defects 2, 3 and 4 were
all caught by that change, within one run of adding it. It is committed as
`code/capture_gallery.py` so the check runs whenever the gallery is regenerated.

### A recurring shape, for the fifth time

A test banning the substring `"def preprocess"` flagged `preprocessing_stages` — a *figure*
function. This is the same over-broad text matching that produced false positives in Phase 09
(a docstring listing excluded terms) and Phase 10 (the Streamlit docstring). The fix is the same
each time: **match the parsed name, not the text.** The Phase 11 test now compares exact function
names from the AST.

---

## 5. Deliverables

| deliverable | location |
|---|---|
| Streamlit app | `streamlit_apps/gaira_v7_demo.py` + `gaira_v7_demo/` |
| Streamlit base theme | `.streamlit/config.toml` |
| Validation + profiling | `code/run_phase11_validation.py` → `artifacts/phase11_validation_v1.json` |
| Gallery capture | `code/capture_gallery.py` |
| Screenshot gallery | `gallery/` — 15 screenshots + index |
| Tests | `tests/test_v7_phase11.py` — 34 |
| User guide | `docs/PHASE_11_USER_GUIDE.md` |
| Deployment SOP | `docs/GAIRA_RUNTIME_DEPLOYMENT_SOP.md` |
| Demo video script | `docs/PHASE_11_DEMO_VIDEO_SCRIPT.md` |
| This report | `reports/PHASE_11_REPORT.md` |

`requirements.txt` was updated so `pip install -r requirements.txt` actually produces a working
runtime — it was missing `pydantic`, `fastapi`, `uvicorn`, `httpx`, `mcp` and `pytest`, which
meant the SOP's own instructions could not have been followed from a clean clone.

---

## 6. Scientific validation

**Nothing changed.** No frozen artefact was touched, no fingerprint moved, no inference number
differs. The engine, runtime, API, SDK and MCP are byte-identical to their Phase 10.1 state; the
only files added are presentation, tests, documentation and validation artifacts.

| check | result |
|---|---|
| frozen artefacts modified | 0 |
| runtime / engine / API / SDK / MCP source modified | 0 |
| Scientific Atlas Fingerprint | `09ed804a40836f4a05a91ba10900cded` — unchanged |
| Frozen Runtime Content Hash | `2e43ddcca7d3be41c5f9da016fb8277f` — unchanged |
| Phase 10 parity | unchanged, 0.0 |
| V7 test suite | 1039 passed, 0 failed |

## 7. Engineering validation

| check | result |
|---|---|
| app renders end to end in a real browser | **yes**, verified by headless driver |
| server-side exceptions on any page | **none** |
| pages exercised | all 5, plus 5 expandable sections |
| concurrent-safe | inherits Phase 10 — engine is stateless and cached |
| memory | one engine instance per process via `st.cache_resource` |
| no scientific computation in the UI | **enforced by AST tests** |

## 8. Limits of this phase

1. **The gallery is one spectrum.** Fifteen screenshots of the cholesterol example. Other
   chemistry families render correctly — the validation covers seven — but only one is
   photographed.
2. **No cross-browser testing.** Chromium only.
3. **No accessibility audit.** The dark palette was designed for contrast but was not measured
   against WCAG, and no screen-reader pass was made.
4. **No mobile layout.** Three-column results assume a desktop viewport.
5. **The animation is a reveal, not a trace.** Stated plainly in §2 and in the app.
6. **PDF generation takes 2.2 s** and blocks the Streamlit thread. Acceptable for a demo, not for
   a multi-user deployment.

## 9. Recommendation

Phase 11 achieves what it set out to: the frozen engine is now approachable without becoming less
honest. Every scope limit that Phase 10 stated in a document is now stated on the screen where a
user will meet it — the modality block, the relative-evidence caveat, the analogue caveat, and
the open-set limitation on the results page rather than in a footnote.

**The runtime remains frozen and ready to tag.** Phase 11 changes nothing about that, and the
`GAIRA-V7-RUNTIME-FROZEN` tag can still be cut at the Phase 10.1 commit if a tag on the frozen
science alone is wanted.

For Phase 12 or a Phase 11.1, the highest-value work is accessibility and a second reviewer on
the language — every failure mode this architecture prevents is a failure of wording, and an
interface is where wording reaches a user most directly.
