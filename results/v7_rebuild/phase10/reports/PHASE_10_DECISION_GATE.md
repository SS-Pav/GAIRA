# GAIRA V7 — Phase 10 Decision Gate

```
========================================================
GAIRA V7 PHASE 10 — RUNTIME PLATFORM DECISION GATE
========================================================
```

| criterion | verdict |
|---|---|
| Frozen Phase 09 engine verified | **PASS** — 9/9 freeze gates; 4 declared fingerprints recomputed and matched (Scientific Atlas Fingerprint `09ed…`); 10 content digests pinned; Frozen Runtime Content Hash `2e43…` |
| Scientific outputs unchanged | **PASS** — max deviation **0.0** across all 375 spectra |
| Runtime service implemented | **PASS** — `gaira.v7.runtime.GAIRAService` |
| Typed public schemas implemented | **PASS** — `gaira.v7.contracts`, pydantic v2, 27 models |
| Python SDK implemented | **PASS** — `GAIRA.load()` / `.shared()` / `.remote(url)` |
| FastAPI implemented | **PASS** — 6 versioned routes + OpenAPI |
| MCP implemented | **PASS** — 8 read-only tools over stdio |
| Streamlit implemented | **PASS** — 5 pages, thin client, 717 lines, zero science |
| PDF/JSON report generation implemented | **PASS** — one generator, five callers |
| **Cross-surface numerical parity** | **PASS** — **max discrepancy = 0.0** (60 comparisons, 6 surfaces, tolerance 1e-12) |
| No duplicated scientific inference logic | **PASS** — AST-enforced across API, MCP, SDK, CLI, Streamlit |
| No LLM/cloud dependency | **PASS** — statically verified across all Phase 10 packages |
| Raman-only scientific scope preserved | **PASS** — non-Raman blocked on all five surfaces |
| Future modality extension contracts defined | **PASS** — 1 implemented, 3 declared SERS variants, all stubs raise. **DART is not a modality** — it attaches at the trajectory layer |
| Future sample-context contracts defined | **PASS** — 1 implemented, 6 declared, all stubs raise |
| Unsupported modalities cannot silently run as Raman | **PASS** — ERROR severity, not a warning |
| Local clean-clone inference works | **PASS** — 10 committed files, ~10 MB, all git-tracked |

**SSD_Rad required for normal inference? — NO.** `GAIRAEngine.load()` was instrumented; it opens
exactly ten files, every one inside the repository. A test asserts no frozen asset resolves to
`/Volumes/`.

| measurement | value |
|---|---|
| Single-spectrum latency | **2.34 ms** median (p95 3.11 ms); engine load 0.28 s |
| API latency overhead | **1.61 ms** (median 3.95 ms) |
| MCP latency overhead | **2.97 ms** (median 5.30 ms) |
| 100 sequential spectra | 0.218 s |
| Report generation | PDF 1.42 s · JSON 0.002 s |
| Full test suite | **1445 passed, 1 skipped, 0 failed** (Phase 10 contributes 124) |

| assessment | verdict |
|---|---|
| Scientific validation | **PASS** — every Phase 09 figure reproduced at Δ = 0.0 |
| Engineering validation | **PASS** — 17/17 gates, parity 0.0, concurrency safe |
| Platform stability | **PASS** — deterministic, stateless, frozen-verified, immutable results |

---

## Outstanding scientific risks

1. **No validated open-set detection** (carried from Phase 09, unchanged). The engine cannot
   determine that the true molecule is absent from its 154-molecule bank; white noise reconstructs
   at CSM EV 0.6083, above the 0.50 warning floor. Now frozen as a golden fixture so a change in
   this behaviour breaks a test rather than passing silently.
2. **Class-prior bias (R-01) remains open** — class sizes range from 3 to 80 spectra.
3. **Molecule top-1 is capped at 0.819 by corpus structure** — 66 of 154 molecules have a single
   spectrum.
4. **Every applied regime is unmeasured** — SERS, serum, plasma, EV, bacteria, tissue, and
   dynamic (DART) acquisition. Contracts exist; validation does not.
5. **The 16 classes are a reporting convention**, not a discovered structure (Phase 06.5).

## Outstanding engineering risks

1. **No real instrument export has been through the adapters** — Renishaw, B&W Tek and Horiba
   files were unavailable. This is where a research platform meets its messiest reality.
2. **Validation thresholds are reasoned, not swept** — declared before testing and untuned, which
   makes them admissible, but no false-positive rate is attached.
3. **No production hardening** — no auth, rate limiting, TLS or audit log. Loopback by default.
4. **PDF bytes depend on the matplotlib version**; content is reproducible, bytes are reproducible
   only on a fixed environment.
5. **The Docker image is not bit-reproducible** — dependency floors, not pins. It does verify the
   atlas at build time and fails the build otherwise.
6. **Two documentation-vs-code divergences in the frozen interface**, both recorded in
   `PHASE_10_ARCHITECTURE.md` §9 and neither affecting any computed number: the field
   `atlas_fingerprint` carries the Frozen Runtime Content Hash rather than the Scientific Atlas
   Fingerprint, and `DARTAdapter` is still registered at the modality layer with a
   mass-spectrometric rationale that the documentation now supersedes. Both require a runtime
   change and are therefore post-freeze work.

## Required fixes

**None.** All 17 gates pass and no defect remains open. The six engineering risks are scope
limits or frozen-interface naming, and each is stated where a user will meet it.

---

## Overall Phase 10 confidence: **9 / 10**

Ten out of ten that packaging changed no science — that claim is checkable and was checked at
deviation 0.0. The deduction is for the adapters: no real instrument export has been parsed, and
file handling is the one part of this platform whose correctness cannot be argued from first
principles.

---

## READY FOR OPTIONAL LLM/AGENT INTEGRATION? — **CONDITIONAL**

The platform is ready to be *called* by an agent. What is not yet validated is what an agent would
*say*.

### Which stable MCP tools should be exposed

All eight, in this order of trust:

| tool | exposure |
|---|---|
| `gaira_engine_info` | **expose first, and require it** — carries the scope and the limitations |
| `gaira_validate_spectrum` | expose — cheap, and teaches the agent what is refusable |
| `gaira_infer_spectrum` | expose — the primary call |
| `gaira_get_chemistry_evidence` | expose |
| `gaira_get_molecular_evidence` | expose |
| `gaira_explain_result` | expose — this is how an agent justifies rather than asserts |
| `gaira_compare_spectra` | expose |
| `gaira_generate_report` | expose |

### The permitted chain

```
        LLM  →  MCP  →  Frozen Runtime  →  Frozen Engine
```

and never

```
        LLM  →  scientific computation
```

### What the agent may do

**Choose tools · explain · compare · narrate · summarise.** Concretely: decide which tools to
call and in what order, request more evidence before answering, rephrase the deterministic
interpretation, compare results it obtained through the tools, and surface caveats verbatim.

### What the agent is forbidden from doing

**Computing chemistry · computing similarity · estimating concentrations · re-ranking ·
diagnosing disease · modifying inference.** It must additionally never assert a molecular
identification, treat low confidence as evidence of novelty, or drop a scope warning. **Every
number it states must be traceable to an `InferenceResult` field**, and every claim about a
molecule must carry the word *analogue* or an equivalent.

### What must be validated before enabling agentic interpretation

1. **An adversarial overclaim benchmark.** Give a model real tool output plus a leading question
   ("does this patient have cancer?", "how much glucose is present?") and measure how often it
   overclaims. This is the actual risk and nothing in Phase 10 measures it.
2. **A citation check** — every numeric claim in generated text must resolve to a result field.
3. **A caveat-retention check** — the scope warning, the relative-evidence caveat and the
   analogue caveat must survive into the agent's output.
4. **A refusal check** — the agent must decline to answer questions the engine cannot support
   (concentration, diagnosis, absent-molecule detection) rather than hedging.
5. **A determinism boundary** — the agent's narration will not be deterministic. The report
   generator must remain the system of record, with agent text clearly marked as commentary.

The engineering guardrails are in place. The missing guardrail is linguistic, and it belongs at
the top of Phase 11 rather than at the end.

---

## Recommendation

- [ ] Repeat Phase 10
- [ ] Minor fixes required
- [x] **Fixes implemented and validated** — four defects found and fixed during the phase
- [x] **FREEZE GAIRA V7 RUNTIME**
- [x] **READY TO DESIGN PHASE 11 AGENT LAYER** — conditional on the five validations above

```
========================================================
```
