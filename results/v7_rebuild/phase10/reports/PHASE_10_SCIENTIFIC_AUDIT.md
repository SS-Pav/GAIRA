# GAIRA V7 — Phase 10 Scientific Audit

Every claim this phase makes, sorted by how much weight it can bear, followed by the defects found
while producing it.

Phase 10 is an **engineering** phase with one scientific obligation: prove that packaging did not
change the science. That obligation is met to the digit. What it does *not* do is add scientific
evidence, and the audit is careful about the difference.

---

## A. Strongly supported

**A1. Packaging changed no number.** Molecule top-1, top-3, top-5, top-10, MRR, in-sample
chemistry top-1 and CSM mean explained variance all reproduce through the runtime path at
deviation **exactly 0.0**. This is a binary claim, checkable, and checked on all 375 spectra.

**A2. Six surfaces produce identical scientific output.** 60 comparisons at a 1e-12 tolerance,
**0 divergent, max |Δ| = 0.0**, over twelve spectra chosen for behavioural variety rather than
convenience. Result digests are equal across the service, SDK, API and Streamlit paths.

**A3. The science exists in exactly one implementation.** Enforced structurally: every surface is
AST-parsed and must reference no scientific primitive and import no scientific module. This is
stronger than a convention because it fails the build.

**A4. Local inference needs no external volume.** `GAIRAEngine.load()` was instrumented; it opens
exactly ten files, all inside the repository, all git-tracked. A test asserts every frozen asset
resolves under the repository root and contains no `/Volumes/` path.

**A5. Unsupported modalities cannot silently run as Raman.** Blocked on the API (422), the SDK
(`SpectrumRejected`), the MCP layer (`can_run: false`), the CLI (exit 2) and the Streamlit client
(red block, button disabled path). Verified on all five.

**A6. `sample_type` does not change a single number.** Four sample types over the same spectrum
produce one identical `result_digest`. The scope warning is metadata, and the test proves the
contract is not lying.

**A7. Concurrency is safe.** 16 requests over 8 threads reproduce the serial digests exactly —
structural, since the engine holds no mutable state and draws no random numbers.

**A8. Single-spectrum inference is interactive.** 2.34 ms median, 1.61 ms API overhead. Live
Raman use is feasible; acquisition is the bottleneck, not inference.

**A9. No stub fabricates a result.** All ten unimplemented adapters raise `NotImplementedAdapter`
with a statement of what a working implementation must supply.

---

## B. Weakly supported

**B1. "The report is reproducible."** Byte-identical apart from the generation timestamp, which
is isolated in the metadata block. The PDF path additionally depends on the matplotlib version —
a different version would render the same numbers into different bytes. The *content* is
reproducible; the *bytes* are reproducible on a fixed environment.

**B2. Performance figures.** Measured on one laptop, single process, warm engine. The 596 MB peak
RSS includes pytest and matplotlib; a bare API process will be smaller, and the figure was not
isolated.

**B3. "The adapters handle real instrument exports."** They handle CSV, TSV and two-column text
with detected delimiters, headers, column names and axis direction, tested against constructed
fixtures. **No real Renishaw, B&W Tek or Horiba export was tested**, because none was available.
The declared-but-unimplemented formats are honest about this; the implemented ones are validated
against synthetic cases only.

**B4. "The validation thresholds are right."** `MIN_POINTS = 32`, `MIN_COVERAGE_ERROR = 0.10`,
`MIN_COVERAGE_WARN = 0.70`, `MIN_DISTINCT_FRACTION = 0.05` are *reasoned*, not measured. They
were declared before testing and not tuned afterwards, which makes them admissible, but no sweep
established them. A different set would flag a different population.

**B5. "Cross-surface parity holds in general."** It holds on twelve locked spectra plus six more
in the test suite. It is not a proof over all inputs — though since every surface calls the same
function, the mechanism by which it could fail is narrow: a translation error in the service
layer, which the digest comparison would catch.

---

## C. Unsupported — do not claim these

**C1. Any new scientific capability.** Phase 10 added none. Every performance figure it displays
is quoted from Phase 09.

**C2. That the platform is production-hardened.** There is no authentication, no rate limiting, no
audit log, no TLS and no multi-tenant isolation. It binds to loopback by default because that is
the only posture it has been reasoned about in. Exposing it to a network is out of scope and
untested.

**C3. That validation catches bad spectra in general.** It catches the specific conditions
enumerated in `validation/spectrum.py`. A well-formed spectrum of the wrong substance, a
mislabelled sample, or a systematically miscalibrated instrument all pass cleanly.

**C4. Open-set detection, in any form.** Unchanged from Phase 09 and restated on every surface:
the engine cannot determine that the true molecule is absent from its bank. The golden fixture
freezes the measured behaviour — white noise reconstructs at CSM EV 0.6083, above the 0.50 floor.

**C5. That the MCP server is safe for an autonomous agent today.** It is read-only, local and
schema-validated, which is necessary and not sufficient. What an agent would *say* about the
results is the risk, and nothing in Phase 10 constrains that — see §F.

**C6. Docker image reproducibility.** The Dockerfile pins no dependency versions beyond floors, so
two builds a month apart may differ. It verifies the atlas at build time, which is the property
that matters scientifically, but the image is not bit-reproducible.

---

## D. Defects found and fixed during this phase

| # | defect | consequence had it stood | fix |
|---|---|---|---|
| 1 | **The freeze audit hand-rolled the leave-one-out retrieval loop** and dropped every spectrum of the query molecule instead of only the query spectrum. | It reported molecule top-1 of **0.0000** — an apparent catastrophic regression that was entirely an artefact of the audit. Chasing it would have wasted the phase; *believing* it would have falsely condemned the engine. | Replaced with calls to the frozen `gaira.v7.retrieval` modules. Deviation dropped to exactly 0.0. |
| 2 | **The text adapter desynchronised its columns.** A row whose wavenumber parsed and whose intensity did not appended the wavenumber and then failed, pairing every subsequent intensity with the wrong wavenumber. | A silently mangled spectrum that **still looks like a spectrum** — the worst failure mode an adapter has, because nothing downstream can detect it. | Parse both values before appending either. A regression test checks the pairing arithmetic explicitly. |
| 3 | **`load()` swallowed parse exceptions** with a bare `except Exception: continue`, reporting "unrecognised format" for a file the adapter had accepted. | Defect 2 was invisible for as long as this stood. It presented as a format problem, so the investigation would have started in the wrong place. | Only `sniff` is guarded now; a raising `parse` returns an explicit `input.parse_failed` diagnostic naming the adapter and the exception. |
| 4 | **The static architecture test flagged its own documentation.** A naive substring search for `NMF` matched the Streamlit docstring listing what the file excludes. | A permanently red test that would be disabled rather than fixed — and then the rule it encodes would be gone. | AST-based checking of referenced names and imported modules, ignoring strings and comments. Identical to the fix Phase 09 made for the same reason. |

**Defect 1 is the important one, and not because of its size.** The very first script Phase 10
wrote reproduced the exact failure mode the phase was created to prevent — reimplementing
scientific logic outside the engine — within twenty minutes of starting. The freeze audit was not
exempt from its own rule, and it broke the rule immediately. That is why P-19 is enforced by tests
rather than stated as a principle: the discipline does not survive contact with convenience.

**Defects 2 and 3 are one defect.** A bad exception handler hid a data-corruption bug. This is the
same shape as Phase 06.5's bare `except` around `silhouette_score`, which turned an uncomputable
index into `NaN` across an entire 56-row sweep. **Second occurrence in V7 of "a broad exception
handler concealed a real defect."** The countermeasure both times was the same: guard only the
operation that is *allowed* to fail, and let everything else surface.

---

## E. What a referee would ask next

1. **Test against real instrument exports** (B3). The adapters are validated on constructed
   fixtures. One Renishaw and one B&W Tek file would be worth more than ten more synthetic cases.
2. **Sweep the validation thresholds** (B4) against the corpus and any available field spectra, so
   the error and warning floors come with a false-positive rate rather than a rationale.
3. **Measure a bare API process's memory** (B2), isolated from the test harness.
4. **Pin the Docker dependency versions** (C6) if image reproducibility is wanted.
5. **Decide the security posture explicitly** (C2) before anything is exposed beyond loopback.

None of these changes the decision. All five would make it easier to defend.

---

## F. Assessment: is this ready for an agent?

The platform is ready to be *called* by an agent. It is read-only, local, schema-validated,
deterministic, and every tool carries its caveats in its own description.

The remaining risk is not in the tools — it is in the narration. Every failure mode this
architecture has been built to prevent is a failure of **language**: calling a retrieved analogue
an identification, calling relative evidence a concentration, calling low confidence a novel
molecule, or reading a serum spectrum as a clinical finding. An agent is a language system, and
those are precisely the sentences it would find natural to produce.

So the honest answer is **conditional**. The engineering is done; the guardrail that is missing is
a validation of what an agent *says*, not what it *calls*. That should be a gate on Phase 11, and
it should be adversarial: give a model the tool output and a leading question, and measure how
often it overclaims. Nothing in Phase 10 measures that, and Phase 10 should not pretend otherwise.

**Confidence that packaging changed no science: 10 / 10.** Checkable, and checked at deviation
0.0.

**Confidence that the platform is stable enough to build on: 9 / 10.** The deduction is B3 — no
real instrument export has been through the adapters, and file parsing is where a research
platform meets the messiest reality it will ever face.
