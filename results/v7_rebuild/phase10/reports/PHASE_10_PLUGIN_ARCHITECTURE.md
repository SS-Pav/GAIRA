# GAIRA V7 — Plugin Architecture

**Specifications only.** `gaira.v7.plugins` performs no inference and never may. Every
unimplemented adapter **raises `NotImplementedAdapter`**; a test asserts that none returns a
fabricated result, because a stub that produces plausible numbers is worse than no stub at all.

---

## The four contracts

| protocol | runs | may do | may never do |
|---|---|---|---|
| `ModalityAdapter` | **before** the core | correct, veto, pass through | touch the dictionaries, retrieval or chemistry model |
| `SampleContextAdapter` | **after** the core | add caveats and framing | change a number — the protocol returns no evidence field |
| `InterpretationAdapter` | after the core | rephrase | compute a scientific quantity |
| `TrajectoryAdapter` | over a sequence of results | analyse change | recompute any individual result |

The line: **scientific representation ≠ domain interpretation.**

## Modality adapters

| adapter | status | what a working implementation must supply |
|---|---|---|
| `PureRamanAdapter` | **implemented** | identity — the core's validated domain |
| `AgSERSAdapter` | contract only | a silver observation model (enhancement is analyte- and orientation-dependent, not a constant); a detection gate — Ag homogenises many analytes onto a purine attractor; a validated transfer function; its own held-out corpus |
| `AuSERSAdapter` | contract only | the gold equivalents — chemisorption chemistry differs from silver's |
| `SERSGenericAdapter` | contract only | substrate identification; "SERS" without a named substrate is not a well-defined channel |
| `DARTAdapter` | contract only | a mass-spectrometric to vibrational correspondence, which does not exist as a spectral transform — and a decision about whether DART belongs at this layer at all |

**Why the boundary is hard.** Phase 04 measured a Raman motif dictionary reconstructing **real
Ag-SERS** at AUROC 0.548. A non-negative Raman basis reconstructs SERS of the same metabolites
comfortably, so a SERS spectrum run through the Raman core produces confident numbers with no
validated meaning. Unsupported modalities are therefore **blocked**, not warned.

## Sample-context adapters

| adapter | status | open questions |
|---|---|---|
| `PureAnalyteContext` | **implemented** | — |
| `MixtureContext` | contract only | do activation shares track component proportions? unmeasured, and L2 normalisation removes absolute scale first |
| `SerumContext` | contract only | which analytes are visible at physiological concentration; albumin dominance; a serum corpus |
| `PlasmaContext` | contract only | everything serum needs, plus the anticoagulant's signature |
| `EVContext` | contract only | membrane vs cargo attribution; lipid saturation; isolation-method confounding |
| `BacteriaContext` | contract only | does envelope abstraction survive transfer; growth-phase confounding |
| `TissueContext` | contract only | spatial heterogeneity; fixation artefacts; is a pixel the right unit |

## Why DART is downstream

DART has no vibrational correspondence to Raman, so it is not a modality transform. It is a
**trajectory** over an orthogonal measurement, and a trajectory of CSM activations is meaningful
only if every activation was produced by the same frozen path. Placing it upstream would mean
inventing a spectral transform that does not exist.

## Adding a format, modality or context

1. Implement the protocol.
2. Register it (`adapters.ADAPTERS`, `modality.REGISTRY`, `context.REGISTRY`).
3. Add a held-out validation corpus for that domain — **no V7 number transfers**.
4. Add tests, including one asserting the adapter is honest about what it has not validated.

No other GAIRA module changes.
