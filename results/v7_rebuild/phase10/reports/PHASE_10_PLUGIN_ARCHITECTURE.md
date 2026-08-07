# GAIRA V7 — Plugin Architecture

**Specifications only.** `gaira.v7.plugins` performs no inference and never may. Every
unimplemented adapter **raises `NotImplementedAdapter`**; a test asserts that none returns a
fabricated result, because a stub that produces plausible numbers is worse than no stub at all.

---

## The four contracts

| protocol | runs | handles | may do | may never do |
|---|---|---|---|---|
| `ModalityAdapter` | **before** the core | the physics between sample and spectrum — substrate observation models, detection gates, transfer functions, wavelength corrections | correct, veto, pass through | touch the dictionaries, retrieval or chemistry model |
| `SampleContextAdapter` | **after** the core | the biology around the measurement — domain caveats, interpretation framing, dataset context | add caveats and framing | change a number — the protocol returns no evidence field |
| `InterpretationAdapter` | after the core | narration | rephrase | compute a scientific quantity |
| `TrajectoryAdapter` | over a **sequence** of results | time and perturbation — **this is where DART belongs** | analyse how the frozen representation moves | recompute any individual result |

The line: **scientific representation ≠ domain interpretation.**

## Modality adapters

| adapter | status | what a working implementation must supply |
|---|---|---|
| `PureRamanAdapter` | **implemented** | identity — the core's validated domain |
| `AgSERSAdapter` | contract only | a silver observation model (enhancement is analyte- and orientation-dependent, not a constant); a detection gate — Ag homogenises many analytes onto a purine attractor; a validated transfer function; its own held-out corpus |
| `AuSERSAdapter` | contract only | the gold equivalents — chemisorption chemistry differs from silver's |
| `SERSGenericAdapter` | contract only | substrate identification; "SERS" without a named substrate is not a well-defined channel |

**DART is deliberately absent from this table.** It is not a modality. See the next section.

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

## DART — a dynamic perturbation layer, not a modality

**DART is not a new spectral modality.** It is a **dynamic perturbation protocol built on
Raman/SERS measurements**, and treating it as a modality misdescribes both the instrument and the
architecture.

DART-Met produces

```
    I(wavenumber, potential, time)
```

which is **still a vibrational measurement**. Every slice through that volume is a Raman or SERS
spectrum the frozen engine already reads correctly. There is no spectral transform to invent,
because nothing about the measurement axis has changed — what has been added is *perturbation*
and *time*.

### The correct architecture

```
    Dynamic Raman / SERS acquisition       I(wavenumber, potential, time)
                 ↓
    Frozen canonical preprocessing         unchanged
                 ↓
    Frozen LSM projection                  unchanged
                 ↓
    Frozen CSM projection                  unchanged — the canonical representation
                 ↓
    Frozen Chemistry Evidence              unchanged
                 ↓
    TrajectoryAdapter                      NEW: how the representation moves under perturbation
                 ↓
    Dynamic biochemical interpretation
```

DART therefore attaches at the **`TrajectoryAdapter`** layer, downstream of the frozen
representation, consuming a *sequence* of ordinary `InferenceResult` objects with their potential
and time coordinates. It does not attach at the modality layer, and no `DARTAdapter` should exist
there.

### Why downstream is the right placement, not merely a convenient one

A trajectory of CSM activations is interpretable **only if every activation along it was produced
by the same frozen path**. Placing the dynamic layer downstream is what guarantees that: each
point on the trajectory is an ordinary, fingerprint-verified inference, and what the trajectory
layer analyses is how those verified coordinates *move*.

Placed upstream, a DART "modality adapter" would have to collapse the potential and time axes
before the core ever saw the data — discarding exactly the information the protocol exists to
produce, and inventing a transform to do it.

### What a working trajectory layer must supply

| requirement | why |
|---|---|
| an ordering over results — potential, time, or both | a trajectory without an axis is a scatter of points |
| a distance or motion measure in CSM space | the frozen representation is where motion should be measured, not the raw spectrum |
| a stability criterion distinguishing real motion from measurement noise | Phase 09 measured replicate consistency at 0.8927; motion below that scale is not evidence |
| its own held-out validation | **no static V7 number transfers to a dynamic claim** |

### Known divergence (documentation vs code)

`src/gaira/v7/plugins/modality.py` still registers a `DARTAdapter` at the modality layer, and its
refusal message describes a *"mass-spectrometric to vibrational correspondence"*. **That
description is wrong and this section supersedes it.** The adapter raises rather than running, so
no incorrect science can result from it, but the string is runtime code and the runtime is frozen.

The correct resolution — retire the modality-layer `DARTAdapter` and implement DART through
`TrajectoryAdapter` — is a **post-freeze** change. It is recorded here so the repository is honest
about the discrepancy rather than silently inconsistent.

## Adding a format, modality or context

1. Implement the protocol.
2. Register it (`adapters.ADAPTERS`, `modality.REGISTRY`, `context.REGISTRY`).
3. Add a held-out validation corpus for that domain — **no V7 number transfers**.
4. Add tests, including one asserting the adapter is honest about what it has not validated.

No other GAIRA module changes.
