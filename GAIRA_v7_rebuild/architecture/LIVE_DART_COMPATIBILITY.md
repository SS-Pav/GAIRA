# GAIRA V7 — Live Projection and DART Compatibility

Requirements the V7 representation must satisfy to support live projection, future
Raman→SERS observation models, and DART perturbation trajectories.

**None of this is implemented in V7.** These are forward constraints on the design, stated
now so that Phases 01–05 do not build something that forecloses them.

---

## 1. Live projection

**Requirement.** A spectrum arriving now must land in exactly the same coordinate system as
one projected a year ago, with no fitting, no batch dependence, and no lab volume mounted.

| Property | Implication for the design |
|---|---|
| Fixed dictionary | the CSM basis is frozen; NNLS against it is the only projection |
| Batch independence | no cross-spectrum statistics anywhere in the inference path |
| Bounded latency | one NNLS solve (M × 676) plus two matrix multiplies — the cost is set by `M`, which is why `M` is selected with a redundancy penalty and not allowed to proliferate |
| Streaming-safe | state is per-spectrum; nothing accumulates that changes later outputs |
| Portable | atlas bundle is self-contained |

**Design consequence.** Motif proliferation is not only an interpretability problem
(risk R-04) — it is a latency and stability problem for live use. A CSM count chosen for
marginal reconstruction gain is paid for on every projection forever.

---

## 2. Future Raman→SERS observation model

**The architectural position, stated once and binding:**

> The Raman-derived representation is the **latent biochemical state**.
> SERS is a **measurement channel** applied to that state.

Formally, the intended future structure:

```
latent state        z = BSV(x_raman)              ← learned from pure Raman only
observation model   x_sers ~ g(x_raman ; substrate, excitation, chemistry)
inference           given x_sers, infer z through g
```

`g` is learned **later**, from paired Raman↔SERS material, and it is **never** allowed to
shape the Raman foundation.

**Why this is not a stylistic preference.** SERS enhancement is selective by surface affinity,
orientation, and resonance — not by concentration or chemical importance. GAIRA's own prior
work established the concrete failure: on Ag colloid, 50 of 51 analytes homogenise onto a
purine-like attractor, and a raw theme cosine of 0.92 is a *baseline artefact* requiring null
correction rather than evidence of preservation. A foundation fitted on SERS would encode
substrate preference as chemistry.

**What V7 must therefore preserve:**

| Requirement | Reason |
|---|---|
| Non-negative additive representation | enhancement is (approximately) multiplicative per mode; a non-negative additive latent state composes with a multiplicative channel model cleanly, and a signed latent state does not |
| Band-resolved provenance on every CSM | `g` must be expressible per band/mode — impossible if a CSM's band support is unknown |
| LSM layer retained | mode-level structure is where enhancement selectivity acts; the LSM layer is the finest structure available |
| Raman-only fitting | the exclusion list in the corpus card stays enforced |
| Explicit validation boundary in the manifest | so a SERS spectrum projected through a Raman atlas is flagged, not silently scored |

**Design consequence — do not discard the LSM layer after Phase 02.** It is tempting to treat
LSMs as scaffolding and ship only CSMs. The observation model will need mode-level structure.
The LSM dictionary is therefore part of the frozen atlas and part of its fingerprint.

---

## 3. DART trajectories

DART applies a controlled perturbation and observes the biochemical response over time. The
representation must support:

$$\mathrm{BSV}(E, t) = \big[t_1(E,t), \ldots, t_K(E,t)\big]$$

for stimulus `E` and time `t`.

| Requirement | Why | Design consequence |
|---|---|---|
| **Absolute coordinates** | a trajectory is only meaningful if every point is in one fixed frame | the BSV is absolute (P-08); `BSV(E,t₂) − BSV(E,t₁)` is a well-defined ΔBSV precisely because both are absolute |
| **Comparable across time** | frame must not drift within a run | frozen atlas, no re-fitting, no per-batch normalisation |
| **Continuous** | dose-response and kinetics need continuity, not class flips | continuous non-negative activations, never argmax |
| **Uncertainty per point** | a trajectory of point estimates hides whether a change is real | uncertainty propagated to every BSV, and it is per-axis |
| **Non-negativity** | "negative lipid chemistry" is meaningless at a time point | non-negative throughout |
| **Support-aware** | a trajectory on a singleton-supported axis is weak evidence | singleton/anchor status inflates that axis's uncertainty (§4 of `INFERENCE_MODE_ARCHITECTURE.md`) |

### Trajectory quantities and their names

| Quantity | Definition | Nature |
|---|---|---|
| Trajectory | `{BSV(E, t_i)}_i` | sequence of **absolute** vectors |
| Displacement | `BSV(E, t_i) − BSV(E, t_0)` | **ΔBSV** — signed, derived |
| Velocity | `d BSV / dt` (finite difference) | signed, derived |
| Path length | `Σ_i ‖BSV(t_{i+1}) − BSV(t_i)‖` | scalar summary |
| Endpoint displacement | `‖BSV(t_end) − BSV(t_0)‖` | scalar summary |
| Response direction | normalised displacement | unit vector |

**Naming discipline.** Every quantity in this table except the first is a *derived* signed
quantity. None of them is a BSV. Conflating a displacement with a BSV is a correctness bug,
not a naming quibble — it would place a difference into an absolute coordinate frame.

### What DART must not do

- **No trajectory-specific re-fitting.** The axes are fixed before the experiment starts.
- **No per-run normalisation.** It would make trajectories from different runs incomparable —
  the exact property DART exists to exploit.
- **No hard classification per time point.** A trajectory through label space discards the
  continuous information that makes kinetics readable.
- **No cohort standardisation inside a trajectory.** That is a visualisation, computed at
  the end, on top of absolute values.

---

## 4. Forward compatibility checklist

Every V7 phase must preserve all of:

| # | Property | Phase most at risk |
|---|---|---|
| 1 | Non-negativity end to end | 03 (consensus operator), 04 (mapping) |
| 2 | Fixed dictionary; no inference-time fitting | 06 |
| 3 | Batch independence | 06 |
| 4 | Absolute BSV; deltas separately named | 05 |
| 5 | Band-resolved provenance on every CSM | 03 |
| 6 | LSM layer retained in the frozen atlas | 03, 06 |
| 7 | Per-axis uncertainty, support-aware | 05 |
| 8 | Continuous outputs; no argmax in the representation | 04, 05 |
| 9 | Raman-only fitting; SERS excluded | 01, 02 |
| 10 | Self-contained portable bundle | 06 |
| 11 | Declared validation boundary in the manifest | 06 |

**A property lost here is a capability lost later.** Non-negativity dropped in Phase 02 for a
cleaner consensus operator would quietly break the observation-model composition in §2.
Dropping the LSM layer in Phase 05 to slim the bundle would remove the mode-level structure
the SERS channel needs. These are the specific ways a good local decision becomes a bad
architectural one, and they are why the checklist is stated before implementation rather
than discovered after it.
