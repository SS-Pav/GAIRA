# GAIRA V7 — Phase 09 Decision Gate

Phase 09 was declared a **packaging phase** before any code ran. Its gate therefore differs in
kind from Phases 05–08: those phases asked *"is this change worth adopting?"*, and this one asks
*"is the frozen architecture faithfully implemented, honestly validated, and ready to be
declared final?"* Nothing here can adopt an architecture change, because the phase was forbidden
from proposing one.

---

## 1. Pre-registered criteria

Declared before the run, from the brief:

| # | criterion | outcome |
|---|---|---|
| C1 | All four frozen fingerprints verify; nothing upstream is regenerated | **MET** |
| C2 | No new representation, optimisation, retrieval strategy, clustering, dimensionality reduction, threshold or heuristic is introduced | **MET** |
| C3 | The engine is deterministic and holds no mutable state | **MET** |
| C4 | Every score reconciles to its displayed components | **MET** — 375 × 10 candidates |
| C5 | The frozen retrieval baseline is reproduced exactly | **MET** — all six metrics identical |
| C6 | Every spectrum in the corpus is processed, no exceptions | **MET** — 375 / 375 |
| C7 | All four layers are validated, with chemistry reported held-out | **MET** |
| C8 | BSV2, PCA, UMAP, clustering and latent geometry are absent from inference | **MET** |
| C9 | The radar is labelled relative evidence, never concentration | **MET** |
| C10 | Scope is pure Raman only | **MET** |

16 of 16 gates PASS. `phase09_gates_v1.csv` records each.

---

## 2. Decision

**Outcome A — ship the engine as implemented and freeze the V7 architecture.**

The three outcomes available to this phase were declared in advance:

- **A** — the engine is faithful and validated; freeze.
- **B** — the engine is faithful but a validation reveals a defect that packaging cannot fix;
  document it, ship with the limitation stated, and open a research phase.
- **C** — the engine cannot reproduce the frozen behaviour, meaning some earlier phase's result
  does not survive integration. This would invalidate a prior conclusion and require reopening it.

**C did not occur.** Retrieval reproduces Phase 05/08 to the digit (top-1 0.6053, top-3 0.7627,
top-5 0.7947, top-10 0.8107, MRR 0.6870, nDCG@5 0.7112). The chemistry layer, the CSM projection
and the LSM projection all reproduce their source phases. Integration surfaced no contradiction
between phases, which is itself the strongest single piece of evidence that the rebuild's chain
of decisions is sound.

**B partially occurred and was resolved inside the phase.** Two defects were found — an in-sample
Validation 4 and a pinned retrieval-calibration temperature — and both were fixed by making the
evaluation honest rather than by changing the engine. See `PHASE_09_SCIENTIFIC_AUDIT.md` §D.

---

## 3. What is now frozen

| component | status |
|---|---|
| canonical preprocessing (450–1800, 2.0, 676 bins, asLS, SG(9,3), L2) | **frozen** |
| 50-motif LSM dictionary | **frozen** |
| 49-motif CSM dictionary | **frozen** — the canonical representation (A-08) |
| retrieval: CSM cosine over the 154-molecule bank | **frozen** — Phase 08 outcome A |
| Chemistry Evidence: Model `D:A_max_idf:lam0.5` → 16 axes | **frozen** — A-19 |
| calibration: temperature, $T = 0.4538$ | **frozen** |
| confidence and warning rules | **frozen** — Phase 05 thresholds, unchanged (P-13) |
| `GAIRAEngine` API surface | **frozen** |

Any change to the above is a **new architecture version**, not a patch. Changing an artefact
changes its fingerprint, and the engine refuses to load — the freeze is enforced by the code, not
by convention.

## 4. What is explicitly *not* frozen

- **BSV2** (Phase 07) — a derived description downstream of Chemistry Evidence. It may be
  revised without touching the engine, because it is not on the inference path.
- **Figures, reports and thresholds used for presentation** — how the radar is drawn, how many
  candidates are displayed, where an operator sets an abstention point. The risk–coverage curve
  is published so operators can choose; no operating point is baked in.
- **The corpus.** Adding molecules is the intended path forward and does not require an
  architecture change — though it does change the atlas fingerprint and therefore constitutes a
  new atlas version.

---

## 5. Open risks carried forward

| id | risk | state |
|---|---|---|
| R-01 | class-prior bias — the 16 classes range from n = 3 to n = 80, and the chemistry layer inherits that prior | **OPEN**, mitigated by the idf size correction, not eliminated |
| R-10 | in-sample evaluation | **closed for chemistry** by G15; the in-sample figures are retained but explicitly labelled |
| R-12 | singleton molecules cap molecule retrieval at 0.819 | **OPEN**, structural; a corpus problem, not an engine problem |
| R-17 | the ontology is a curated cut through a continuum, not a natural kind (Phase 06.5 A1) | **OPEN by design**, stated in every report |
| R-18 | stability without informativeness is not evidence | **closed procedurally** — every selection rule in V7 now carries an informativeness floor |
| — | the engine cannot detect that the true molecule is absent from the bank | **OPEN**, stated in the spec §14; the `unknown` warning detects unexplained *spectra*, not unknown *molecules* |

## 6. Scope limits restated

Nothing in Phase 09 licenses a claim about SERS, serum, plasma, EV, tissue, pathogen or any other
applied regime. The corpus is pure Raman reference spectra of 154 canonical molecules. Transfer
to applied regimes is **unmeasured in V7** and must be established by a separate phase with its
own validation, its own gates, and its own honest negatives.

---

## 7. Recommendation

**Freeze V7 here.** The user's own recommendation — "I would make this the last architecture
phase" — is supported by the evidence rather than merely accepted. Four independent attempts to
add a layer above the CSM have now failed on measurement (themes 0.405, Meta Components 0.392,
grounded axes 0.664, geometric coordinates p = 0.180), and a fifth would be a re-run of the same
experiment with a different name.

The productive next work is not architectural:

1. **Corpus expansion.** 66 of 154 molecules have a single spectrum. This alone caps molecule
   top-1 at 0.819 and is the largest single lever on retrieval performance. It requires no
   engine change.
2. **Transfer.** Establish what survives the move from pure Raman to SERS, serum and EV. This is
   the question GAIRA ultimately exists to answer, and V7 has deliberately not touched it.
3. **Prospective validation.** Every number in V7 comes from the corpus that built it. A held-out
   external library, acquired independently, would test the architecture in the only way that
   remains.

None of these requires reopening a decision made in Phases 00–09.
