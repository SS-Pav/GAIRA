# GAIRA V5 — Architecture & Scientific Context

**Status:** V5 scientific-architecture context (living document). **This is not a replacement for the historical audit/architecture documents** — `GAIRA_FULL_CONTEXT_AND_STATE_AUDIT_2026-07-15.md` (forensic snapshot) and `GAIRA_CURRENT_STATE_AND_ARCHITECTURE_V4.md` (V4 canonical) remain the record for V1–V4 and the demo. This document governs the **V5 rebuild** (785 nm, first-principles, evidence-grounded) and is updated as each V5 phase produces evidence.

Governing docs: `GAIRA_V5_REBUILD_PLAN.md` · `GAIRA_V5_HYPOTHESIS_REGISTER.md` · phase reports under repo root and `results/v5_rebuild/`.

---

## 1. Scientific purpose

GAIRA is a **grounded biochemical reasoning engine** for Raman/SERS biological spectra. It does **not** treat spectra as perfect molecular fingerprints. Biofluid Raman/SERS spectra are mixtures; the literature routinely overclaims molecular assignments by matching nearby wavenumbers. GAIRA's design separates, as distinct layers with distinct evidence standards:

- **observation physics** — what a given instrument/substrate/excitation actually measures;
- **biochemical evidence** — modality-independent evidence about chemistry;
- **ontology** — the biochemical structure that evidence supports;
- **biological interpretation** — disease/perturbation reasoning, downstream and last.

The V5 rebuild reconstructs this stack from first principles at a single excitation (785 nm), treating excitation/substrate as controlled nuisance rather than free variables.

---

## 2. Current V5 evidence (as of Phase 2 Stage A)

- **785 nm restriction.** V5 uses 785 nm only; non-785 spectra are retained in provenance but excluded from representation fitting.
- **Completed grounding corpus (Phase 1.5).** 479 eligible direct grounding spectra: **214 Raman + 265 Ag-SERS**, **87 unique analytes**, **51 matched** Raman/Ag-SERS analytes. Ag-SERS is primarily single-source (Gobbato colloid); matched pairs are largely one instrument ecosystem.
- **Data-role separation.** Controlled adenine concentration-series spectra are **excluded from fitting** (perturbation evaluation, not grounding). Peak-only evidence (ORC-Ag) is retained for later MSS use but excluded from representation learning. Non-785 spectra excluded.
- **Stage A outcome (direct spectra).** Direct spectra contain real chemistry **within** an observation domain (Raman-only clustering ARI ≈ 0.49–0.52) but do **not** support a single shared Raman/Ag-SERS spectral coordinate system:
  - best direct joint preprocessing = **SNV**;
  - modality classifier balanced accuracy ≈ **0.83** under SNV (chance 0.50);
  - cross-modal exact-analyte **top-1 retrieval ≈ 0.16**;
  - matched-pair similarity is statistically above null (permutation **p ≤ 0.02**) but **weak**;
  - matched **peak-position** overlap is **not** materially above mismatched — SERS shifts and re-weights Raman bands.
- **Decision:** direct raw/preprocessed spectra should remain **modality-stratified**; a shared layer, if any, must represent **biochemical evidence**, not a shared raw spectral space. This motivated testing whether a *learned or interpretable evidence representation* can preserve shared chemistry across observation domains (Phase 2 Stage B).

## 2b. Stage B evidence (representation strategy benchmark, 2026-07-19) — **Outcome B4**

A leakage-safe benchmark (splits: held-out analytes / held-out matched pairs / replicate-group / source — the source split is infeasible for single-source Ag-SERS) compared **direct spectra**, **interpretable evidence representations** (adaptive regions, multiscale, sparse dictionary, NMF basis), and **small encoders** (shared, dual, modality-specific, + hybrid) on **held-out matched-analyte cross-modal retrieval** (chance top-1 ≈ 0.098):

- **No representation materially beats direct spectra.** Best held-out MRR 0.460 (interpretable adaptive regions) ≈ direct_SNV 0.452, with overlapping bootstrap CIs.
- **Encoders underperform and collapse:** cross-modal top-1 0.08–0.18 (< direct 0.28); cross-analyte duplicate embedding 0.96–1.00; modality leakage 0.91–1.00; held-out-analyte family retrieval 0.48–0.50 (< direct 0.74); within-modality Raman ARI 0.53–0.77 (< direct 0.94). A dual encoder did **not** beat a shared one.
- **Conclusion:** retain **modality-stratified** representations; **no shared biochemical representation is supported by the current corpus**, and a small encoder adds no value at this scale (an encoder feasibility *negative*, not a foundation model). Frozen shared representation: **none**.

### Validated vs provisional vs rejected (as of Stage B)
- **Currently validated:** modality-stratified direct spectra carry real within-modality chemistry (Raman held-out ARI 0.94); a weak-but-significant cross-modal signal exists but is not strong enough to define a shared space; interpretable adaptive regions are a viable *auditable* companion representation (≈ direct).
- **Provisional / future (hypotheses, not results):** a shared biochemical evidence representation may become supportable on a **larger, multi-source** corpus (Stage C data work); DART-as-trajectory (H8) is a design intention only.
- **Rejected (this corpus):** shared raw-spectral space (Stage A); encoder improves cross-modal retrieval (H1c); encoder reduces modality leakage while preserving identity (H1d); dual > shared encoder (H1e); hybrid best (H3); current corpus sufficient to train a generalizing encoder (H7).
- **Unresolved risk:** single-source Ag-SERS makes observation-domain invariance **untestable in-corpus** — no invariance claim is made; this is the primary Stage C data gap.

---

## 3. Revised V5 architecture

```
Observation domain            (Raman, Ag-SERS; future Au-SERS, DART)
        │
Observation-specific          (domain preprocessing + encoding / evidence extraction)
representation
        │
Canonical biochemical         (shared layer — biochemical EVIDENCE, not raw spectra)
evidence representation
        │
Emergent biochemical          (stable latent factors / clusters, discovered not imposed)
structure
        │
Biochemical ontology          (versioned; parents / children / continuous factors)
        │
Biochemical State Vector      (BSV)
        │
Molecular Spectral Signatures (MSS)
        │
Perturbation & biological     (DART, cohorts — last, and only on a defensible evidence layer)
interpretation
```

**The shared layer is not necessarily a shared raw spectral space.** It is intended to represent biochemical evidence that is comparable across observation domains. Whether such a representation exists, and whether it is best realized by interpretable evidence features or a learned encoder, is exactly what Phase 2 Stage B tests.

---

## 4. Conceptual revision (Stage A → Stage B)

The earlier narrow hypothesis — *"a shared spectral representation exists"* — is replaced by the broader hypothesis:

> **"A canonical biochemical evidence representation can preserve shared chemistry across observation domains."**

An **encoder embedding is a valid candidate** for this representation. But it must **not** be assumed to represent chemistry merely because it reconstructs spectra or produces attractive clusters. The representation is evaluated against explicit scientific constraints (Stage B evaluation framework): cross-modal biochemical retrieval on **held-out** analytes, nuisance (modality/source) suppression **without** loss of analyte identity, seed/split stability, absence of embedding collapse or source shortcuts, interpretability, and uncertainty.

---

## 5. Encoder philosophy

- Encoder embeddings are **candidate** biochemical representations, not an assumed foundation.
- **Reconstruction alone is insufficient** evidence of chemistry; it is at most an auxiliary objective.
- **Contrastive / metric constraints** (analyte identity within and across modality) are the scientifically relevant objectives.
- Chemistry must be **preserved**; modality/source nuisance must be **suppressed** — both at once. Reducing modality accuracy while erasing analyte identity is failure, not success.
- Interpretability must be **restored** through evidence projection (sparse linear probes / attribution), because an encoder is an *observation representation* and does not directly assign molecules.
- The current stage is an **encoder feasibility benchmark**, not foundation-model completion. The corpus (479 spectra, 87 analytes, 51 matched, largely single-source Ag-SERS) is small; no large network, transformer, or pretraining is used, and training-set performance is not interpreted.
- **Stage B outcome:** at this corpus scale the encoder feasibility question is answered **negatively** — small encoders collapse and add no value over direct spectra (see §2b). An encoder is *not* the GAIRA V5 representation layer; it is not revisited until a larger, multi-source corpus first shows a shared signal (Stage C).

---

## 6. DART compatibility (future, not yet achieved)

Future **DART** (dynamic electrochemical perturbation) observations should eventually map to **trajectories through biochemical representation space**, rather than being forced to resemble static Raman or Ag-SERS spectra. This is a design intention and a hypothesis (register H8) — it has **not** been demonstrated. Nothing in V5 to date establishes DART-compatible encoding; Stage B only asks whether the selected representation is *structurally compatible* with a future trajectory formulation.

---

## 7. What V5 has NOT done (scope guardrails)

Not started / out of scope until authorized: ontology construction, BSV, MSS, perturbation analysis, biological-cohort analysis, DART integration, production integration, model scaling, transformer pretraining, external data acquisition, demo modification. Historical V1/V2/V3/V3.1 pipelines and the demo are **untouched**.

---

## 8. Document map

| Concern | Canonical doc |
| --- | --- |
| V1–V4 architecture, demo, production `src/gaira` stack | `GAIRA_CURRENT_STATE_AND_ARCHITECTURE_V4.md`, `GAIRA_FULL_CONTEXT_AND_STATE_AUDIT_2026-07-15.md` |
| V5 scientific architecture & context (this doc) | `GAIRA_V5_ARCHITECTURE_AND_SCIENTIFIC_CONTEXT.md` |
| V5 phase-by-phase plan | `GAIRA_V5_REBUILD_PLAN.md` |
| V5 falsifiable hypotheses | `GAIRA_V5_HYPOTHESIS_REGISTER.md` |
| V5 phase reports | `GAIRA_V5_PHASE1_*.md`, `GAIRA_V5_PHASE2_STAGE_*.md`, `results/v5_rebuild/` |

*Updated after Phase 2 Stage B with validated vs provisional architecture (see §2 update and the Stage B report).*
