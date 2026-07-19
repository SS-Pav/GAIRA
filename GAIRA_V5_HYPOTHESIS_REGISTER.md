# GAIRA V5 — Hypothesis Register

**Purpose:** turn the V5 rebuild into a genuine scientific investigation. Every major architectural assumption is listed as a falsifiable hypothesis and updated as each phase produces evidence. Status vocabulary: **Supported · Partially supported · Rejected · Insufficient evidence · Untested**.

Governing rule: a rejected or insufficient hypothesis is a *successful* scientific result. Do not force a later phase to preserve a hypothesis.

---

## Hypotheses

| ID | Hypothesis | Phase(s) that test it | Status | Evidence / notes |
| --- | --- | --- | --- | --- |
| **H1** | Raman and Ag-SERS spectra can be transformed into a shared observation space. | 1, 1.5, 2 | **Now testable (insufficient evidence to date)** | Phase 1: weak cross-modality similarity + only 7 matched → untestable. Phase 1.5: corpus completed to **51 matched analytes** at 785 nm — H1 is now testable in Phase 2. Test, do not assume. |
| **H2** | Direct molecular spectra contain stable biochemical structure without needing learned embeddings. | 3 (representation), 4 (emergent structure) | **Untested** | — |
| **H3** | Emergent biochemical components are reproducible across data sources/substrates. | 4 | **Untested** | — |
| **H4** | A single shared ontology is preferable to modality-specific ontologies. | 4–5 | **Untested** | — |
| **H5** | Controlled perturbation datasets remain valid held-out evaluations (were not used to fit axes/weights). | 0, 8 | **Untested** | V4 established they were NOT used to fit V3 coordinates; V5 must keep them held out. |

### Supporting sub-hypotheses (added as evidence demands)
| ID | Hypothesis | Phase | Status | Notes |
| --- | --- | --- | --- | --- |
| **H1a** | Enough analytes measured in BOTH Raman and Ag-SERS to *estimate* a cross-mode transform. | 1–2, 1.5 | **Supported** | Phase 1.5 (785 nm + Gobbato integration): **51 matched analytes** (58.6% of the 87-analyte corpus), Raman ~3 reps + Ag-SERS ~5 reps each. Overlap is now sufficient to attempt Phase-2 representation discovery. |
| **H1b** | Matched analytes more similar across modalities than to other analytes. | 1 | **Partially supported** | matched cosine (0.25–0.53) > null (0.02–0.41), but absolute similarity is low and the gap modest; shared chemical info exists but is weak. |
| **H1c** | Shared preprocessing preserves analyte-specific bands. | 1 | **Supported (per-modality)** | common window 520–1750 covers ≥99.7%; adenine 725 SERS band retained; but SNV vs L2 vs area change similarity substantially → preprocessing is consequential. |
| **H6** | Unsupervised structure reflects chemistry, not acquisition modality/source/excitation. | 1,3–4 | **Partially rejected** | modality leakage CV acc 0.74–0.86 (> 0.745 majority baseline); the Raman corpus itself spans 9 excitation domains. Nuisance structure is present; must be controlled before ontology work. |

---

## Update log
- **2026-07-18 (Phase 0 start):** register created. H1a provisionally rejected (7 matched analytes). All others Untested.
- **2026-07-18 (Phase 1 complete):** H1 insufficient (7 matched); H1a rejected; H1b partial; H1c supported; H6 partially rejected. Decision: complete the grounding corpus first (Phase 1.5).
- **2026-07-18 (Phase 1.5 complete):** 785 nm corpus completed via Gobbato integration. **H1a now SUPPORTED (51 matched analytes, 58.6%)**; H1 now TESTABLE. Decision: Phase 2 (Canonical Representation Discovery) is scientifically justified. Do NOT assume a shared space exists — test it (Stage A direct spectra first).
