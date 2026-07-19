# GAIRA V5 — Hypothesis Register

**Purpose:** turn the V5 rebuild into a genuine scientific investigation. Every major architectural assumption is listed as a falsifiable hypothesis and updated as each phase produces evidence. Status vocabulary: **Supported · Partially supported · Rejected · Insufficient evidence · Untested**.

Governing rule: a rejected or insufficient hypothesis is a *successful* scientific result. Do not force a later phase to preserve a hypothesis.

---

## Hypotheses

| ID | Hypothesis | Phase(s) that test it | Status | Evidence / notes |
| --- | --- | --- | --- | --- |
| **H1** | Raman and Ag-SERS spectra can be transformed into a shared observation space (jointly analyzable without erasing chemistry). | 1 (comparability), 2 (observation layer) | **Insufficient evidence — leaning against direct sharing** | Phase 1: matched-analyte cross-modality cosine only 0.25–0.53; modality leaks into PCA (CV acc 0.74–0.86 vs 0.745 majority); only 7 matched analytes — too few to fit a transform. Direct joint analysis not yet defensible; needs Phase-2 observation model AND more matched analytes. |
| **H2** | Direct molecular spectra contain stable biochemical structure without needing learned embeddings. | 3 (representation), 4 (emergent structure) | **Untested** | — |
| **H3** | Emergent biochemical components are reproducible across data sources/substrates. | 4 | **Untested** | — |
| **H4** | A single shared ontology is preferable to modality-specific ontologies. | 4–5 | **Untested** | — |
| **H5** | Controlled perturbation datasets remain valid held-out evaluations (were not used to fit axes/weights). | 0, 8 | **Untested** | V4 established they were NOT used to fit V3 coordinates; V5 must keep them held out. |

### Supporting sub-hypotheses (added as evidence demands)
| ID | Hypothesis | Phase | Status | Notes |
| --- | --- | --- | --- | --- |
| **H1a** | Enough analytes are measured in BOTH Raman and Ag-SERS to *estimate* a cross-mode transform. | 1–2 | **Provisionally rejected** | Only 6 name-matched analytes (RamanBioLib∩metabolite-63) + adenine ≈ 7. Sufficient to *compare*, insufficient to *fit* a transform (Phase-2 decision D risk). |
| **H1b** | Matched analytes more similar across modalities than to other analytes. | 1 | **Partially supported** | matched cosine (0.25–0.53) > null (0.02–0.41), but absolute similarity is low and the gap modest; shared chemical info exists but is weak. |
| **H1c** | Shared preprocessing preserves analyte-specific bands. | 1 | **Supported (per-modality)** | common window 520–1750 covers ≥99.7%; adenine 725 SERS band retained; but SNV vs L2 vs area change similarity substantially → preprocessing is consequential. |
| **H6** | Unsupervised structure reflects chemistry, not acquisition modality/source/excitation. | 1,3–4 | **Partially rejected** | modality leakage CV acc 0.74–0.86 (> 0.745 majority baseline); the Raman corpus itself spans 9 excitation domains. Nuisance structure is present; must be controlled before ontology work. |

---

## Update log
- **2026-07-18 (Phase 0 start):** register created. H1a provisionally rejected (7 matched analytes). All others Untested.
- **2026-07-18 (Phase 1 complete):** H1 insufficient/leaning-against direct sharing; H1a rejected (7 matched, too few to fit); H1b partially supported; H1c supported per-modality; H6 partially rejected (modality leakage present). Decision: modality-stratified analysis; do NOT build a shared observation space until more matched analytes (Gobbato pure Raman+Ag-SERS of 53 metabolites) are loaded.
