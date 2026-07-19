# GAIRA V5 — Hypothesis Register

**Purpose:** turn the V5 rebuild into a genuine scientific investigation. Every major architectural assumption is listed as a falsifiable hypothesis and updated as each phase produces evidence. Status vocabulary: **Supported · Partially supported · Rejected · Insufficient evidence · Untested**.

Governing rule: a rejected or insufficient hypothesis is a *successful* scientific result. Do not force a later phase to preserve a hypothesis.

---

## Hypotheses

| ID | Hypothesis | Phase(s) that test it | Status | Evidence / notes |
| --- | --- | --- | --- | --- |
| **H1** | Raman and Ag-SERS spectra can be transformed into a shared observation space. | 1, 1.5, 2 | **Not supported from direct spectra (open for later regimes)** | Phase 1: untestable (7 matched). Phase 1.5: 51 matched → testable. **Phase 2 Stage A:** even under the best preprocessing (SNV), modality classifier bal-acc **0.83** (chance 0.50) and cross-modal top-1 only **0.16** → no defensible single shared coordinate system from *direct* spectra. Not rejected outright (weak p≤0.02 residual signal); re-test only if Stage B/observation-model regimes strengthen it. Do not assume. |
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
| **H1d** | Direct (unlearned) spectral representations preserve cross-modal analyte identity — the same analyte's Raman and Ag-SERS spectra are closer to each other than to other analytes, in the raw representation. | 2 (Stage A) | **Partially supported (weak)** | Matched cosine > unmatched with permutation **p ≤ 0.02** (MRR p ≤ 0.014) across L2/SNV, BUT absolute retrieval weak: top-1 ≤ 0.16, top-5 ≤ 0.31, reciprocal-NN ≤ 0.18. Peak positions do NOT align (matched ≈ mismatched band overlap). Shared chemical info exists but is too weak to identify analytes across modalities from raw spectra. |
| **H4-preliminary** | Raman and Ag-SERS require modality-stratified representations (aligned later at ontology level), rather than one shared direct-spectral space. | 2 (Stage A) | **Supported (direct-representation stage)** | Stage A Outcome B. Direct spectra recover chemistry *within* modality (Raman-only clustering ARI 0.49–0.52) but modality dominates the joint space (bal-acc 0.83–0.94); shared-space gates fail. Represent modalities separately; align at analyte/ontology level. |

---

## Update log
- **2026-07-18 (Phase 0 start):** register created. H1a provisionally rejected (7 matched analytes). All others Untested.
- **2026-07-18 (Phase 1 complete):** H1 insufficient (7 matched); H1a rejected; H1b partial; H1c supported; H6 partially rejected. Decision: complete the grounding corpus first (Phase 1.5).
- **2026-07-18 (Phase 1.5 complete):** 785 nm corpus completed via Gobbato integration. **H1a now SUPPORTED (51 matched analytes, 58.6%)**; H1 now TESTABLE. Decision: Phase 2 (Canonical Representation Discovery) is scientifically justified. Do NOT assume a shared space exists — test it (Stage A direct spectra first).
- **2026-07-18 (Phase 2 Stage A complete — direct representation):** Decision **Outcome B (modality-stratified defensible)**. Direct 785 nm spectra do not support a shared Raman/Ag-SERS coordinate system (best-preproc SNV: modality bal-acc 0.83, cross-modal top-1 0.16) but recover chemistry within modality (Raman ARI 0.49–0.52) with a weak significant residual cross-modal signal (p≤0.02). **H1 not supported from direct spectra; H1d partially supported (weak); H4-preliminary supported; H6 partially rejected (modality leakage); H5 maintained.** SNV best preserves joint chemistry-over-nuisance. Recommend Stage B chemical features next (not yet started). Report: `GAIRA_V5_PHASE2_STAGE_A_DIRECT_REPRESENTATION_REPORT.md`.
- **2026-07-18 (Phase 2 §4 input audit — role correction):** The 6 adenine bAgNPs Ag-SERS spectra were confirmed to be a controlled concentration series (perturbation evaluation), not independent grounding, and were **excluded from representation fitting** (data-role separation, H5). Adenine stays matched via Gobbato, so **H1a is unaffected (51 matched retained)**. Corrected representation corpus: **479 spectra (214 Raman + 265 Ag-SERS), 87 analytes**. Manifest: `results/v5_rebuild/phase2_stage_a/tables/phase2_input_manifest.csv`. **H5 evidence:** perturbation data verified held out of Stage-A fitting (partially supported → maintained). Added **H1d** below (direct spectra preserve cross-modal analyte identity).
