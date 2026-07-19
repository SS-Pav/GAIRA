# GAIRA V5 — Hypothesis Register

**Purpose:** turn the V5 rebuild into a genuine scientific investigation. Every major architectural assumption is listed as a falsifiable hypothesis and updated as each phase produces evidence. Status vocabulary: **Supported · Partially supported · Rejected · Insufficient evidence · Untested**.

Governing rule: a rejected or insufficient hypothesis is a *successful* scientific result. Do not force a later phase to preserve a hypothesis.

---

## V5 hypotheses (restructured 2026-07-19 for the observation-vs-evidence architecture)

> **Reframing:** Stage A rejected the narrow *"a shared **spectral** space exists"* hypothesis. The register is restructured around the broader question — *can a canonical **biochemical evidence** representation preserve shared chemistry across observation domains?* — and around the Stage B branches (interpretable vs encoder). Legacy Phase 0–2A hypotheses and their evidence are preserved in the **Legacy appendix** below (nothing is deleted). Some IDs are reused with new, broader meanings; the appendix maps old→new.

| ID | Hypothesis | Phase(s) | Status | Evidence / notes |
| --- | --- | --- | --- | --- |
| **H1** | A canonical biochemical **evidence** representation exists across observation domains (Raman, Ag-SERS, future Au-SERS/DART). | 2A, 2B, 2C | **Not supported by current corpus** | Stage A rejected the raw-spectral form. **Stage B:** no interpretable rep and no encoder materially beats modality-stratified direct spectra on held-out matched-analyte cross-modal retrieval (best held-out MRR 0.460 = I1 regions ≈ direct_SNV 0.452, CIs overlap). Open only pending more/diverse data (Stage C). |
| **H1a** | The current **51 matched analytes** are sufficient for a **feasibility benchmark** (not for a production encoder). | 1.5, 2B | **Supported (for feasibility)** | 51 matched / 87 analytes, Raman ~3 + Ag-SERS ~5 reps; enough for permutation-tested retrieval with analyte-bootstrap CIs. Sufficiency for *generalization* is H7. |
| **H1b** | Cross-modal matched analytes contain shared biochemical information **beyond random similarity**. | 1, 2A | **Partially supported** | Stage A: matched cosine > unmatched, permutation **p ≤ 0.02** (MRR p ≤ 0.014); but absolute signal weak (top-1 ≤ 0.16) and not carried by peak-position overlap. |
| **H1c** | An **encoder** can improve cross-modal retrieval relative to direct spectra. | 2B | **Rejected (this corpus)** | Every encoder is *below* direct held-out: E1 top-1 0.176 / E2 0.137 vs direct_SNV 0.275. Encoders reduce, not improve, cross-modal retrieval. |
| **H1d** | An encoder can **reduce modality leakage while preserving analyte identity** (both at once). | 2B | **Rejected (this corpus)** | Encoders raise modality leakage to 0.91–1.00 while collapsing (cross-analyte duplicate embedding 0.96–1.00); VICReg lowers leakage only by destroying retrieval (top-1 0.078, p=0.83). Never both at once. |
| **H1e** | Observation-specific encoders with a shared latent space (**dual encoder**) outperform a **single shared encoder**. | 2B | **Rejected (this corpus)** | E2 dual MRR 0.312 < E1 shared 0.338; separate encoders make modality trivially separable rather than aligning identity. |
| **H2** | **Interpretable** evidence representations can recover a canonical biochemical evidence space **without** neural encoders. | 2B | **Not supported** | Best interpretable (I1 regions) only *matches* direct (MRR 0.460 vs 0.452, CI overlap) — it does not recover a *new* shared space beyond direct spectra. |
| **H2a** | Adaptive regions or distributed motifs outperform **isolated peak** features. | 2B | **Weakly supported** | I1 adaptive regions is the best interpretable rep and beats I3 sparse-dictionary (MRR 0.460 vs 0.375); regions/multiscale > sparse-atom codes here. |
| **H3** | A **hybrid** (encoder embedding + sparse evidence projection) offers the best balance of performance and interpretability. | 2B | **Rejected (this corpus)** | E4 hybrid (E1+I1) MRR 0.438 < direct 0.452 and < I1 0.460; adds complexity without gain. |
| **H4** | Direct raw/preprocessed spectra should remain **modality-stratified**. | 2A | **Supported** | Stage A Outcome B: modality dominates the joint direct space (bal-acc 0.83–0.94); direct spectra useful within modality (Raman ARI 0.49–0.52), not across. |
| **H5** | Controlled perturbation datasets remain valid held-out evaluations (not used to fit representation). | 0, 2, later | **Maintained** | Stage A verified adenine concentration series excluded from fitting; Stage B reuses the frozen manifest and role rules. |
| **H6** | A successful canonical representation must encode **chemistry more strongly than modality or source**. | 1, 2A, 2B | **Partially rejected for direct spectra; open for Stage B candidates** | Direct: modality leakage bal-acc 0.83–0.94 ≫ chance. Stage B candidates must demonstrate chemistry ≫ nuisance to pass. |
| **H7** | The present grounding corpus is sufficient to train an encoder that **generalizes to held-out analytes**. | 2B, 2C | **Rejected (high-risk confirmed)** | Encoders collapse (cross-analyte dup 0.96–1.00) and underperform direct on held-out cross-modal (top-1 0.08–0.18), held-out-analyte family retrieval (0.48–0.50 vs direct 0.74), and within-modality ARI (0.53–0.77 vs 0.94). 51 matched / one instrument / single-source Ag-SERS is below encoder scale. |
| **H8** | Future **DART** perturbation sequences can be represented as **trajectories** in the selected biochemical representation. | future (post-2C) | **Untested (future only)** | Design intention only (see architecture context §6). Nothing in V5 to date demonstrates DART-compatible encoding. |

---

## Legacy appendix — Phase 0–2A hypotheses (preserved; superseded by the restructured table above)
These are the original entries and evidence, retained so no falsification history is lost. Old→new mapping in brackets.
| Legacy ID | Legacy hypothesis | Final legacy status | Maps to |
| --- | --- | --- | --- |
| H1 (legacy) | Raman and Ag-SERS spectra can be transformed into a shared **observation/spectral** space. | **Not supported from direct spectra** (Stage A: SNV modality bal-acc 0.83, top-1 0.16). | → new **H1** (broadened to *evidence* representation) + **H4** |
| H1a (legacy) | Enough analytes measured in both modalities to estimate a cross-mode transform. | **Supported** (51 matched, 58.6%). | → new **H1a** (scoped to *feasibility*) |
| H1b (legacy) | Matched analytes more similar across modalities than to others. | **Partially supported** (matched cosine 0.25–0.53 > null). | → new **H1b** |
| H1c (legacy) | Shared preprocessing preserves analyte-specific bands. | **Supported (per-modality)**; SNV vs L2 vs area matter. | folded into Stage A evidence; preprocessing-sensitivity carried into new **H6** |
| H1d (legacy) | Direct spectra preserve cross-modal analyte identity. | **Partially supported (weak)** (p ≤ 0.02, top-1 ≤ 0.16). | → new **H1b** / motivates new **H1c**,**H1d** |
| H2 (legacy) | Direct molecular spectra contain stable structure without learned embeddings. | Untested at the time. | → new **H2** (interpretable branch, sharpened) |
| H3 (legacy) | Emergent components reproducible across sources/substrates. | Untested. | deferred to Stage D |
| H4 (legacy) | A single shared **ontology** preferable to modality-specific ontologies. | Untested. | deferred to Stage D (ontology) |
| H4-preliminary | Raman/Ag-SERS require modality-stratified representations. | **Supported (direct stage).** | → new **H4** |
| H6 (legacy) | Unsupervised structure reflects chemistry, not modality/source/excitation. | **Partially rejected** (modality leakage bal-acc 0.74–0.94). | → new **H6** |
| H5 (legacy) | Perturbation datasets remain valid held-out evaluations. | **Maintained.** | → new **H5** |

---

## Update log
- **2026-07-18 (Phase 0 start):** register created. H1a provisionally rejected (7 matched analytes). All others Untested.
- **2026-07-18 (Phase 1 complete):** H1 insufficient (7 matched); H1a rejected; H1b partial; H1c supported; H6 partially rejected. Decision: complete the grounding corpus first (Phase 1.5).
- **2026-07-18 (Phase 1.5 complete):** 785 nm corpus completed via Gobbato integration. **H1a now SUPPORTED (51 matched analytes, 58.6%)**; H1 now TESTABLE. Decision: Phase 2 (Canonical Representation Discovery) is scientifically justified. Do NOT assume a shared space exists — test it (Stage A direct spectra first).
- **2026-07-18 (Phase 2 Stage A complete — direct representation):** Decision **Outcome B (modality-stratified defensible)**. Direct 785 nm spectra do not support a shared Raman/Ag-SERS coordinate system (best-preproc SNV: modality bal-acc 0.83, cross-modal top-1 0.16) but recover chemistry within modality (Raman ARI 0.49–0.52) with a weak significant residual cross-modal signal (p≤0.02). **H1 not supported from direct spectra; H1d partially supported (weak); H4-preliminary supported; H6 partially rejected (modality leakage); H5 maintained.** SNV best preserves joint chemistry-over-nuisance. Recommend Stage B chemical features next (not yet started). Report: `GAIRA_V5_PHASE2_STAGE_A_DIRECT_REPRESENTATION_REPORT.md`.
- **2026-07-18 (Phase 2 §4 input audit — role correction):** The 6 adenine bAgNPs Ag-SERS spectra were confirmed to be a controlled concentration series (perturbation evaluation), not independent grounding, and were **excluded from representation fitting** (data-role separation, H5). Adenine stays matched via Gobbato, so **H1a is unaffected (51 matched retained)**. Corrected representation corpus: **479 spectra (214 Raman + 265 Ag-SERS), 87 analytes**. Manifest: `results/v5_rebuild/phase2_stage_a/tables/phase2_input_manifest.csv`. **H5 evidence:** perturbation data verified held out of Stage-A fitting (partially supported → maintained).
- **2026-07-19 (register restructured for observation-vs-evidence architecture):** narrow spectral-space H1 rejected → broadened to *biochemical evidence* representation. Added Stage-B hypotheses **H1c, H1d, H1e, H2, H2a, H3, H7, H8**. Legacy entries preserved in the Legacy appendix.
- **2026-07-19 (Phase 2 Stage B complete — representation benchmark):** **Decision Outcome B4 (modality-stratified retained).** Under a leakage-safe framework (splits A/B/C/D; D infeasible for single-source Ag-SERS), no representation materially beats direct spectra on held-out matched-analyte cross-modal retrieval (best held-out MRR 0.460 = I1 regions ≈ direct_SNV 0.452, overlapping CIs). **Encoders underperform and collapse:** cross-modal top-1 0.08–0.18 (< direct 0.28), cross-analyte duplicate embedding 0.96–1.00, modality leakage 0.91–1.00, held-out-analyte family retrieval 0.48–0.50 (< direct 0.74), within-modality Raman ARI 0.53–0.77 (< direct 0.94). **Updates:** H1 not supported by current corpus; **H1c/H1d/H1e/H3 rejected**; **H2 not supported / H2a weakly supported**; **H4 reaffirmed (supported)**; **H6 unmet by any shared candidate**; **H7 rejected (high-risk confirmed)**. Frozen shared representation: **none**. Next: Stage C re-scoped as targeted data acquisition + interpretable refinement (NOT encoder scaling), then re-benchmark; Stage D (ontology) gated. Report: `GAIRA_V5_PHASE2_STAGE_B_REPRESENTATION_STRATEGY_REPORT.md`.
