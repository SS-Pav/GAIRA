# V5 Phase 1.5 — Canonical Grounding Corpus Completion (785 nm)

**Question.** After completing the direct grounding corpus (restricting to 785 nm and integrating the Gobbato pure Raman + pure Ag-SERS metabolite corpus), how many analytes are measured in BOTH modalities, and is that enough to begin representation discovery?

**Methods.** Extended `src/gaira/data` with a Gobbato loader (B&WTek 785 nm; parses Raman-Shift + Dark-Subtracted columns) and a synonym/abbreviation reconciler (`synonyms.canonical`). Assembled the 785-only corpus: RamanBioLib 785 subset + adenine Ag-SERS + Gobbato pure Raman (153) + pure Ag-SERS (265). Excluded non-785 (RamanBioLib 532/1064/488/…), metabolite-63 (633 nm), and ORC-Ag (peak-only, kept for MSS). Built `canonical_analyte_registry_v5` and quantified overlap. **No PCA/clustering/NMF/embeddings/ontology/observation-model/BSV/MSS.**

**Results.**
- **87 unique analytes** at 785 nm.
- **214 Raman spectra** (RamanBioLib 785 ≈61 + Gobbato Raman 153) across **85 analytes**.
- **271 Ag-SERS spectra** (adenine 6 + Gobbato Ag-SERS 265) across **53 analytes**.
- **51 matched analytes** in BOTH 785 Raman AND 785 Ag-SERS (**58.6%** of the corpus).
- **485 spectra enter representation**; **228 excluded** — non-785 RamanBioLib (141), metabolite-63 633 nm (63), ORC-Ag peak-only (24; kept for MSS).
- Matched analytes span amino acids (Phe, Trp, Tyr, His, Arg, Leu, Ile, Val, Ser, Pro, Gly, Ala, Met…), purines/pyrimidines (adenine, guanine, xanthine, hypoxanthine, urate, uracil, thymine), carbohydrates (glucose, fructose, galactose, mannose, glycogen), lipids (oleate, stearate, triolein, cholesterol, phosphatidylinositol), redox/metabolites (glutathione, ergothioneine, ascorbate, lactate, pyruvate, citrate, creatinine), and cofactors (acetyl-CoA, CoA, riboflavin).

**Interpretation.** Two decisions fixed the Phase-1 stall: (1) the **785-only** simplification removes excitation as a nuisance (Raman had spanned 9 excitations); (2) **Gobbato integration** supplies a large internally-matched Raman↔Ag-SERS set (same instrument, same 785 nm, same 51 analytes measured in both powder-Raman and colloid-SERS). Matched overlap rose **7 → 51** (an order of magnitude). This is a chemically broad, replicated, matched reference set — the prerequisite the Phase-1 gate demanded.

**Limitations.** Matched pairs are strongest **within Gobbato** (one instrument); RamanBioLib (digitized literature Raman) adds Raman breadth but different provenance. The Ag-SERS side has only **two sources** (Gobbato colloid + adenine bAgNPs) and **still zero Au-SERS** grounding. Gobbato spectra retain the raw B&WTek axis (−310…3271); preprocessing must crop to the common window. Replicates: Raman ~3/analyte, Ag-SERS ~5/analyte (adequate but modest).

**Decision.** Corpus completion **succeeded**. **H1a (enough matched analytes to attempt a cross-mode analysis) is SUPPORTED**; **H1 (a shared representation exists) is now TESTABLE** (was untestable at 7). **Phase 2 — Canonical Representation Discovery — is scientifically justified.** Begin with **Stage A (direct spectra)** and test H1 rather than assume it; keep modality-stratified analysis as the fallback if modality still dominates structure.

**Next action.** Proceed to Phase 2 (Canonical Representation Discovery), Stage A, on the completed 785 nm corpus — **only when instructed** (this pass stops at the Phase 1.5 gate).
