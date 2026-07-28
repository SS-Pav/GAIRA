# FOUNDATION_CORPUS_REPORT
### The canonical Raman-only corpus that defines GAIRA's biochemical coordinate system

*Part 2 of the GAIRA Foundation Model audit. Every number here is recomputed
deterministically from the same loader the frozen atlas was built with
(`gaira.foundation.dataset.load_reference_corpus`), reproduced by
`foundation_audit/code/corpus_analysis.py`. Chemical classes are assigned from KNOWN
chemistry (name/class rules in `gaira.foundation.families_raman`), never inferred from
spectra.*

---

## 1. Headline

| Property | Value |
|---|---|
| Modality | **Raman only** (Ag-SERS, Au-SERS, DART explicitly excluded) |
| Spectra | **375** |
| Unique analyte labels | **167** (≈ **161** distinct molecules after de-duplication — see §6) |
| Spectral window | **450–1800 cm⁻¹** |
| Grid | 2 cm⁻¹ → **676 bins** |
| Sources | 3 (RamanBioLib, Gobbato Raman, amino-acid grounding) |
| Excitations | 9 wavelengths (785 nm dominant) |
| Spectra / analyte | median **2**, mean 2.25, max 6; **80** singletons, **87** with replicates |

The corpus is a **pure-compound reference library**: each spectrum is a single
biochemical species measured in isolation. This is what makes a parts-based
decomposition meaningful — the basis is learned from spectra that ARE the parts, then
biological mixtures are projected as non-negative combinations of them.

---

## 2. Sources

| Source | Spectra | Unique analytes | Excitations | Origin / citation | Role |
|---|---:|---:|---|---|---|
| **RamanBioLib** | 202 | 141 | 785, 1064, 532, 488, 514.5, 632.8, 457.9, 850, 633 nm | Digitized public reference library of biomolecule Raman spectra (broad chemistry: lipids, fatty acids, triglycerides, sugars, proteins, nucleobases, cofactors) | TRAINING |
| **Gobbato Raman metabolites** | 153 | 51 | 785 nm (B&WTek i-Raman Plus powders) | Gobbato et al. 2025, *Anal. Bioanal. Chem.*, DOI 10.1007/s00216-025-06192-5 (PMC12680727) — the pure-powder Raman arm of the serum-SERS study | TRAINING |
| **Amino-acid grounding** | 20 | 19 | 785 nm | Curated pure amino-acid + key-metabolite Raman sheet (`amino_acid_raman_grounding/aa.xlsx`) | TRAINING |
| **TOTAL (deduped union)** | **375** | **167** | — | — | — |

See figure `figures/per_source.png`.

**Why three sources.** RamanBioLib supplies breadth of chemistry but is light on the
clinical serum metabolite panel; Gobbato supplies exactly that panel (21 metabolites
RamanBioLib lacks — glucose, urate, hypoxanthine, xanthine, creatinine, lactate, urea,
sugars, ergothioneine…); the amino-acid sheet reinforces the 20 canonical amino acids,
the most reused biochemical sub-vocabulary. 34 analyte labels appear in ≥2 sources —
genuine replication that strengthens the within-analyte stability signal
(`tables/corpus_cross_source_duplicates.csv`).

---

## 3. Chemical-class distribution

Assigned by rule from molecular identity. Figure: `figures/class_balance_analytes.png`.

| Reporting class | Analytes | Spectra |
|---|---:|---:|
| Protein | 32 | 81 |
| Saccharide (mono/oligo) | 27 | 43 |
| Amino acid | 17 | 73 |
| Lipid — triglyceride | 15 | 17 |
| Organic acid | 15 | 35 |
| Lipid — fatty acid | 12 | 14 |
| Sterol / steroid | 9 | 9 |
| Cofactor / vitamin | 6 | 18 |
| Purine | 5 | 17 |
| Polysaccharide | 5 | 10 |
| Lipid — other (chol/oleate/PI…) | 5 | 22 |
| Nucleic acid (DNA/RNA) | 3 | 3 |
| Pyrimidine | 3 | 9 |
| Lipid — phospholipid | 2 | 3 |
| Carotenoid | 2 | 2 |
| small nitrogenous (creatinine, urea) | 2 | 6 |
| polyol (glycerol) | 1 | 4 |
| Other / unclassified | 6 | 9 |

**Reading it.** By analytes the corpus is protein- and lipid-rich (RamanBioLib's
character); by *spectra* the balance shifts toward **amino acids (73)** and
**protein (81)** because those are the most-replicated species (grounding sheet +
Gobbato + RamanBioLib all carry them). Lipids as a super-class (triglyceride + fatty
acid + phospholipid + sterol + other = **43 analytes**) are the single largest chemical
territory — important context for Part 6, where several NMF components turn out to be
acyl/triglyceride motifs.

---

## 4. Structure

- **Spectra per analyte** (`figures/spectra_per_analyte.png`): median 2, max 6.
  **80 / 167 analytes are singletons** (one spectrum) — the corpus is broad but shallow.
  This is the central statistical limitation: within-analyte replicate structure exists
  for only 87 analytes, so component-stability and replicate-robustness metrics (Parts
  4, 6) are carried by roughly half the corpus.
- **Excitation** (`figures/excitation_distribution.png`): 785 nm dominates (234 / 375);
  1064 (55), 532 (50), 488 (29) follow, with a long tail of singletons. Excitation is
  tracked as a **nuisance factor** and its leakage into the latent space is explicitly
  measured (excitation leakage 0.019 for the frozen NMF — near zero; Part 4).
- **Multi-excitation analytes**: 41 analytes appear at >1 excitation, enabling the
  cross-excitation transfer test in Part 9.

---

## 5. Coverage gaps

Honest map of what the coordinate system is and is not grounded to observe:

- **Nucleic acids are thin.** 3 DNA/RNA polymers + 5 purines + 3 pyrimidines. The
  purine theme is nonetheless GAIRA's best-validated axis (adenine/urate/hypoxanthine
  perturbation drives it), but DNA/RNA polymer coverage is minimal.
- **No porphyrins / heme.** The ontology carries a *provisional* `heme_porphyrin` theme,
  but the corpus contains essentially no isolated porphyrin references (cytochrome c and
  hemoglobin appear as whole proteins, not as heme cofactor). Any heme interpretation is
  therefore under-grounded — flagged in Parts 7–8.
- **Flavins / vitamins are folded into "cofactor" (6).** Riboflavin/FAD-type flavin
  chemistry is present but sparse; there is no dedicated flavin axis.
- **Phospholipids (2) and free nucleotides are sparse**, despite being central to EV /
  membrane biology — a real limit for the EV validation cohorts.
- **Sterols are mostly esters/hormones (9)**, not free cholesterol families, which
  couples the sterol and acyl-lipid themes (documented collision in Parts 6–7).

These gaps are **representation-grounding limits**, not bugs: the model can only ground
a biochemical theme it has seen pure examples of. They define where projected biological
spectra should be read with the most caution.

---

## 6. Data-quality findings (honest)

The audit found **6 unmerged duplicate molecule groups** — the same molecule counted as
two analyte labels because canonicalization did not merge an abbreviation, a Unicode
ligature, or a salt/acid spelling:

| Label A | Label B | Cause |
|---|---|---|
| `alb` (1 sp) | `albumin` (6 sp) | abbreviation from the amino-acid sheet not expanded |
| `gluth` (1) | `glutathione` (4) | abbreviation not expanded |
| `ure` (1) | `urea` (3) | abbreviation not expanded |
| `riboﬂavin` (1) | `riboflavin` (3) | U+FB02 "ﬂ" ligature vs ASCII "fl" |
| `aspartic acid` (1) | `aspartate` (1) | acid vs conjugate-base spelling |
| `acetyl coenzyme a` (1) | `acetyl-coa` (3) | spacing/hyphen variant |

**Impact.** The reported **167** analyte labels correspond to ≈ **161** distinct
molecules (over-count ≈ 3.6 %). Consequences:
1. Minor **leakage risk** in the analyte-grouped benchmark: a molecule split across two
   group labels could place near-identical spectra in both a train and a test fold. The
   effect is small (each duplicate is 1–2 spectra) and does not change the k-selection
   (Part 4), but it very slightly inflates neighbourhood/replicate scores.
2. It splits a few analytes' replicate mass (e.g. albumin's 6th spectrum is orphaned as
   `alb`), marginally weakening their MSS/BSV stability.

**Recommendation (for a FUTURE atlas rebuild only — the current atlas is frozen):**
extend `AA_NAME_FIX` / `canonical()` to expand `alb→albumin`, `gluth→glutathione`,
`ure→urea`, normalize the "ﬂ/fi" ligatures, and unify acid/conjugate-base spellings.
This would drop the label count to ≈161 and remove the residual leakage. It must NOT be
applied silently: it changes the corpus and therefore the fingerprint.

A handful of rule misclassifications were also noted (e.g. `acetyl coenzyme a` bucketed
as *protein* rather than *cofactor*, `melanin` as *other*). These affect only the
post-hoc class labels in §3, never the representation.

---

## 7. Reproduction

```
python results/v5_rebuild/foundation_audit/code/corpus_analysis.py
```
Deterministic. Outputs: `tables/corpus_summary.json`, `tables/corpus_analytes.csv`,
`tables/corpus_cross_source_duplicates.csv`, and the four figures referenced above.

**Verdict.** The corpus is a scientifically appropriate, pure-Raman reference set: broad
across protein/lipid/saccharide chemistry, with a deliberate clinical-metabolite core
and a strong amino-acid backbone. Its principal limitations are shallowness
(48 % singletons), thin nucleic-acid/porphyrin coverage, and a small, fixable
canonicalization debt (≈6 duplicate labels). None of these compromise the frozen
representation; all of them bound where its projections should be trusted.
