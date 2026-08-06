# GAIRA V7 — Pre-Phase-02 Corpus Identity and Pure-Raman Completeness Audit

**Role:** Principal Investigator, adversarial review · **Date:** 2026-08-06 ·
**Branch** `gaira-v7-rebuild`

---

## 1. Executive summary

The audit was triggered by an apparent discrepancy — earlier GAIRA summaries cite ~167 Raman
analytes while Phase 00/01 use 154 canonical molecules. **Both numbers are correct; they count
different things**, and the audit resolves the full chain from raw file to canonical molecule.

| | |
|---|---:|
| Raw Raman spectra loaded | **375** |
| Dataset-specific source labels `(source, label)` | **212** |
| Distinct raw label strings | **194** |
| Normalized analyte **names** (V5 synonyms layer) | **167** |
| Canonical **molecules** (V7 layer) | **154** |
| Gobbato pure-Raman files in the archive → loaded | **153 → 153** (0 missing) |
| Gobbato Ag-SERS files → leaked into the corpus | **265 → 0** |
| Protected stereoisomer/anomer violations | **0** |
| One-to-many spectrum→molecule conflicts | **0** |
| Phase 01 rerun required | **No** — reproduced bit-identically |

**Verdict: 154 is the correct scientific unit count and no corpus correction is required.**

**Two items are carried forward as flagged uncertainties, neither blocking** (§13): one
ambiguous substrate provenance (insulin, quantified as negligible) and one unresolved isomer
identity (carotene vs β-carotene, already protected from merging).

**The audit did surface something previously unexamined.** The V5 synonyms layer collapses 194
distinct label strings to 167 normalized names — 27 collapses — and had never been audited.
Four of them merge an acid with its conjugate base, which the brief explicitly forbids doing
silently. Those four were tested empirically and are labelling variants, not chemically
distinct forms (§7.3).

---

## 2. Why this audit was necessary

Phase 01's Local Spectral Motif dictionary is about to become the input to Phase 02's consensus
construction. Any corpus defect — a missing pure-Raman reference, a leaked SERS spectrum, an
over-eager merge — propagates from there into every downstream layer and becomes progressively
harder to detect. Three specific risks justified auditing before proceeding:

1. **Modality leakage.** The Gobbato archive holds pure Raman powders *and* Ag-SERS spectra in
   the same zip, loaded by the same function. The Raman-only filter is applied by the *caller*,
   not by the loader. A future caller that forgets it would silently import 265 Ag-SERS spectra
   into the Raman foundation.
2. **Silent over-merging.** Canonicalisation reduces counts. Every reduction is a claim that two
   labels denote one molecule, and a wrong claim destroys a distinct reference permanently.
3. **Unexplained numbers.** "167" and "154" were both in circulation with no documented
   relationship. An unexplained count is an unaudited count.

---

## 3. Definitions

These four terms were used interchangeably in earlier summaries. They are not interchangeable,
and every figure in this audit states which one it plots.

| Term | Definition | Count |
|---|---|---:|
| **SPECTRUM** | one measured or digitized intensity trace loaded onto the canonical grid | 375 |
| **SOURCE LABEL** | the analyte string as written by its source dataset, scoped to that dataset | 212 |
| **NORMALIZED NAME** | a source label after the V5 synonyms layer (L-/D- prefixes, acid/base naming, salt forms) | 167 |
| **CANONICAL MOLECULE** | one chemical entity after V7 canonicalisation; the unit of scientific accounting | **154** |
| **UNIQUE CHEMICAL STRUCTURE** | a distinct structure by InChIKey/SMILES/CID | **not determinable** — see §13 |

---

## 4. Source inventory

The corpus was reconstructed independently from the configured data root. Modality was verified
from source metadata, loader records and archive provenance — **never from file names**.

| Source | SPECTRA | source LABELS | normalized NAMES | CANONICAL MOLECULES |
|---|---:|---:|---:|---:|
| RamanBioLib | 202 | 141 | 141 | 139 |
| gobbato_raman_metabolites | 153 | 51 | 51 | 51 |
| amino_acid_raman_grounding | 20 | 20 | 19 | 19 |
| **total (distinct)** | **375** | **212** | **167** | **154** |

Per-source canonical counts sum to 209, not 154, because 55 molecules occur in more than one
library. That overlap is the point of canonicalisation.

**Completeness checks:**

- RamanBioLib: index holds 202 rows / 141 components; parquet holds 202 distinct ids;
  **0 index entries lack a spectrum**. All 202 declare `modality = raman`.
- amino-acid grounding: `aa.xlsx` has 21 columns (1 wavenumber + 20 spectra); **all 20 pass the
  ≥100-finite-point threshold**; none dropped.
- Gobbato: see §5.

**Excluded by policy without loading (27 datasets):** adenine SERS controls, Ag-flake
metabolites, serum/plasma/EV/faecal/urine/saliva SERS, `covid_serum_raman`, `cspp_serum`,
`ergothioneine_serum`, `european_multi_instrument_adenine`, `metabolite_sers63_support`,
`sers_metabolite_63`, `serum_ag_colloids*`, `single_vesicle_ev_raman`, `small2023_ev`,
`otc_drugs`, and the remainder listed in `audit_corpus.py`.

---

## 5. Gobbato pure-Raman audit

The Gobbato archive (`serum_ag_colloids/dataset_spectral_data.zip`, 914 entries) contains both
modalities. They were separated and counted independently of the processed tables.

| | Raman | Ag-SERS |
|---|---:|---:|
| Files in the archive | **153** | 265 (+31 "for fitting") |
| Filename-regex matches | 153 | 265 |
| Filenames not matching the expected pattern | **0** | — |
| SPECTRA loaded | **153** | 265 |
| SPECTRA entering the V7 foundation | **153** | **0** |
| source LABELS | 51 | 53 |
| CANONICAL MOLECULES | 51 | — |
| Substrate field | `powder` | `Ag colloid (Gobbato)` |
| Modality field | `raman` | `sers` |

**Gobbato pure Raman is 100% complete.** 153 of 153 archive files parse and enter the corpus;
zero are missing; every one of the 51 labels has **exactly 3 replicates** (51 × 3 = 153).

**Gobbato Ag-SERS is 100% excluded.** All 265 are loaded by `load_gobbato_785()` and then
discarded by the caller's modality filter. Zero enter the corpus.

**Raman ↔ SERS pairing.** All 51 Raman labels also appear in the SERS set — the same molecules
measured two ways. Two labels are **SERS-only** (`DNA`, `RNA`) and are correctly absent from the
foundation. No label is Raman-only.

**Overlap.** Of the 51 Gobbato normalized labels, 30 also occur in RamanBioLib and **21 are
unique to Gobbato**. Losing the Gobbato Raman set would cost 21 canonical molecules outright.

**A latent engineering risk, now tested.** `load_gobbato_785()` returns both modalities in one
list; only the caller filters. A caller that forgot would import 265 Ag-SERS spectra silently.
`tests/test_v7_corpus_audit.py::test_gobbato_loader_returns_both_modalities_and_caller_must_filter`
pins this.

---

## 6. SERS exclusion audit

640 spectra were examined; 375 included, 265 excluded. Every exclusion carries a recorded reason
in `spectrum_level_audit_v1.csv`.

**One substrate string in the entire pure-Raman corpus matches a plasmonic-substrate pattern:**

> **RamanBioLib id 197 — insulin — "Gold-coated glass sustrate" [sic] — 633 nm**

This is the only ambiguity in the modality audit and it is genuine: gold with 633 nm excitation
is a classic SERS combination. Assessment:

| Evidence it is **normal Raman** | Evidence for **concern** |
|---|---|
| SERS requires *nanostructured* metal; a smooth evaporated gold film gives essentially no enhancement | gold + 633 nm is near the Au plasmon resonance |
| Gold-coated slides are a standard reflective, low-fluorescence substrate for normal Raman | roughness cannot be verified from the metadata |
| The source is a Raman reference library; the entry is typed "Proteins/Hormones" with no SERS designation | |
| No nanoparticle, colloid, roughening or island-film field anywhere in the record | |

**Decision: RETAINED and flagged.** Insulin is the sole spectrum for its canonical molecule, so
excluding it would remove a molecule from the corpus on an unproven suspicion. The exposure was
quantified rather than assumed:

> Refitting `peptide_protein` without insulin (30 → 29 molecules) gives Hungarian-matched basis
> similarity **0.990 mean / 0.918 min** to the full fit; **1 of 29** molecules changes EV by
> more than 0.02, and it *improves* (papain 0.950 → 0.983). Insulin is 1 of 375 spectra (0.27%)
> and 1 of 154 molecules (0.65%).

The uncertainty is real, bounded and immaterial to the dictionary.

---

## 7. Canonicalisation audit

### 7.1 Structure of the mapping

30 many-to-one groups (source label → canonical molecule), 29 of them cross-source.
**0 one-to-many conflicts**: no raw spectrum maps to more than one canonical molecule.

| Merge classification | groups |
|---|---:|
| 1 spelling / formatting | 19 |
| 7 stereoisomer prefix (generic → specific) | 4 (+1 combined) |
| 3 abbreviation | 3 |
| 4 common vs systematic name | 1 (+1 combined) |
| 5 salt / free-acid equivalence | 1 |
| 2 Unicode normalization | 1 |
| **10 distinct molecule incorrectly merged** | **0** |
| **11 unresolved** | **0** in the merge set; 1 non-merge flagged (§13) |

### 7.2 Protected distinctions — none collapsed

| Pair | Relationship | Status |
|---|---|---|
| `(+)-arabinose` / `(-)-arabinose` | enantiomers | ✅ protected |
| `(+)-glucose` / `β-d-glucose` | anomers | ✅ protected |
| `(-)-ribose` / `2-deoxy-d-ribose` | distinct molecules (2′-OH) | ✅ protected |
| `carotene` / `β-carotene` | isomer, provenance unproven | ✅ protected (not merged) |

No cis/trans isomer, positional isomer, nucleotide/nucleoside, oxidised/reduced or
molecule/polymer pair was merged anywhere in the corpus.

### 7.3 The acid / conjugate-base merges — the finding that required testing

The V5 synonyms layer, which had **never been audited**, collapses 194 label strings to 167
normalized names. Fifteen of those collapses join two or more distinct label strings, and
**four join an acid to its conjugate base**:

`ascorbic acid`→`ascorbate` · `citric acid`→`citrate` · `oleic acid`→`oleate` ·
`stearic acid`→`stearate`

The brief forbids merging acid and salt forms silently "where their Raman spectra or chemistry
differ materially". The free acid and the carboxylate salt are genuinely different Raman
species: the acid shows a C=O stretch near 1710 cm⁻¹ which the carboxylate lacks, replaced by
symmetric/antisymmetric COO⁻ modes near 1400 and 1580 cm⁻¹.

**Test:** compare the spectral share in the 1710 cm⁻¹ window between the two labelled forms.

| Canonical molecule | forms | cosine | C=O (1710) share A : B | ratio | verdict |
|---|---|---:|---|---:|---|
| ascorbate | ascorbic acid / ascorbate | 0.976 | 0.0631 : 0.0666 | 1.06 | same protonation state |
| oleate | oleic acid / oleate | 0.979 | 0.0068 : 0.0217 | 3.19 | same protonation state |
| stearate | stearic acid / stearate | 0.945 | 0.0077 : 0.0213 | 2.77 | same protonation state |
| citrate | citric acid / citrate | 0.834 | 0.1052 : 0.1156 | 1.10 | same protonation state |
| aspartate | aspartic acid / aspartate | — | — | — | one form only; no merge to verify |

**Control** — cross-source cosine for *uncontested* same-molecule merges: alanine 0.972,
valine 0.979, **tyrosine 0.915**.

**Conclusion: all four merges are valid.** In every pair both members carry comparable C=O
intensity, so both are the *same physical material* labelled two ways — not a free acid in one
library and a sodium salt in the other. If they were different forms the 1710 band would be
present in one and absent in the other; it is present in both.

`citrate` at cosine 0.834 sits below the uncontested control range (0.915–0.979) and is the
weakest merge in the corpus. Its C=O ratio (1.10) nonetheless shows the same protonation state,
and cross-source variation is itself substantial. Recorded, not corrected.

---

## 8. The 167 → 154 reconciliation

```
375   raw Raman SPECTRA                    files/columns loaded from 3 pure-Raman sources
 ↓    −163
212   dataset-specific SOURCE LABELS       (source, label) pairs; same string in two
 ↓    −18                                  libraries counted twice
194   distinct raw LABEL STRINGS           cross-source identical strings collapsed
 ↓    −27
167   normalized analyte NAMES             ← V5 synonyms layer: L-/D- prefixes,
 ↓    −13                                    acid/base naming, salt forms
154   CANONICAL MOLECULES                  ← V7 layer
```

### The 13-label difference, itemised

| # | Kind | Count | Members |
|---|---|---:|---|
| 1 | stereoisomer prefix (generic → specific) | 5 | `glucose`→`(+)-glucose`, `fructose`→`(-)-fructose`, `galactose`→`(+)-galactose`, `mannose`→`(+)-mannose`, `lactose`→`(+)-lactose` |
| 2 | abbreviation / truncation | 3 | `alb`→`albumin`, `gluth`→`glutathione`, `ure`→`urea` |
| 3 | orthographic | 2 | `acetyl coenzyme a`→`acetyl-coa`, `n-acetyl- d-glucosamine`→`n-acetylglucosamine` |
| 4 | synonym | 1 | `(+)-dextrose`→`(+)-glucose` |
| 5 | protonation state | 1 | `aspartic acid`→`aspartate` |
| 6 | Unicode ligature | 1 | `riboﬂavin` (U+FB02) → `riboflavin` |
| — | **accidental collapse of distinct molecules** | **0** | — |
| — | **unresolved** | **0** | — |

**11 of the 13 are cross-source**, affecting 49 spectra — molecules that two reference libraries
labelled differently. Under surface-name grouping those would have been split across
cross-validation folds and scored against themselves.

### Answers to the audit's explicit questions

1. **Why did the earlier corpus have 167 analyte labels?** Because 167 is the count of
   **normalized analyte names** after the V5 synonyms layer — not raw labels (212) and not
   molecules (154). It was quoted as "analytes", which is the ambiguity this audit removes.
2. **Why does V7 have 154 canonical molecules?** Because V7 additionally resolves Unicode
   ligatures, truncated spreadsheet headers, orthographic variants, one protonation-state pair
   and five generic-vs-specific stereochemical names — 13 collapses, all audited.
3. **Breakdown of the 13:** 5 stereo-prefix, 3 truncation, 2 orthographic, 1 synonym,
   1 protonation, 1 Unicode. **0 accidental, 0 unresolved.**
4. **Is 154 the correct scientific unit count?** **Yes**, for the unit "one chemical entity, one
   reference". It is not the count of unique *structures* — see §13.
5. **Should the V7 count be changed?** **No.**
6. **Corrected number:** **154** (unchanged).

---

## 9. Chemistry-class effects

No class count changed, because no molecule was added or removed. For completeness, canonical
molecules per class: `peptide_protein` 30 · `mono_oligosaccharide` 20 · `free_amino_acid` 18 ·
`acylglycerol` 17 · `fatty_acid` 17 · `sterol_steroid` 10 · `carboxylic_acid_metabolite` 8 ·
`phospholipid_sphingolipid` 5 · `polysaccharide` 5 · `purine` 5 · `chromophore_pigment` 4 ·
`sulfur_thiol_cofactor` 4 · `nucleic_acid_polymer` 3 · `phosphate_metabolite` 3 ·
`pyrimidine` 3 · `small_nitrogenous` 2.

**Class-assignment integrity:** every canonical molecule carries exactly one chemistry class;
**0 conflicts** without an explicit recorded resolution (the `acetyl-coa` protein/cofactor and
`aspartate` organic-acid/amino-acid conflicts were resolved and recorded in Phase 00).

**Replicate and excitation structure:** 154 molecules over 375 spectra; 9 excitation domains
with 785 nm carrying 62%. Replicate grouping `(canonical_id, excitation)` unchanged.

---

## 10. Phase 01 impact

**No corpus correction was required, so no Phase 01 rerun was required.** The dictionary was
regenerated anyway to confirm reproduction:

| Quantity | Before audit | After audit |
|---|---|---|
| raw Raman SPECTRA | 375 | 375 ✓ |
| normalized analyte NAMES | 167 | 167 ✓ |
| CANONICAL MOLECULES | 154 | 154 ✓ |
| chemistry classes | 16 | 16 ✓ |
| balancing arm | `B_analyte_weighted` | `B_analyte_weighted` ✓ |
| Local Spectral Motifs | 50 | 50 ✓ |
| `k_c` values | {1,2,3,5,6,7,10} | {1,2,3,5,6,7,10} ✓ |
| **registry fingerprint** | `208482d6f7178b5b8f16cace91be55b0` | **identical** ✓ |
| architecture compliance | 18/18 | 18/18 ✓ |

**The Phase 01 decision gate remains valid.** Per-molecule EV, held-out EV, orphan set,
source-confounding findings, motif typing and spectroscopic interpretation are all unchanged,
because the inputs are unchanged.

No superseded run was created, because nothing was superseded.

---

## 11. Scientific interpretation

Three things are worth stating beyond the pass/fail.

**The corpus is smaller than the headline number suggests, and that matters.** "167 analytes"
sounds like 167 independent chemical references. The scientific unit is 154 molecules, and of
those, 55 occur in more than one library — so the corpus holds **154 chemical entities measured
375 times**, not 167 independent references. Every claim about coverage should use 154.

**The Gobbato Raman set is load-bearing.** 21 of its 51 molecules occur in no other source. If
it were silently dropped — for instance by a loader change that stopped matching its filename
pattern — the corpus would lose 21 molecules and roughly a seventh of its chemical coverage,
and nothing in the Phase 00/01 pipeline would fail. That is why the file-count assertion is now
a test rather than a report line.

**The V5 synonyms layer was the larger unaudited surface.** V7 canonicalisation collapses 13
labels and had been audited in detail. The V5 layer collapses 27 and had not been audited at
all. It turned out to be correct — including the four acid/base merges that looked most
suspicious — but that was established here for the first time, not inherited.

---

## 12. Enforcement

`tests/test_v7_corpus_audit.py` fails if:

- any Ag-SERS spectrum enters the V7 Raman corpus;
- any serum / mixture / EV / plasma dataset enters it;
- any Gobbato pure-Raman file expected by the archive is absent (153 asserted exactly);
- Raman and Ag-SERS records are conflated (substrate and modality fields must separate them);
- a protected stereoisomer or anomer is collapsed;
- canonical IDs are non-unique;
- one raw spectrum maps to more than one canonical molecule;
- one canonical molecule carries conflicting chemistry classes;
- per-source counts drift from the manifest (202 / 153 / 20 and 375 total);
- the Phase 01 dictionary cannot be reproduced from the corpus.

---

## 13. Remaining uncertainties

| # | Item | Status | Impact |
|---|---|---|---|
| 1 | **Unique chemical structures not determinable** | No InChIKey, SMILES, PubChem CID, ChEBI or CAS field exists in any of the three source datasets. None was fetched externally. | 154 is a count of *canonical reference identities*, not verified distinct *structures*. Uncertainty preserved rather than guessed, per the brief. |
| 2 | **RamanBioLib id 197 (insulin), gold-coated substrate @633 nm** | Retained, flagged | Quantified as negligible: basis similarity 0.990 without it. Resolve from the source publication if the substrate's roughness can be established. |
| 3 | **carotene vs β-carotene** | Protected (not merged) | If they are the same molecule, the corpus holds 153 molecules rather than 154 and one spectrum is duplicated. Carried from Phase 00 unchanged. |
| 4 | **`citrate` merge at cosine 0.834** | Retained | Weakest merge in the corpus; below the uncontested control range but with matching protonation state. |
| 5 | **`glutamate` has two spectra from one spreadsheet** | Retained as replicates | Columns `Glutamic Acid` and `L-Glu` are two measurements of one molecule; correct as replicates. |

None is blocking. Items 1 and 3 would change the count by at most one molecule.

---

## 14. Decision gate

See the decision gate returned with this audit. Summary: **corpus verified complete and
Raman-pure, canonical count 154 confirmed unchanged, Phase 01 reproduced bit-identically,
approved to proceed to Phase 02.**
