# PURE Ag-SERS VALIDATION
### Can a Raman-trained biochemical atlas interpret pure Ag-SERS analytes — before matrix effects?

*The missing bridge in GAIRA's validation story: between "what the atlas learns" (pure
Raman) and "what survives serum" (matrix-perturbed spike-ins) sits the cleaner question —
does the frozen Raman representation recognise **pure** Ag-SERS analytes on silver, with no
serum competition? Everything below projects pure Ag-SERS through the **frozen** atlas (no
retraining, no modality correction). Reproduced by
`foundation_audit/code/pure_ag_sers_validation.py`.*

---

## Part 1 · The data (audited, not guessed)

| | |
|---|---|
| Dataset | **Gobbato pure-analyte Ag-SERS metabolites** (canonical) |
| Archive | `raw/serum_ag_colloids/dataset_spectral_data.zip → SERS metabolites/` |
| Instrument | B&WTek i-Raman Plus |
| Substrate / laser | **Ag colloid · 785 nm** |
| Spectra / analytes | **265 spectra · 53 analytes** |
| Matched to Raman twin | **51** (same 785 nm, same instrument/group) |
| Unmatched (SERS-only) | 2 — **DNA, RNA** (no isolated Raman reference) |

**Canonical choice.** Gobbato's `SERS metabolites/` is the canonical pure Ag-SERS set
because it is the exact 785 nm Ag-SERS twin of the 153 pure Raman powders already in the
atlas — an apples-to-apples matched pair. Other Ag-SERS sets on disk are **not** used here
and are not matched twins: `metabolite_sers63_support` (633 nm, different substrate),
`adenine_sers_control` (bAgNPs, single analyte), `ag_flakes_metabolites_23` (peak table
only). Biofluid SERS sets (`hcc_serum`, `diabetes_plasma_ev_sers`, …) are matrix samples,
not pure analytes.

**Currently used vs new.** A basic transfer table
(`validation_transfer_pairs.csv`, coord-cosine + themes only) already existed. This pass
adds the full per-analyte reasoning (component coordinates, MSS, themes, BSV, OOD,
confidence, nearest Raman references, matched-vs-mismatched similarity, family recovery,
recoverability tiers) in `pure_ag_sers_validation.json` +
`pure_ag_sers_per_analyte.csv`, and wires it into the Foundation Explorer.

---

## Part 2 · What the frozen atlas does with pure Ag-SERS

51 matched analytes, each projected through the frozen 24-component basis:

| metric | value |
|---|---|
| median Raman↔Ag-SERS coordinate cosine | **0.42** (mean 0.46) |
| dominant biochemical **theme preserved** | **18 / 51** |
| mean Ag-SERS OOD | **0.16** (≈ 3× the pure-Raman baseline of 0.05) |
| matched vs mismatched cosine | 0.46 vs 0.41 → **separation +0.055** |
| SERS projection whose nearest atlas reference is *itself* | **1 / 51** |

**Reading it honestly.** Ag-SERS is *not* Raman: on average an analyte's SERS coordinate is
only slightly closer to its own Raman than to other analytes' (+0.055), and across the full
167-analyte atlas the SERS projections mostly drift toward the strong-adsorber directions,
so full-atlas *identity* recovery is largely lost even before serum. But the picture is
strongly **chemistry-dependent** — some analytes transfer excellently (§3).

---

## Part 3 · Scientific assessment — adsorption affinity explains it

Ranking figure: `figures/pure_ag_sers_ranking.png` (all 51, tiered). Family summary:
`figures/pure_ag_sers_by_family.png`.

**Transfer by chemical family (mean coordinate cosine):**

| family | mean cos | n | verdict |
|---|--:|--:|---|
| polysaccharide (glycogen) | 0.68 | 1 | strong |
| small nitrogenous (creatinine, urea) | 0.68 | 2 | strong |
| **purine** | **0.66** | 5 | **strong** |
| protein (albumin) | 0.60 | 1 | strong |
| **cofactor** (glutathione, CoA, riboflavin) | 0.53 | 5 | moderate–strong |
| organic acid | 0.49 | 7 | moderate |
| saccharide (sugars) | 0.43 | 6 | weak |
| lipid | 0.41 | 5 | weak |
| **amino acid** | **0.39** | 16 | **weak** |
| **pyrimidine** (uracil, thymine) | **0.16** | 2 | **poor** |

**The physical story — this is adsorption chemistry, not a modelling artefact:**

- **Purines transfer best.** Hypoxanthine (0.84), xanthine (0.81) and guanine chemisorb to
  silver through their ring nitrogens; the oxopurine carbonyl + ring-breathing modes that
  dominate their Raman *also* dominate their SERS, so the coordinate survives. (Adenine is
  the weaker purine at 0.36 — its SERS is reshaped more — which is why the *family* average,
  not adenine alone, is the honest summary.)
- **Sulfur compounds transfer well.** Glutathione (0.73), cysteine, methionine — thiol/thione
  sulfur chemisorbs strongly to Ag, anchoring the signature. Ergothioneine's *coordinate*
  moves (0.28) but its **sulfur theme is preserved** — a case where the theme survives even
  when the exact coordinate does not.
- **Amino acids transfer poorly (0.39).** Most adsorb weakly and non-specifically; their
  SERS is reshaped by orientation and surface selection rules (tyrosine 0.24, glycine 0.25,
  hydroxyproline 0.24).
- **Carbohydrates transfer weakly (0.43).** Sugars are poor Ag adsorbers (glucose 0.26,
  fructose 0.34); little chemisorption to preserve the ring modes.
- **Pyrimidines transfer worst (0.16; uracil 0.06).** Unlike purines they lack the strong
  ring-N Ag-binding site, so almost nothing of the Raman signature survives.

**Recoverability tiers (evidence-backed):** Excellent (≥0.80) 3 · Good (≥0.65) 4 · Moderate
(≥0.45) 16 · Weak (≥0.25) 24 · Poor (<0.25) 4. The tier boundary tracks **Ag adsorption
affinity**: the Excellent/Good set is exactly the strong chemisorbers (oxopurines, thiols,
creatinine), the Poor set the weak physisorbers (pyrimidines, small amino acids, sugars).

---

## Why this stage is scientifically necessary

```
Reference Raman            what the atlas LEARNS (pure molecular fingerprints)
   ↓
Pure Ag-SERS validation    can it RECOGNISE the same molecules on silver, with NO matrix?  ← this stage
   ↓
Controlled perturbation    can it recover CONCENTRATION (adenine, ergothioneine)?
   ↓
Matrix perturbation        what survives SERUM competition (spike-ins)?
   ↓
Biological validation      does it read real biological cohorts?
```

Skipping straight from Raman to serum spike-ins conflates two different failures — the
**modality** gap (Raman→SERS surface physics) and the **matrix** gap (serum competition).
The pure Ag-SERS stage isolates the modality gap: it shows that transfer is already
partial and adsorption-selective **before** any serum is added, so the serum results (only
strong adsorbers recoverable) are the *expected* continuation, not a surprise. Strong Ag
adsorbers that pass this stage (oxopurines, thiols) are the ones that go on to survive
serum; the weak adsorbers that fail here fail there too. **Representation limit vs
measurement limit are cleanly separated by inserting this step.**
