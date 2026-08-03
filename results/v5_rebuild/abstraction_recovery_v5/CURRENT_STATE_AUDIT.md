# V5 Current-State Audit (pre-analysis)

*Verified before writing any V5 code. Frozen atlas `09ed804a40836f4a05a91ba10900cded`; nothing
here modifies it.*

## Git
Branch `gaira-v5-rebuild-plan`, HEAD `0a3256e` (V4 tests). Remote `SS-Pav/GAIRA`. Explorers
V1–V4 present and committed.

## Inherited unchanged (regression-checked, not recomputed with new thresholds)
- **Level 0 — exact analyte identity (V4):** latent-specific **7/51**, MSS-specific **3/51**,
  theme-specific **4/51** (rank-1 + jackknife-stable), from
  `hierarchical_recoverability_v4/tables/per_analyte_evidence_profile.csv`. V5 reproduces the
  underlying `C_latent/C_MSS/C_theme_raw` bit-for-bit (same frozen calls) and cites the V4 flags.
- **Level 5 — perturbation:** adenine dose→purine, ergothioneine dose→sulfur, uricase→oxopurine
  (`validation_results.json`). Unchanged; used as functional confirmation only.
- **Level 6 — matrix:** serum spike (`phase7_serum_vs_pure.csv`). Unchanged; secondary this pass.

## New analyses required (V5)
1. **NMF component-evidence recovery** — top-k component overlap Raman↔Ag-SERS, component-mass
   retention over the Raman-dominant set, related-component redistribution, mismatched null.
2. **MSS motif recovery** — expected motif(s) per analyte (from chemistry + Raman activation, NOT
   from Ag-SERS height), null-adjusted enrichment, recovery rule, 90/95/99 sensitivity.
3. **Molecular-subclass overlay + LOAO classification** — a versioned evaluation overlay (NOT a
   new ontology axis); leave-one-analyte-out nearest-centroid in latent/MSS/theme spaces at
   subclass / broad-family / theme granularity; balanced accuracy, macro-F1, confusion,
   permutation null, bootstrap CI.
4. **Broad-theme recovery** — expected-theme rank/top-k, family-mismatched null enrichment,
   common-background (purine) correction.

## Dataset counts (verified)
- 51 matched Raman↔Ag-SERS analytes. Ag-SERS: 265 spectra / 5 replicates each (used only for the
  inherited V4 identity/stability; V5 classification uses per-analyte means → **no replicate
  leakage by construction**).
- 12 biochemical MSS motifs (2 purine: `purine_ring_breathing`, `oxopurine_carbonyl`), 11 themes.
- Families: amino_acid 16, organic_acid 7, saccharide 6, cofactor 5, purine 5, lipid 5,
  small_nitrogenous 2, pyrimidine 2, protein 1, polyol 1, polysaccharide 1.

## Canonicalization
Analyte labels already canonical (V4 `canonical()` / `GOBBATO_ABBREV`). `family_of()` provides the
broad family. No relabelling in V5.

## Ambiguous / mixed classifications (flagged for the overlay)
- **CoA cofactors** (acetyl-coa, coenzyme a) chemically **contain adenine** → a purine theme here
  is *legitimate chemistry*, not a purine-attractor artifact. Marked multi-label (purine_cofactor).
- **guanine** is both amino- and oxo-purine → multi-label.
- **phosphate** is inorganic (no carboxylate / no clean biochemical MSS motif) → expected MSS
  unassigned; theme organic_acid_metabolism is a weak fit → low confidence.
- **creatinine, urea** (guanidino small-N) have no strong ontology theme → low confidence.
- **glycerol** (polyol) grouped with saccharide_glycan by proximity → low confidence.
- Sulfur amino acids (cysteine thiol, methionine thioether) map to `sulfur_heterocycle_thione`
  only approximately (that motif is defined for thiones/S-heterocycles) → noted.

## Low-count subclasses (flagged EXPLORATORY; excluded from primary LOAO accuracy denominator)
Singleton subclasses cannot be classified under leave-one-analyte-out (the true class centroid
disappears when the sole member is held out): flavin (riboflavin), sterol (cholesterol),
triacylglycerol (triolein), phospholipid (phosphatidylinositol), phosphorylated_sugar
(fructose-6-phosphate), amino_sugar (n-acetylglucosamine), polyol (glycerol), polysaccharide
(glycogen), protein (albumin), inorganic_phosphate (phosphate). Reported separately; the primary
subclass metric is over subclasses with ≥2 members.

## Hard-coded family/theme rules currently in use (reused, documented)
`families_raman.family_of()`; the V2/V3/V4 `FAM_THEME` + `ANALYTE_THEME` expected-theme map
(purine→nucleic_purine, sulfur amino acids/thiols→sulfur_antioxidant, riboflavin→redox_broad,
etc.). V5 adds an explicit `expected_mss` and `subclass` overlay on top — it does not change these.

## Integrity checks (must pass before analysis)
- 51 analytes load; families sum to 51. ✓
- MSS motif→parent_theme map has 12 biochemical motifs. ✓
- Every analyte gets a broad family; subclass may be `mixed`/`unassigned` (not forced). ✓
- Frozen fingerprint `09ed804a…`. ✓
