# GAIRA V7 — Dataset and Provenance Context

What V7 learns from, what it must never learn from, and how provenance is carried.

Everything here describes the **current** state as of the V7 branch point. Phase 00 freezes
it into versioned manifests; until then, this document is the reference.

---

## 1. The grounding corpus — the only thing V7 fits

From `assets/foundation/manifold.json → corpus_card`:

| Field | Value |
|---|---|
| Observation domain | **Raman only** (canonical observation domain) |
| Spectra | 375 |
| Canonical analytes | 167 |
| Grid | 676 bins, 450–1800 cm⁻¹, 2.0 cm⁻¹ step |
| Preprocessing | asls baseline → savgol smoothing → L2 normalisation |
| Replicate groups | 272 (median size 1, max 3) |
| Analytes with replicates | 87 of 167 |
| Analytes with >1 excitation | 41 of 167 |

### Sources

| Source | Spectra | Provenance |
|---|---:|---|
| `RamanBioLib` | 202 | digitized reference library, 141 compounds, 9 excitations |
| `gobbato_raman_metabolites` | 153 | B&WTek 785 nm pure metabolite powders |
| `amino_acid_raman_grounding` | 20 | 20 pure amino-acid Raman references |

### Excitation distribution

| Excitation (nm) | Spectra |
|---:|---:|
| 785 | 234 |
| 1064 | 55 |
| 532 | 50 |
| 488 | 29 |
| 514.5 | 3 |
| 632.8 / 850 / 633 / 457.9 | 1 each |

**Excitation is a nuisance factor, and it is not evenly distributed.** 785 nm dominates
(62%), and the `gobbato` source is entirely 785 nm while `RamanBioLib` spans nine excitations.
Source and excitation are therefore **partially confounded**. The V5 build tracked excitation
for leakage and found all sources share the 450–1800 cm⁻¹ window; V7 must do the same and,
additionally, must check that the class partition (Strategy D) does not accidentally
partition by source. If a class is drawn overwhelmingly from one source, its local
decomposition may be modelling instrument response rather than chemistry. This is risk R-14
and Phase 02 must report a source/excitation composition table per class.

### Class composition

From `results/v6_rebuild/tables/p2_family_census.csv` — 18 families, 167 analytes:

| Family | Analytes | Phase-02 viability (indicative) |
|---|---:|---|
| protein | 32 | full local decomposition, multiple `k_c` candidates |
| saccharide | 27 | full local decomposition |
| amino_acid | 17 | full local decomposition |
| triglyceride | 15 | full local decomposition |
| organic_acid | 15 | full local decomposition |
| fatty_acid | 12 | full local decomposition |
| sterol | 9 | local decomposition, low `k_c` |
| cofactor | 6 | low `k_c`; chemically heterogeneous — review the partition |
| unknown | 6 | **partition review required — "unknown" is not a chemistry** |
| purine | 5 | low `k_c` |
| polysaccharide | 5 | low `k_c`; consider merging with saccharide — decide in Phase 00 |
| lipid | 5 | low `k_c`; overlaps fatty_acid/triglyceride — partition review |
| nucleic_acid | 3 | `k_c ≤ 1`, or anchor |
| pyrimidine | 3 | `k_c ≤ 1` |
| phospholipid | 2 | anchor candidate |
| small_nitrogenous | 2 | anchor candidate |
| carotenoid | 2 | anchor candidate |
| polyol | 1 | **anchor only** — no local fit possible |

**Three partition problems Phase 00 must resolve before any fitting:**

1. **`unknown` (6 analytes) is not a chemical class.** These analytes must be assigned real
   chemistry or excluded from the class partition (they can still be evaluated). Fitting a
   local decomposition over "unknown" produces a motif with no chemical meaning.
2. **`lipid` (5) overlaps `fatty_acid` (12) and `triglyceride` (15).** A three-way split with
   an ambiguous residual bucket will produce redundant LSMs across the three fits.
3. **`polysaccharide` (5) vs `saccharide` (27).** Chemically these differ by glycosidic
   polymerisation, which is spectroscopically real (V6 gave them separate motifs:
   `glycan_co_network` and `polysaccharide_glycosidic`). Keeping them separate is defensible;
   the decision must be recorded with its justification either way.

The partition used by V7 is **frozen in Phase 00** with a written chemical rationale per
class, and is versioned as an input artefact.

---

## 2. What V7 must never fit on

### Excluded observation domains (inherited from the V5 corpus card, unchanged)

- Ag-SERS
- Au-SERS
- DART
- serum Ag-colloid
- `metabolite-63` (633 nm Ag-SERS)
- adenine Ag-SERS series
- `european_multi_instrument_adenine` (cAg / sAg / cAu substrates)

**Why this exclusion is architectural, not a convenience.** SERS enhancement is selective:
it amplifies modes by surface affinity, orientation, and resonance, not by concentration or
chemical importance. Prior GAIRA work established this concretely — on Ag colloid, 50 of 51
analytes homogenise onto a purine-like attractor, and raw theme cosine similarity of 0.92 is
a *baseline artefact* requiring null correction, not evidence of preservation. A
representation fitted on SERS would encode the substrate's preferences as if they were
chemistry.

V7's position: **the Raman representation is the latent biochemical state; SERS is a
measurement channel applied to it.** The channel is modelled later, explicitly, as an
observation model — never by training the foundation on SERS.

### Projection-only sets

| Set | Role |
|---|---|
| `covid_serum_raman` | biological serum Raman (COVID / suspected / healthy / tube) — **projection only, never used for fitting** |

Biological, serum, and SERS material may be **projected through** a frozen V7 atlas to
evaluate transfer. It is never in the fit. This is the two-tier grounding rule GAIRA already
operates under and V7 inherits it verbatim:

> Pure Raman trains the frozen atlas. SERS, serum, and biological material are only
> projected through it, never fitted to it.

---

## 3. Canonical molecule identity — the V7 unit of account

V7's unit is the canonical molecule, so identity resolution is not housekeeping — it is the
foundation of the whole balancing argument. Phase 00 must produce a canonical analyte table.

**Known alias hazards observed in existing tables:**

| Observed variants | Issue |
|---|---|
| `riboflavin` / `riboﬂavin` | Unicode ligature `ﬂ` (U+FB02) — these appear as *separate* top-activating analytes in `p2_motif_audit.csv` |
| `(+)-dextrose` / `β-d-glucose` / `glucose` | stereochemical prefixes and common names |
| `acetyl coenzyme a` (listed under `protein`) / `acetyl-coa` (listed under `cofactor`) | same molecule, two names, **two different family assignments** |
| `urea` / `ure` | truncation |
| `13-methylmyristicacid` / `15-methylpalmiticacid` | missing space — inconsistent with `12-methyltetradecanoic acid` |
| `(+)-arabinose` / `(-)-arabinose` | enantiomers — **genuinely distinct**, must NOT be merged |

The acetyl-CoA case is the important one: it is not only an alias collision, it is a
**class-assignment inconsistency**. The same molecule sits in two families, which under
Strategy D would put it in two independent local fits.

**Phase 00 requirements:**

1. A canonical ID per molecule, with every observed surface form mapped to it.
2. **Unicode normalisation (NFKC) plus whitespace and case normalisation** before matching.
3. Stereochemistry preserved: enantiomers and anomers are distinct canonical IDs unless a
   written chemical justification says otherwise.
4. Exactly one class per canonical ID.
5. A manual review list of every merge and every near-miss that was *not* merged.
6. **Leakage test:** no canonical ID may appear in more than one CV fold. Aliases that survive
   undetected are a direct leakage path — the same molecule in train and test under two
   spellings — and would inflate every V7 metric. This is risk R-09 and it is a gate, not a
   nice-to-have.

---

## 4. Replicate groups and quality metadata

**Current state:** 272 replicate groups over 375 spectra; 87 analytes have >1 spectrum; max
group size 3. So the replicate imbalance is real but modest — the dominant imbalance is at
the *class* level (32:1), not the replicate level (3:1).

**Implication for Phase 01.** Strategies B and C differ by less than one might expect at this
corpus size, because most analytes have exactly one spectrum. The comparison is still
necessary, but Phase 01 should report the effect **restricted to the 87 replicated analytes**
as well as corpus-wide, or the headline number will be diluted to near-zero by the 80
singletons and the strategies will look falsely equivalent.

**Replicate group definition (Phase 00 must freeze):** whether two spectra of the same
molecule at *different excitations* form one replicate group or two. This is not obvious:

- as *one group*, excitation variation is absorbed into within-analyte spread — realistic for
  uncertainty, but it means the prototype averages across excitations, and 41 analytes are
  affected;
- as *separate groups*, each excitation gives its own reference — cleaner spectroscopically,
  but it re-inflates weight for the 41 multi-excitation analytes, partially undoing the
  balancing.

**Recommendation to be ratified in Phase 00:** treat `(canonical_id, excitation)` as the
replicate group, and apply analyte balancing at the level of `canonical_id` *across* those
groups. This preserves excitation as a tracked nuisance factor without letting
multi-excitation analytes buy extra weight.

**Quality metadata (Phase 00 must define and freeze).** Strategy B's weights depend on a
quality score `q`. Candidate components — SNR estimate, baseline-fit residual, cosmic-ray /
spike flags, grid coverage fraction within 450–1800 cm⁻¹, saturation flags, and a
per-source prior. The score must be frozen before Phase 01, or it becomes a hyperparameter
tuned to the outcome. A uniform-`q` sensitivity arm is mandatory.

---

## 5. Provenance chain

Every V7 artefact must be traceable to raw sources through an unbroken chain:

```
source dataset  →  raw spectrum file  →  canonical analyte ID
                                             ↓
                                     replicate group
                                             ↓
                          quality score + preprocessing config hash
                                             ↓
                            balanced reference row (Phase 01)
                                             ↓
                                  chemical class block
                                             ↓
                                     LSM (Phase 02)   ── stability, redundancy, bands
                                             ↓
                                     CSM (Phase 03)   ── contributing LSMs, classes,
                                             ↓                analytes, bands, uncertainty
                                theme membership (Phase 04)
                                             ↓
                                    BSV axis (Phase 05)
```

**Rule.** Each artefact records the IDs of its inputs one level down, plus the manifest hash
of the build that produced it. Given any BSV axis, the full set of supporting canonical
analytes, source datasets, and excitations must be recoverable by traversal. Given any source
dataset, everything downstream that depends on it must be recoverable by the reverse
traversal — which is what makes it possible to answer "what breaks if this dataset is
retracted?"

---

## 6. Data-root policy

**No raw spectra in Git. No absolute lab paths. No `SSD_Rad` defaults.**

Resolution order for raw data, inherited from `tools/reproduce_gaira_foundation.py`:

```
--data-root  >  $GAIRA_DATA_ROOT  >  optional documented default  >  error
```

Frozen atlas assets are committed and self-contained: `assets/foundation/` supports
inference and interpretation-only reproduction with **no raw data and no lab volume mounted**.
V7 must preserve this property — a clean clone must be able to run V7 inference from the
frozen V7 bundle alone. This is a Phase-06 gate.

Committed enabling assets that make raw-free reproduction possible:

| Asset | Path |
|---|---|
| Reference coordinates | `results/v5_rebuild/reproduction/manifests/nmf_reference_coordinates.npz` |
| Dataset role map | `results/v5_rebuild/reproduction/audits/dataset_role_map.csv` |

V7 needs equivalents of both, produced in Phase 00 and Phase 06 respectively.

---

## 7. Known corpus limitations carried into V7

Stated plainly so no V7 result over-claims:

1. **167 analytes is a small corpus** for a 18-class partition. Several classes are at or
   below the size where a local decomposition is meaningful.
2. **Coverage reflects reference-library availability, not biological importance.** Protein
   and saccharide references are abundant; sterol, flavin, and phospholipid references are not.
3. **Sphingolipids are absent entirely.** Not thin — absent.
4. **Source and excitation are partially confounded** (see §1).
5. **All references are pure compounds**, mostly powders. Real biological spectra are
   mixtures in matrices. The corpus supports learning a *chemical vocabulary*, not a *mixture
   model*; mixture behaviour is tested by projection (serum spike, EV, pathogen sets), never
   assumed.
6. **The corpus cannot validate the SERS observation model** — by construction, since SERS is
   excluded from fitting. That validation needs the paired Raman↔SERS material, and is out of
   V7's scope.

Phase 09 targets 1–3 specifically, driven by V7's own residual analysis rather than by
dataset availability.
