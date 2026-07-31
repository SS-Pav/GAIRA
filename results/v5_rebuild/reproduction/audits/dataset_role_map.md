# Dataset Role Map — representation vs interpretation vs validation

*Part 1 of the reproduction audit. Every dataset used anywhere in the foundation +
interpretation build, classified by role so that **representation-training spectra can
never be confused with validation spectra**. Machine-readable version:
`dataset_role_map.csv`.*

## The one rule to remember

```
Raman reference data  → representation construction (NMF)  → derived interpretation layers
SERS / perturbation   → (a) EXTERNAL VALIDATION, and
                        (b) PERTURBATION EVIDENCE feeding the interpretation layer only —
                            never the NMF, never the reference normalization.
```

## Representation-training corpus (Raman only — feeds the NMF)

| Dataset | Modality | Excitation | Spectra / labels | Loader | Raw or cached |
|---|---|---|---:|---|---|
| **RamanBioLib** | Raman | 9 excitations | 202 / 141 | `data.loader.load_ramanbiolib` | cached (in-repo derived parquet) |
| **gobbato_raman_metabolites** | Raman | 785 | 153 / 51 | `data.gobbato.load_gobbato_785` (Raman subset) | **raw** (Gobbato zip) |
| **amino_acid_raman_grounding** | Raman | 785 | 20 / 19 | `dataset._load_amino_acids` | **raw** (aa.xlsx) |
| **Union → X_ref** | Raman | — | **375 / 167** | `dataset.load_reference_corpus` | mixed |

`load_reference_corpus` **asserts** `modality == "raman"` — Ag-SERS/Au-SERS/DART cannot
leak into X_ref. These three sources (and only these) build the NMF basis **and** the
reference normalization frame.

## SERS / perturbation datasets (never train the representation)

| Dataset | Modality | Role in interpretation | Role in validation | How consumed |
|---|---|---|---|---|
| **gobbato_sers_metabolites** (pure Ag-SERS) | Ag-SERS 785 | — | Raman→SERS transfer | committed `phase3_projection_pure_sers.csv` |
| **adenine ILS series** | Ag/Au-SERS | dose fields (registry) + **25 % perturbation evidence** (theme weights) | dose-response | `phase3_projection_ils_adenine.csv` → `perturbation_response/part1,part2` |
| **ergothioneine series** | Ag-SERS 785 | dose fields + **25 % perturbation evidence** | dose-response | `phase3_projection_ergothioneine.csv` → perturbation tables |
| **serum spike-ins** | Ag-SERS 785 | serum-spike activators (registry) + perturbation evidence | recoverability | `phase3_projection_spiked_serum.csv` → `part2_response_fingerprints.csv` |
| **serum baseline** | Ag-SERS 785 | — | spike control | `phase3_projection_serum_baseline.csv` |
| **uricase depletion** | Ag-SERS 785 | depletion fields (registry) | purine-specific test | `phase3_projection_uricase.csv` + `part8_uricase.json` |

## Other

| Dataset | Role | Note |
|---|---|---|
| **raman_knowledge_core** | literature corroboration in the **foundation C3 axis** step only | self-authored ("GAIRA Curated Seed Pack"); **not** read by the engine's `build_theme_weights`/`build_registry` |
| **covid_serum_raman** | foundation C7 out-of-domain projection / blank control | never fits anything |

## The precise answer to "does SERS affect training or only validation?"

- **NMF representation:** Raman **only**. SERS never touches it. ✅
- **Reference normalization:** Raman **only** (projects X_ref). ✅
- **Interpretation layer (registry + component→theme weights):** consumes **SERS-derived
  perturbation evidence** — the adenine/ergothioneine dose responses, serum-spike
  activators, and uricase depletion — as documented annotation signals: **25 % of each
  theme weight** and the registry's dose/spike/depletion fields. This is *evidence-informed
  annotation*, not representation training, but it means the statement "SERS is validation
  only" is **true for the representation and incomplete for the interpretation**. State it
  as: *SERS validates the representation and additionally provides perturbation evidence to
  the interpretation layer; it never fits the NMF or the normalization frame.*
