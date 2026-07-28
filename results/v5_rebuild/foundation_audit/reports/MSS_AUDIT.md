# MSS_AUDIT
### The Molecular Spectral Signature layer — how motifs are made, and are they still right?

*Part 7 of the GAIRA Foundation Model audit. The MSS layer is the interpretable MIDDLE
of the hierarchy: between the 24 latent components ("what latent evidence") and the 11
biochemical themes ("which systems"), it names the **spectral motifs** that explain a
state. Source: `gaira.engine.mss`. Data dumped to `tables/mss_registry.json`.*

```
Radar (themes)   →  "which biochemical systems changed?"
MSS (this layer) →  "which spectral MOTIFS explain the change?"
Components       →  "what latent evidence supports the motifs?"
Reference        →  "which reference chemistries contributed?"
```

---

## 1. How an MSS motif is generated (exactly)

Each motif has two halves — a **curated definition** and a **derived provenance** — and
nothing frozen is touched:

**Curated (in `data/mss_motifs_v1.yaml`):** `id`, `name`, a short list of characteristic
Raman `bands_cm`, a few `exemplars` (reference chemistries), and a `parent_theme`. These
come from established Raman spectroscopy, not from the data.

**Derived (a pure function of the frozen artifacts).** For every component *j*, a score:

```
score(j) = 0.40 · band_score(j)      # do the component's dominant bands hit the motif's bands? (±16 cm⁻¹)
         + 0.35 · exemplar_score(j)  # do the motif's exemplar analytes load on this component?
         + 0.25 · theme_score(j)     # component→parent-theme weight from the ontology W
keep components with score ≥ 0.15 ;  take top 6 ;  normalize their scores → contributor weights
```

Then:
- **confidence = stability × evidence_breadth**, where stability is the weighted bootstrap
  stability of the contributing components and evidence_breadth is the mean fraction of the
  three evidence axes (band/exemplar/theme) each contributor satisfies.
- **perturbation** evidence (dose-response / serum-spike / uricase-depletion) is pulled
  from the component registry and matched to the motif's exemplars — linking each motif
  back to the calibration studies (Part 9). Uricase deltas are attached **only** to
  purine-parented motifs (they are purine-specific; elsewhere they would be misleading).

Region-based band matching (±16 cm⁻¹), never exact-peak matching — honouring GAIRA's
"peak ≠ molecule" principle. The derivation is deterministic → the MSS registry is
reproducible byte-for-byte.

---

## 2. The 13 motifs (derived from the frozen Raman atlas)

| Motif | Parent theme | Bands (cm⁻¹) | Exemplars | Top contributors | Conf. | Perturbation (dose/spike/deplete) |
|---|---|---|---|---|--:|---|
| purine_ring_breathing | nucleic_purine | 720,730,1250,1334,1480,1580 | adenine, guanine, hypoxanthine, xanthine | c3·.37, c15·.32, c0·.10 | 0.26 | 5 / 11 / 5 |
| oxopurine_carbonyl | nucleic_purine | 640,938,1230,1330,1580,1660 | xanthine, guanine, urate, hypoxanthine | c15·.44, c20·.15, c0·.12 | 0.26 | 6 / 7 / 5 |
| pyrimidine_ring | nucleic_pyrimidine | 560,578,790,1236,1396,1660 | uracil, cytosine, thymine | c17·.43, c13·.30, c23·.19 | 0.25 | 4 / 0 / 0 |
| aromatic_ring_residue | aromatic_amino_acid | 640,828,852,1006,1178,1614 | phenylalanine, tyrosine, tryptophan | c5·.32, c6·.20, c11·.15 | 0.27 | 6 / 0 / 0 |
| protein_amide_backbone | protein_peptide | 856,1000,1130,1240,1336,1654 | albumin, ubiquitin, trypsin | **c2·.49**, c0·.11 | **0.30** | 6 / 3 / 0 |
| lipid_acyl_chain | lipid_acyl | 1062,1130,1260,1302,1442,1654 | palmitic/arachidonic acid, tripalmitolein | c7·.28, c16·.23, c1·.21 | 0.29 | 6 / 0 / 0 |
| sterol_ring_system | sterol_membrane | 536,700,1250,1440,1670 | estrone, cholesterol | c3·.34, c7·.21, c8·.19 | 0.29 | 5 / 2 / 0 |
| glycan_co_network | saccharide_glycan | 816,872,916,1054,1108,1122 | glucose, galactose, fructose, cellulose | c10·.20, c4·.18, c18·.18 | 0.26 | 6 / 1 / 0 |
| carboxylate_organic_acid | organic_acid_metabolism | 760,940,1382,1398,1628 | citrate, succinic acid, pyruvate | c23·.40, c14·.35 | 0.26 | 4 / 0 / 0 |
| sulfur_heterocycle_thione | sulfur_antioxidant | 536,640,674,838,970,1200 | ergothioneine, cysteine, glutathione | c19·.28, c21·.19, c15·.18 | 0.26 | 6 / 2 / 0 |
| porphyrin_macrocycle | heme_porphyrin | 750,1130,1362,1590,1620 | hemoglobin, myoglobin, cytochrome c | c8·.33, c13·.25, c0·.23 | 0.27 | 5 / 0 / 0 |
| flavin_redox_cofactor | redox_broad | 754,1178,1348,1408,1578 | riboflavin, ubiquinone, NADH, FAD | c0·.47, c22·.16 | 0.28 | 5 / 0 / 0 |
| colloid_matrix_background | background_matrix (non-bio) | 930,1050,1400 | citrate, phosphate, urea | c14·.25, c21·.22 | 0.26 | 6 / 1 / 0 |

---

## 3. Does the MSS layer genuinely bridge maths → chemistry? (assessment)

**Yes, for the well-grounded motifs; honestly weak where the corpus is thin.**

- **Best-bridged: the purine system.** `purine_ring_breathing` and `oxopurine_carbonyl`
  draw on the two adenine components (c0/c3) + the clean purine component c15, hit the real
  adenine bands (722/1334) and oxopurine carbonyl (640/1660), and carry by far the richest
  perturbation evidence (dose 5–6, **serum-spike 7–11**, uricase-depletion 5). This is the
  motif→component→chemistry bridge working end to end — a spectral motif that also *moves
  correctly* when you dose or deplete its molecules (Part 9).
- **Strong: protein/amide (c2, conf 0.30 — the highest), lipid-acyl (c1/c7/c16),
  glycan (c4/c10/c12/c18), pyrimidine (c17).** Each maps to a clean component and the
  right bands.
- **Honestly weak: `porphyrin_macrocycle` and `flavin_redox_cofactor`.** Their exemplars
  (hemoglobin/cytochrome c; riboflavin/FAD) are barely present as *isolated* references in
  the corpus (Part 2 coverage gap), so their contributors are borrowed from purine/protein
  components (c8, c0, c13) and their band matches are weak. They are **provisional** and
  should be read as "possible heme/flavin-like signal," not assignments. The audit
  cross-validates the Part 2 gap: the two under-grounded motifs are exactly the two
  chemistries the corpus lacks pure examples of.

**A systematic ceiling worth noting:** `evidence_breadth = 0.33` for **every** motif — i.e.
on average each contributing component satisfies only **one of the three** evidence axes
(band OR exemplar OR theme), rarely two. Because confidence = stability × breadth, this
caps every motif's confidence in the 0.25–0.30 band regardless of how good its stability
is. This is not a bug — it reflects that a single NMF component rarely aligns on band,
exemplar *and* theme simultaneously — but it means the absolute MSS confidence numbers are
low by construction and should be read comparatively (purine/protein > porphyrin/flavin),
not as calibrated probabilities.

---

## 4. Should the MSS layer change after the Raman-only rebuild?

**No change required — and this is the key point of the audit.** The MSS layer is *already*
derived from the Raman-only frozen atlas (it reads the same `manifold_components.npz`,
Component Registry and ontology W that Parts 4–6 reproduced). There is no SERS contamination
to purge; the "important scientific change" the audit was asked to enforce is already the
state of the model. Concretely:

- The 13 curated motif definitions are chemistry, not data — they are unaffected by any
  rebuild.
- The derived contributors are a deterministic function of the frozen atlas — they
  reproduce exactly (fingerprint match, Part 4), so re-deriving changes nothing.

**Recommended (future, non-urgent):** (1) when the corpus gains isolated porphyrin/flavin
references, re-derive so those two motifs stop borrowing purine/protein components; (2)
consider a breadth-aware confidence that rewards multi-axis agreement, so the ceiling at
0.33 relaxes for genuinely multi-evidenced motifs. Neither requires touching the frozen
representation. **Do not** fold SERS references into the motif derivation — the motifs must
stay Raman-grounded; SERS is the thing they are used to *interpret*, not to define.
