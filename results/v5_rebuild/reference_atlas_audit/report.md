# GAIRA Raman Reference Atlas v0.1 — Component Audit

**Read-only audit** of the frozen NMF k=24 Raman reference atlas. Branch `gaira-v5-rebuild-plan`; nothing pushed. The atlas was **not retrained, reweighted or reselected** — its fingerprint (`09ed804a40836f4a05a91ba10900cded`) was verified on load and re-verified after every analysis. Stage A / Stage B / Stage B0, previous reports and the demo were not modified.

> **Principal finding: the atlas has learned molecular CLASS, not molecular species.** Lipids, carbohydrates and proteins each cohere significantly against a random-analyte null (p = 0.005); amino acids, purines, pyrimidines, cofactors and organic acids do not. Components are highly reproducible (mean bootstrap stability **0.812**) but only modestly pure (mean class purity **0.347**) — they are chemically *enriched* rather than chemically *pure*. The 24 components should therefore remain canonical as the **numerical** layer, with biochemical meaning carried by a revisable **theme overlay** (hierarchical BSV, Option C).

---

## Reproduce

```bash
cd results/v5_rebuild/reference_atlas_audit/code
python run_audit.py            # P1-P7, P9, P12, P13  (~6 s)
python run_confusability.py    # P12 supplement — chemistry vs atlas failure
python run_synthesis.py        # P10 ontology, P11 BSV design, P14 assessment
python make_atlas_pdf.py       # figures + 39-page atlas PDF
```

---

## What the atlas learned (P9)

Within-family BSV cosine against a 200-draw random-analyte null:

| family | within-family cosine | null | p | coheres |
| --- | --- | --- | --- | --- |
| polysaccharide | 0.872 | 0.254 | 0.005 | **yes** |
| triglyceride | 0.797 | 0.252 | 0.005 | **yes** |
| fatty acid | 0.643 | 0.258 | 0.005 | **yes** |
| protein | 0.590 | 0.253 | 0.005 | **yes** |
| sterol | 0.576 | 0.253 | 0.005 | **yes** |
| saccharide | 0.400 | 0.254 | 0.005 | **yes** |
| cofactor | 0.381 | 0.245 | 0.065 | no |
| organic acid | 0.299 | 0.252 | 0.144 | no |
| amino acid | 0.299 | 0.252 | 0.139 | no |
| purine | 0.287 | 0.254 | 0.284 | no |
| pyrimidine | 0.286 | 0.239 | 0.323 | no |

Lipid chemistry is the strongest signal in the atlas, then carbohydrates and proteins. The failure of amino acids and nucleobases to cohere is chemically sensible: individual amino acids differ mainly in side-chain modes that are weak relative to the shared backbone, and only five purines / three pyrimidines are present.

---

## Component inventory (P1–P5)

- **13 high / 11 moderate** interpretive confidence; none uninterpretable.
- Mean class purity **0.347**, mean bootstrap stability **0.812**.
- Enrichment reaches **14×** (pyrimidine, c17) and **10×** (purine, c15) even where purity is modest — the signature of a small but real chemical theme inside a broader mixture.
- **Only 1 of 24** components satisfies the strict "true biochemical theme" test (purity ≥ 0.40 **and** stability ≥ 0.80): **c2**, a protein component (purity 0.803, stability 0.965). Six more exceed purity 0.40 but fall just below the stability bar.
- Themes assigned: protein ×7, saccharide ×6, amino acid ×3, triglyceride ×2, pyrimidine ×2, organic acid ×2, sterol ×1, purine ×1.

Spectroscopically the components are legible. **c1** (variance 0.119, stability 0.955) peaks at **1302 and 1442 cm⁻¹** — the CH₂ twisting and scissoring modes of aliphatic acyl chains, i.e. a textbook lipid-chain component drawing on triacylglycerols and fatty acids.

**Verdict on P4:** most components are best described as *stable mathematical mixtures capturing shared Raman motifs* — chiefly the acyl-chain C–H bands common to every lipid subclass and the C–O/C–C ring modes common to carbohydrates — rather than as single biochemical themes.

---

## Component relationships and grouping (P6–P7)

| k | silhouette | bootstrap reproducibility | chemical coherence | composite |
| --- | --- | --- | --- | --- |
| 6 | 0.059 | 0.884 | 0.259 | 0.000 |
| 8 | 0.078 | 0.900 | 0.260 | 0.162 |
| 10 | 0.079 | 0.938 | 0.281 | 0.501 |
| 12 | 0.094 | 0.960 | 0.308 | 0.872 |
| **14** | **0.095** | **0.969** | **0.308** | **0.953** |
| 16 | 0.083 | 0.976 | 0.312 | 0.908 |

**Silhouette never exceeds 0.10 at any k.** The components do *not* form geometrically well-separated clusters. Bootstrap reproducibility is high (0.88–0.98) because the same weak structure is recovered consistently — reproducible is not the same as well-separated. k=14 wins the composite, but the margin over k=12 and k=16 is small.

**Conclusion:** higher-order grouping is justified as an **interpretive overlay**, never as a discovered partition or a frozen basis.

---

## MSS readiness (P12) — the key nuance

Median signature uniqueness is **0.053**, which looks disqualifying until the confusions are examined:

| nearest-neighbour relation | n |
| --- | --- |
| same chemical family | 66 |
| homologous series (same subfamily) | 33 |
| same molecular class | 26 |
| duplicate entry / synonym | 10 |
| **genuinely different chemistry** | **32** |

**88%** of low-uniqueness cases are duplicates, homologous series or same-class chemistry — and only **16** analytes are both low-uniqueness *and* confused with genuinely different chemistry. The saturated triacylglycerols (trilaurin → tribehenin) are mutually indistinguishable because in the 450–1800 cm⁻¹ fingerprint region they differ only in CH₂ count; estradiol/estriol are likewise near-identical. **This is correct spectroscopy, not atlas failure.**

**Verdict:** MSS can be frozen at **class/theme level with documented confusable groups**. It cannot be frozen at molecular-species level. Species resolution would need the 2800–3000 cm⁻¹ C–H stretch region (which separates acyl chain lengths) or an orthogonal measurement — not a change to this atlas.

---

## Ontology v0.1 (P10) and BSV recommendation (P11)

Provisional ontology tiers: **6 high · 10 moderate · 6 low · 2 unknown** confidence. Not frozen.

**BSV recommendation: Option C — hierarchical.**

| option | verdict |
| --- | --- |
| A — 24 canonical coordinates | retain as the canonical *numerical* layer, but not as the reported BSV (24 weakly-named axes are not communicable) |
| B — compressed to ~14 | insufficient alone; discards resolution the atlas actually has, and rests on a weak-silhouette partition |
| **C — hierarchical** | **recommended**: 24 frozen latent components → versioned biochemical themes → optional visual summary |

The hierarchy keeps the two properties in the right places: the components are where the atlas is *stable* (0.812), the themes are where it is *interpretable*. Because the theme layer is an overlay, the ontology can be revised without ever touching the frozen numerical basis. **The BSV is not defined or frozen here — only its architecture is recommended.**

---

## Out-of-domain stress test (P13) — NOT validation

Three **Ag-SERS** calibration sets were projected into the **Raman** atlas purely to observe off-domain behaviour: adenine concentration series (6), ergothioneine calibration (55), uricase depletion (20). Median out-of-domain distances are 0.241 / 0.281 / 0.241 — materially higher than in-domain Raman references, as expected for different observation physics.

**No dose-response claim is made.** Prior work (Stage B0, spectral audit) established that Ag-SERS spectra in this corpus are background-dominated, so any apparent trajectory here is not evidence of biochemical response. These projections were not used to modify, tune or judge the atlas.

---

## Final assessment (P14)

1. **What chemistry?** Lipid-dominated, then carbohydrate and protein; nucleobases appear as small but strongly enriched components; amino acids, cofactors and organic acids do not cohere.
2. **Strongest components:** c2 (protein, purity 0.80, stability 0.97), c1 (lipid acyl chain, variance 0.119, stability 0.955), c17/c15 (pyrimidine/purine, 14×/10× enrichment).
3. **Weakest:** c13, c19, c22 and c9 — low purity, low enrichment or stability below 0.75; these should not anchor ontology entries alone.
4. **Keep 24 canonical?** Yes, as the numerical layer — it is the level at which the atlas reproduces.
5. **Group components?** Yes, but only as an overlay.
6. **Most defensible grouping:** k=14 by composite, provisional (small margin over k=12/16).
7. **BSV dimensionality:** hierarchical, not flat.
8. **MSS mature?** At class level yes; at species level no.
9. **Ontology confidence:** 6 high / 10 moderate / 6 low / 2 unknown.
10. **Next phase:** define the hierarchical BSV v0.1 on the frozen 24-component layer with a versioned theme overlay, and publish MSS at class level with explicit confusable-group annotations. Do not expand k, do not retrain, do not admit Ag-SERS.

---

## Method notes and limitations

- **Audit-stage family refinement.** The audit found gaps in the foundation family assignment (monounsaturated fatty acids such as elaidic/vaccenic/palmitoleic falling to "organic acid", triacylglycerols with digits in their names, greek-lettered phospholipids). These are corrected **locally in the audit code**; `src/gaira/foundation/families_raman.py` was deliberately left unchanged so the foundation report stays reproducible.
- **Molecular weight and biochemical role are reported as `unavailable`** for every analyte: this corpus contains no formulas, SMILES or curated role table, and inventing them was not acceptable.
- **Average molecular similarity (P4) is `unavailable`** for the same reason — no structures means no structural fingerprint similarity.
- Purity, enrichment and theme labels depend on the family assignment, which is rule-based from chemical names; 6 of 167 analytes remain unassigned.
- UMAP appears once, explicitly marked exploratory, and is not used as evidence anywhere.

---

## Outputs

| path | contents |
| --- | --- |
| `GAIRA_Raman_Reference_Atlas_v0.1_Component_Audit.pdf` | 39 pages: cover, executive summary, 4 atlas-overview figures, **one page per component (24)**, coherence, grouping, ontology, BSV study, MSS readiness, stress-test appendix, final assessment |
| `tables/p1_component_inventory.csv` | master 24-row inventory (15 columns) |
| `tables/p2_full_analyte_composition.csv` | every analyte contributing to every component |
| `tables/p4_chemical_coherence.csv` | entropy, purity, enrichment, MI, spectral similarity, ranked |
| `tables/p5_spectral_interpretation.csv` | bands, widths, importance, uniqueness, literature groups |
| `tables/p6_*.csv`, `artifacts/p6_relationship_matrices.npz` | correlation / cosine / shared-analyte / shared-band matrices |
| `tables/p7_grouping_study.csv`, `p7_group_composition_k14.csv` | grouping evaluation and composition |
| `tables/p9_biological_plausibility.json` | family coherence vs null |
| `tables/p10_ontology_v0_1.json` | provisional ontology (not frozen) |
| `tables/p11_bsv_design_study.json` | A/B/C evaluation and recommendation |
| `tables/p12_mss_readiness*.csv`, `p12_confusability_summary.json` | MSS readiness + confusability classification |
| `tables/p13_out_of_domain_stress_test.csv` | Ag-SERS projections (labelled out of domain) |
| `tables/p14_final_assessment.json` | the ten answers |
| `figures/` | 5 publication figures · `artifacts/audit_manifest.json` (fingerprint verification) |
