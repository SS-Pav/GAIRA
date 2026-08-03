# GAIRA — Pure Ag-SERS Evaluation (V6)

### A detection gate before recovery: separating measurement failure from representation failure

*Additive on the frozen atlas `09ed804a40836f4a05a91ba10900cded`. Reuses V5 recovery flags
unchanged; adds a Stage-0 detection gate. Deterministic. Tables `tables/`, figures `figures/`,
metrics/thresholds validated in `code/validate_detection.ipynb` before freezing.*

Tags: **[obs]** · **[metric]** · **[interp]** · **[infer]** · **[lim]**.

## 1 · Executive summary
V5 evaluated all 51 analytes equally — including analytes whose Ag-SERS is essentially blank. That
conflates two failures: **measurement failure** (invisible on silver) and **representation failure**
(measured but chemistry not recovered). V6 inserts a **Stage-0 detection gate**: *does this Ag-SERS
spectrum contain reproducible analyte information above noise/background?* **22/51 analytes pass; 29
are undetectable.** Restricting the V5 hierarchy to the 22 detectable analytes roughly **doubles**
exact-identity recovery (14% → 23%) and lifts broad presence (MSS 40% → 55%, theme 49% → 59%) — but
analyte-**specific** recovery stays low (MSS 10%, theme 4.5%). **[metric]** So part of V5's poor
recovery was measurement failure; the residual, among measured analytes, is **genuinely
representational**. A learned Raman→SERS transfer model is justified for **~11 detectable,
representation-limited analytes**; the rest need a better substrate, not a model. **[infer]**

## 2 · Detection-gate philosophy
The gate asks a **measurement** question, not a GAIRA question. An analyte that gives no reproducible
peaks on silver cannot be "recovered" by any representation — that is a substrate problem. Only
detection-passing analytes are eligible for identity/motif/theme evaluation. This mirrors how
spectroscopy is actually done: confirm signal before interpreting it. **[interp]**

## 3 · Detection metrics & thresholds (validated, then frozen)
Deterministic weighted **Detection Confidence** (0–1), NO ML, from complementary spectroscopic
metrics on the 5 Ag-SERS replicates + the Ag serum-blank background:

| metric | what | weight |
|---|---|--:|
| replicate Pearson | reproducibility of mean-centred peak structure | 0.45 |
| replicate Spearman | rank reproducibility | 0.10 |
| peak SNR | max local peak signal vs replicate noise (log-scaled) | 0.20 |
| variance concentration | fraction of spectral variance in the top peaks | 0.15 |
| reproducible peaks | # peaks with local SNR > 3 | 0.10 |

Tiers: **GOOD ≥ 0.65 · MODERATE ≥ 0.50 · POOR ≥ 0.40 · UNDETECTABLE < 0.40**; pass = DC ≥ 0.50.
**Validated before freezing** (`validate_detection.ipynb`): the frozen weights/thresholds reproduce
the physically-expected pattern — the anchors PASS (xanthine 0.99, ergothioneine 0.97, urate 0.89,
adenine 0.63) and FAIL (glucose 0.43, tyrosine 0.42 POOR; oleate 0.33 UNDETECTABLE) for adsorption
reasons. Replicate **cosine** was rejected as a metric — it is baseline-inflated (0.93–0.99 for all),
the same shared-background problem as raw theme cosine. **[metric]**

## 4 · Representative spectra
The intuitive core (Figure 3): the oxopurines + ergothioneine show sharp, reproducible peaks well
above the Ag blank; glucose/tyrosine/oleate/methionine are blank-like noise with only spurious
peaks. This visually demonstrates *why* they fail — poor adsorption, not a GAIRA defect. **[obs]**

## 5 · Detectable vs undetectable
Tier counts: **GOOD 11 · MODERATE 11 · POOR 18 · UNDETECTABLE 11.** Pass (22): the oxopurines,
ergothioneine, glutathione, albumin, cholesterol, urea, guanine, hypoxanthine, adenine, and several
organic acids / weak-but-reproducible analytes. Fail (29): most amino acids, sugars, and unreactive
lipids. Two **edge cases** — creatinine and thymine — were exact-identity-recovered in V4/V5 yet fall
just below the gate (POOR), showing the gate is conservative and identity recovery ≠ detection in
rare cases. **[obs]**

## 6 · Hierarchical recovery (detectable only)
| level | all 51 | detectable-only | gain |
|---|---|---|--:|
| exact analyte | 7/51 (14%) | **5/22 (23%)** | +0.09 |
| NMF component | 2/51 (4%) | 2/22 (9%) | +0.05 |
| MSS present (top-3) | 19/48 (40%) | **11/20 (55%)** | +0.15 |
| MSS specific | 2/48 (4%) | 2/20 (10%) | +0.06 |
| theme present (top-3) | 25/51 (49%) | **13/22 (59%)** | +0.10 |
| theme specific | 1/51 (2%) | 1/22 (4.5%) | +0.03 |
| perturbation | 3/51 | 3/22 (14%) | +0.08 |

## 7 · Does abstraction improve once measurement failure is removed?
**Partly.** Presence and exact identity rise meaningfully among detectable analytes — so a real
fraction of V5's apparent failure was measurement, not representation. But **analyte-specific**
MSS/theme recovery stays low even among the 22 measured analytes: the deeper failure is
representational (the Ag surface reshapes the fingerprint), not merely that the analyte was invisible.
**[infer]**

## 8 · Transfer-model decision framework
Per analyte (Figure 7):
- **Case A · measurement-limited (29):** undetectable — no transfer model helps; needs a better
  substrate/observation channel.
- **Case C · already recoverable (5 detectable; 7 incl. the 2 edge cases):** detectable and exact
  identity already recovered — transfer unnecessary.
- **Case B · representation-limited, promising (11):** detectable, broad chemistry present, exact
  identity lost — **a learned Raman→SERS transfer model may help.**
- **Case B · representation-limited, hard (6):** detectable but no expected motif/theme present —
  transfer help uncertain.

## 9 · Learned-transfer roadmap (Figure 8)
- **already recoverable:** 7 · **potentially recoverable (worth trying):** **11** ·
  probably impossible (weak signal): 16 · impossible (measurement-limited): 11 · probably impossible
  (no chemistry present): 6.
The **~11 "potentially recoverable"** analytes are the concrete target set for future modality
correction. **[infer]**

## 10 · What GAIRA can / cannot claim
**Can:** state which analytes are measurable on Ag-SERS at all (22/51); confirm exact identity for a
strong-chemisorber subset; scope a learned transfer model to a defined ~11-analyte target. **Cannot:**
claim any recovery for the 29 undetectable analytes (a substrate problem, not a GAIRA problem); claim
class-discriminative recovery for most detectable analytes; treat detection confidence as identity. **[interp]**

## 11 · Limitations
No pure Ag-colloid buffer blank (the serum blank is used as the Ag background reference); 5 replicates
limit reproducibility resolution; detection weights are a transparent choice (validated, but not
unique); the gate is conservative (2 identity-recovered analytes fall just below it); Raman-trained
atlas; no learned modality model yet. **[lim]**

## 12 · Conclusions
Inserting a measurement gate before interpretation cleanly separates "we can't see it" from "we can't
recover it." Half the analytes are simply invisible on this substrate; among those that are visible,
recovery improves but specific chemistry still largely fails to transfer — a real representation gap
that a learned Raman→SERS model could target for ~11 analytes, while the rest await a better
substrate. **Recommended next experiment:** build that learned transfer model on the 11-analyte
target set, and extend dynamic perturbation (DART) — the only route that recovered class chemistry
in V5. **[interp]**

## 13 · Reproduction
```bash
python results/v5_rebuild/detection_gate_v6/code/detection_gate.py
python results/v5_rebuild/detection_gate_v6/code/restricted_hierarchy.py
python results/v5_rebuild/detection_gate_v6/code/make_figures_v6.py
python results/v5_rebuild/detection_gate_v6/code/make_report_v6_pdf.py
```
Threshold validation: `code/validate_detection.ipynb`. Interactive:
`streamlit run gaira_foundation_explorer_v6/app.py`. Frozen atlas unchanged.
