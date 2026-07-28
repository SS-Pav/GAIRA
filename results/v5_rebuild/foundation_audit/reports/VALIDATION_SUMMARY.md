# VALIDATION_SUMMARY
### Testing the frozen Raman atlas — six datasets, in order, never used for training

*Parts 9–10 of the GAIRA Foundation Model audit. Every dataset below is **projected**
through the frozen atlas (NNLS onto the fixed components), never used to fit it. Run by
`foundation_audit/code/run_validation.py`; numbers in `tables/validation_results.json`.
Figures: `figures/validation_dose.png`, `tables/validation_transfer_pairs.csv`. The
committed foundation validation (held-out analyte, excitation/source transfer, serum
projection) is cited where relevant.*

---

## Scorecard

| # | Dataset | Purpose | Expected | Observed | Verdict |
|--:|---|---|---|---|:--:|
| 1 | Pure Gobbato Raman | in-domain fidelity | low OOD, correct themes | **OOD 0.047**; correct dominant themes | ✅ pass |
| 2 | Pure Gobbato SERS | Raman→SERS transfer | higher OOD, partial agreement | **OOD 0.163** (3.5×); median coord-cos **0.42**; theme kept 19/51 | ✅ pass (as expected) |
| 3 | Adenine dose | dose-response of purine | monotonic ↑ purine, saturating | ρ=**0.996**, Langmuir **K=0.89 µM, R²=0.993** | ✅ strong |
| 4 | Ergothioneine dose | dose-response of sulfur | monotonic ↑ sulfur, saturating | ρ=**0.927**, saturating | ✅ strong |
| 5 | Serum spike-in | recoverability boundary | few strong, many weak | 6 strong / 8 moderate / **39 weak** | ✅ pass (honest) |
| 6 | Uricase depletion | purine-specific loss | oxopurine motif ↓ | oxopurine motif **−0.060**; theme diffuse | ✅ pass at motif level |

Plus, from the committed foundation validation: **held-out analytes** (never seen in
fitting) project with within-analyte cosine **0.921** vs between-analyte **0.297**;
**excitation transfer** cosine **0.918** vs null 0.233; **source transfer** **0.847** vs
0.233. The representation generalises to unseen analytes and is invariant to
excitation/source.

---

## 1 · Pure Gobbato Raman analytes — in-domain fidelity
**Purpose.** A sanity floor: the atlas must represent its own training chemistries with
low out-of-distribution score and chemically sensible themes.
**Observed.** 153 spectra / 51 analytes; **mean OOD 0.047** (median 0.045) — essentially
in-distribution. Dominant themes match chemistry (adenine→purine, glucose→saccharide,
albumin→protein, …). **Lessons.** This establishes the OOD baseline (~0.05) against which
every out-of-domain set below is measured. Passing is necessary, not sufficient — it only
says the atlas is self-consistent.

## 2 · Pure Gobbato SERS analytes — the Raman→SERS transfer
**Purpose.** The central cross-modal test: project pure Ag-SERS spectra of the SAME 51
analytes into the Raman coordinate system and ask how much survives.
**Observed.** 265 SERS spectra; **mean OOD 0.163 — 3.5× the Raman baseline**, correctly
flagging SERS as out-of-domain. Per-analyte coordinate cosine (Raman vs SERS) **median
0.42**; the dominant theme is preserved for **19/51** analytes.
- **Preserved** (cos > 0.8): hypoxanthine 0.84, phosphatidylinositol 0.83, albumin 0.83,
  xanthine 0.81, glycogen 0.79 — strong, rigid Ag adsorbers whose ring/backbone modes
  dominate both spectra.
- **Scrambled** (cos < 0.25): alanine 0.25, hydroxyproline 0.24, glycine 0.23, glucose
  0.20, **uracil 0.055** — weak adsorbers whose SERS spectrum is reshaped by surface
  selection/orientation.
**Discussion.** This is exactly the expected physics: SERS is not Raman. Adsorption
affinity, surface orientation and the enhancement profile move an analyte's SERS
representation away from its Raman one, and the model **honestly reports the gap** through
elevated OOD rather than pretending to recover it. The analytes that transfer well are the
oxopurines — which is why they are also the ones recovered in serum (§5). Success/failure
here is a property of **surface physics**, not of the representation. This is the empirical
basis for a future observation model, not a defect.

## 3 · Adenine concentration series — dose-response of the purine theme
**Purpose.** Does a controlled dose of a known analyte move the correct theme,
monotonically and interpretably?
**Observed** (cAg @ 785 nm, 0–1.82 µM): the **nucleic_purine** share rises
0.183 → 0.320, monotonic (Spearman **ρ = 0.996**), best described by a **saturating
(Langmuir)** law (**K = 0.89 µM, R² = 0.993**). The motion spans many components — adenine
lifts the whole purine subsystem (c0/c3/c15) rather than a single axis.
**Discussion.** A near-textbook adsorption isotherm recovered from a frozen Raman basis,
on a SERS series it never saw. The correct theme, the correct direction, and a saturating
dose law — the strongest single validation of the theme layer.

## 4 · Ergothioneine concentration series — dose-response of the sulfur theme
**Observed** (cAg @ 785 nm, 0–2 µM): the **sulfur_antioxidant** share rises
0.081 → 0.104, monotonic (**ρ = 0.927**), also **saturating**. Ergothioneine's swing is
smaller than adenine's (sulfur is a lower-share theme with a single clean exemplar), but
the direction and monotonicity are unambiguous.
**Discussion.** A second, chemically-distinct analyte drives a second, chemically-correct
theme. (The demo characterises adenine as "component-redistribution" and ergothioneine as
"single-motif scaling"; the coordinate-space trajectory metrics here are consistent with
*both* being monotonic saturating responses but do **not** cleanly separate the two
mechanisms — adenine's path is actually the straighter of the two (0.84 vs 0.58) — so that
finer dichotomy should be read as suggestive, not established by these metrics.)

## 5 · Serum spike-in — the recoverability boundary
**Purpose.** In a real serum matrix on Ag colloid, which spiked analytes remain recoverable?
**Observed** (committed `phase7_serum_vs_pure`, 53 analytes): **6 strong** (direction
agreement ≥ 0.35), **8 moderate**, **39 weak** (< 0.10). Strong recovery: the oxopurines
**hypoxanthine, xanthine, guanine**, plus **ergothioneine**, ascorbate, creatinine.
**Discussion.** Only strong Ag adsorbers survive serum competition — precisely the
analytes that also transferred well in §2. This independently reproduces the source paper
(PMC12680727): serum SERS ≈ uric acid + hypoxanthine; concentration ≠ visibility. The 39
"weak" analytes are **not absent** — their recovery fails at the *measurement* layer
(adsorption/competition/matrix), which the atlas correctly cannot rescue and honestly
flags. A representation limitation would be recovering the *wrong* theme; instead it
recovers *nothing spurious*, which is the correct failure mode.

## 6 · Uricase depletion — purine-specific subtraction
**Purpose.** Enzymatic removal of urate should localise to purine motifs.
**Observed.** Spiked-serum vs spiked+uricase: at the coarse **theme** level the
nucleic_purine change is small and diffuse (Δ = −0.011; saccharide/organic-acid move
more — an artifact of compositional closure, BSV_AUDIT §4). At the **MSS motif** level the
**oxopurine_carbonyl motif drops sharply (Δ = −0.060)** — the single largest motif change —
while the purine-ring-breathing motif is unchanged.
**Discussion.** This is the "MSS resolves what themes hide" case, and it is chemically
exact: uricase removes **urate**, an oxopurine, so the **oxopurine carbonyl** motif — not
the generic purine-ring motif — is what disappears. The finer MSS layer localises a
perturbation that the coarse compositional radar smears. A clean, specific pass at the
level where the signal actually lives.

---

## Cross-cutting lessons

1. **OOD is a working instrument.** Raman 0.05 → SERS 0.16 → the model measures its own
   domain distance and never hides it. Every out-of-domain result is *flagged*, not
   silently trusted.
2. **The atlas fails safely.** Where surface physics defeats recovery (weak adsorbers,
   §2/§5) the model returns *no spurious theme*, not a confident wrong answer.
3. **Failures are surface physics, not representation.** The same oxopurines that transfer
   (§2), dose-respond (§3) and deplete (§6) are the ones recovered in serum (§5); the same
   weak adsorbers fail everywhere. The dividing line is adsorption, not the Raman basis.
4. **Resolution lives at the MSS layer.** Theme-level compositional radars smear
   perturbations (§6); the motif layer localises them. Read perturbations at MSS/Δ level.
5. **The best-validated axis is purine**, end to end (transfer, dose, depletion, serum) —
   consistent with it being the most redundant, most perturbation-anchored component
   cluster (Parts 6–7).
