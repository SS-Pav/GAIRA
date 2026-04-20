# GAIRA-core concepts

## 0. Purpose of this document

This document defines the core concepts, scientific philosophy, system boundaries, architecture, and development constraints for GAIRA.

It is the long-term reference that future Claude prompts, implementation work, analysis scripts, and model decisions must follow.

If a future implementation, prompt, prototype, or analysis conflicts with this document, this document should be treated as the source of truth.

This document is intentionally comprehensive. It is not a temporary note. It is the architectural backbone for the GAIRA program.

---

## 1. What GAIRA is

GAIRA is a grounded biochemical state inference system for Raman and SERS spectra of biological samples.

Its central aim is not generic peak matching, document retrieval, or disease classification.

Its central aim is:

> To convert a Raman or SERS spectrum into a transparent, evidence-grounded biochemical state representation that is stable enough to validate on controlled calibration datasets and useful enough to compare across biological datasets and dynamic perturbation experiments.

The primary representation in GAIRA is the **Biochemical Support Vector (BSV)**.

A BSV is not a disease score and not a latent embedding label. It is a structured biochemical state estimate derived from grounded spectral evidence.

---

## 2. Why GAIRA exists

Raman and SERS spectra of biological samples are underdetermined mixture measurements.

A biological spectrum is usually a superposition of many contributors:
- proteins and peptides
- nucleic acids and nucleobases
- lipids and membrane-associated chemistry
- carbohydrates and glycans
- redox-active molecules
- salts, matrix effects, substrate effects, and measurement artifacts

This makes several naïve approaches insufficient:

### 2.1 Pure peak assignment is insufficient
A single peak can have multiple plausible contributors depending on matrix, substrate, adsorption behavior, and acquisition conditions.

### 2.2 Pure classification is insufficient
A disease classifier may separate cohorts while learning nuisance structure such as substrate type, acquisition protocol, or dataset-specific preprocessing.

### 2.3 Pure literature retrieval is insufficient
Literature retrieval can support interpretation, but it does not by itself generate a stable biochemical coordinate system for spectra.

### 2.4 Pure global embeddings are insufficient
A model trained across many heterogeneous datasets may learn dataset identity, acquisition regime, or sample type instead of cross-dataset biochemical structure.

GAIRA exists to solve this by building an intermediate, explainable biochemical state layer between raw spectra and downstream biological interpretation.

---

## 3. The core scientific hypothesis

The core hypothesis of GAIRA is:

> A Raman or SERS spectrum can be mapped into a constrained biochemical state space using evidence grounded in reference spectra and grounded literature assignments, and that state space can be validated through controlled compositional perturbations.

If that hypothesis holds, then the resulting BSV representation should:
- respond correctly to spike-ins
- respond correctly to degradation or depletion
- behave coherently for mixtures
- remain stable across replicates
- preserve shared biochemical signals across datasets better than raw spectral similarity alone

If it does not do these things, then the representation is not yet scientifically valid, regardless of how attractive the plots look.

---

## 4. What success looks like

GAIRA succeeds only if the biochemical state representation is experimentally and computationally defensible.

### 4.1 Controlled calibration success
For calibration datasets, known perturbations should produce expected directional changes in BSV space.

Examples:
- spiking a purine-associated analyte should increase the relevant biochemical support vector(s)
- enzymatic degradation of nucleic-acid-associated content should decrease the relevant biochemical support vector(s)
- concentration series should show monotonic or saturating trends where chemically expected
- mixtures should occupy sensible intermediate regions in BSV space

### 4.2 Reproducibility success
Replicates within the same condition should produce similar BSV vectors relative to the shift induced by the controlled perturbation.

### 4.3 Cross-dataset coherence success
Datasets with different acquisition regimes may still differ in raw spectral appearance, but once mapped into BSV space, shared biology should become easier to compare.

### 4.4 Dynamic utility success
Under DART-Met perturbation, time-varying spectra should generate time-varying BSV trajectories that are more interpretable than raw spectral movies.

---

## 5. What GAIRA is not

GAIRA is not:
- a generic LLM retrieval demo
- a pure literature peak-assignment engine
- a molecule identification engine claiming exact composition
- a disease classifier disguised as biochemical reasoning
- a database browser for Raman references
- a replacement for careful spectroscopy or chemistry

GAIRA should never overclaim certainty.

It should prefer statements such as:
- "evidence supports increased purine-associated contribution"
- "the biochemical state shifts toward nucleic-acid-associated structure"
- "lipid-associated support increases under this perturbation"

It should avoid statements such as:
- "this spectrum contains molecule X with certainty"
- "this disease contains these exact molecules"

---

## 6. The architectural separation that must be preserved

A major design requirement is strict separation between biochemical grounding and disease interpretation.

### 6.1 GAIRA-core concepts layer map

#### Layer A — GAIRA-Base
The grounded biochemical state engine.

Purpose:
- convert spectra to BSV vectors
- use only grounded spectral and biochemical evidence
- remain disease-agnostic

#### Layer B — GAIRA-Validate
The calibration and validation engine.

Purpose:
- test whether GAIRA-Base behaves correctly under controlled perturbations
- quantify monotonicity, specificity, reproducibility, and coherence

#### Layer C — GAIRA-Cohort
The dataset comparison and cohort analysis engine.

Purpose:
- apply GAIRA-Base to target datasets at sample level
- summarize cohort-level distributions and variability
- identify recurring biochemical differences without redefining the BSV axes

#### Layer D — GAIRA-Interpret
The literature- and context-aware interpretation engine.

Purpose:
- explain BSV shifts using biological context and literature
- add domain framing such as serum vs EV vs tissue
- never alter the core BSV computation rules directly

#### Layer E — GAIRA-Dynamic
The dynamic biochemical trajectory engine for DART-Met.

Purpose:
- compute BSV as a function of time, voltage, waveform, and perturbation state
- represent biochemical trajectories under controlled electrochemical perturbation

This separation is essential. If disease literature starts defining the biochemical axes, the system becomes circular.

---

## 7. The representation: what a BSV actually is

A BSV is a vector of biochemical axis scores derived from grounded spectral evidence.

Each axis score should reflect compatibility between an observed spectrum and a body of grounded evidence associated with a biochemical theme.

Examples of biochemical themes include:
- nucleic-acid-associated structure
- purine or nucleobase-associated structure
- protein or peptide-associated structure
- aromatic amino acid-associated structure
- lipid or membrane-associated structure
- oxidized lipid or carbonyl-associated structure
- carbohydrate or glycan-associated structure
- sulfur, thiol, or redox-metabolite-associated structure

A BSV axis is therefore:
- evidence-based
- continuous rather than binary
- explainable in terms of supporting and conflicting spectral features
- independent of disease labels

A BSV is not required to identify exact molecules. It is a biochemical support estimate.

---

## 8. How BSV axes should be developed

The initial BSV axes should not be treated as arbitrary design choices, but they should also not be left to unconstrained discovery.

The correct strategy is **chemically constrained experimental discovery**.

### 8.1 Start with an overcomplete candidate axis set
Begin with a larger fine-grained candidate set informed by chemistry and spectroscopy.

Examples of candidate axes may include:
- nucleic acid phosphate backbone
- purine or nucleobase
- pyrimidine-like features if justified
- protein backbone or peptide
- aromatic amino acid side-chain features
- aliphatic lipid chain
- unsaturated or oxidized lipid/carbonyl
- carbohydrate or glycan
- sulfur, thiol, glutathione-like redox features
- small organic acid or metabolite-associated features

### 8.2 Evaluate those candidates on grounding and calibration datasets
For each candidate axis, test:
- specificity
- monotonicity under spike-ins
- behavior under degradation
- co-activation overlap with other axes
- robustness across contexts

### 8.3 Merge, prune, or stabilize
Axes that are too redundant, too unstable, too overlapping, or too weakly grounded should be merged or removed.

The final v1 operational axis set should therefore emerge from constrained evidence and validation, not from either pure intuition or pure unsupervised learning.

---

## 9. Grounding: the evidence base that defines GAIRA

GAIRA must be built from a grounding layer consisting of both spectral reference data and grounded literature evidence.

### 9.1 Grounding dataset types
These include:
- molecular and biochemical reference spectra
- RamanBioLib and similar reference collections
- amino acid reference datasets
- nucleic acid or nucleobase reference datasets
- metabolite and small-molecule reference datasets
- isotopic or other validation spectra where relevant
- other clearly attributable grounding spectra

### 9.2 Grounding literature types
These include only literature evidence relevant to biochemical grounding, such as:
- peak assignments for specific biochemical entities or classes
- band assignments linked to reference spectra
- careful SERS or Raman biochemical interpretation papers
- evidence describing context dependence, adsorption bias, or band ambiguity

Grounding literature should not be mixed with disease literature when constructing GAIRA-Base.

### 9.3 Existing evidence tables
Existing structured evidence tables should not be discarded.
They should be reused, audited, and upgraded.

The new grounding table should absorb the useful parts of the current evidence layer and reorganize them into a stricter spectral evidence schema.

---

## 10. Spectral primitives are the base unit, not motifs

The first-class unit in GAIRA-Base should be the **spectral primitive**.

Spectral primitives are structural descriptors extracted from spectra before assigning biochemical meaning.

Examples include:
- peak presence
- peak position
- peak prominence
- peak width
- regional band energy
- local envelope behavior
- co-occurrence of nearby bands
- absence of an expected companion band
- relative rank ordering of peaks
- broad versus narrow band behavior

### 10.1 Why primitives matter
Single peak assignment is too brittle. Biochemical evidence is usually pattern-based.

A primitive-based system allows GAIRA to:
- match real spectral structure more faithfully
- stay interpretable
- remain debuggable when calibration fails

### 10.2 What about motifs?
Motifs are useful, but they should not be the first-class representation in GAIRA-Base.

Motifs are higher-order recurring structures composed of multiple primitives.

Recommendation:
- In GAIRA-Base, primitives are primary.
- Motifs may later emerge from repeated primitive co-occurrence and become useful in interpretation layers.

Motifs should therefore be treated as a downstream construct unless and until they become empirically well-supported and calibration-stable.

---

## 11. The rebuilt grounding table

The grounding table is one of the most important artifacts in GAIRA.

It should not be a simple list of peak assignments.

Each row should represent a grounded evidence object.

### 11.1 Recommended conceptual fields
- evidence_id
- source_type (reference spectrum, literature assignment, validation spectrum, etc.)
- provenance
- molecule
- molecular class
- biochemical class
- candidate axis
- optional bond or chemistry annotation
- Raman/SERS modality context
- matrix context if known
- peak center
- allowed band range
- expected prominence
- co-occurring expected bands
- conflicting or exclusionary bands
- confidence tier
- notes on ambiguity
- primitive or feature type

### 11.2 Why this matters
This structure allows GAIRA to reason with richer evidence than:
- one peak = one molecule

Instead, it supports:
- pattern-level evidence
- context-sensitive matching
- explicit ambiguity handling
- interpretable scoring

---

## 12. GAIRA-Base scoring philosophy

The BSV scoring engine should be transparent, modular, and inspectable.

The broad conceptual flow is:

1. preprocess spectrum
2. extract spectral primitives
3. match primitives against grounded evidence objects
4. accumulate positive evidence for each candidate axis
5. apply penalties for conflicting evidence or absence of required support
6. normalize and estimate confidence
7. produce final BSV vector and explanation trail

Each axis score should have:
- numerical score
- confidence score
- supporting evidence list
- conflicting evidence list

This should be implemented as an explicit operator before any black-box learning layer is allowed to take center stage.

---

## 13. Why global embedding-first approaches are not the foundation

Past attempts using large shared encoders across heterogeneous datasets showed that models can easily learn:
- sample type
- dataset identity
- acquisition regime
- other nuisance structure

This is expected in Raman and SERS because cross-dataset measurement conditions vary substantially.

Therefore:
- global learned embeddings are not the right foundation for GAIRA-core concepts
- learned layers may still be useful later, but only after the biochemical state representation is validated

Learning should be layered on top of a grounded scaffold, not used as the primary definition of biochemical state.

---

## 14. Validation philosophy

GAIRA must earn trust through calibration.

Calibration is not a side exercise. It is the legitimacy test.

### 14.1 Validation categories
- spike-in experiments
- degradation experiments
- mixtures
- concentration series
- replicate stability
- matrix robustness where available

### 14.2 Metrics to care about
- monotonicity
- specificity
- coherence under mixing
- reproducibility
- directional agreement with known perturbation
- axis separability
- confidence calibration

The exact metric implementation can evolve, but the principle cannot.

---

## 15. Target dataset philosophy

When analyzing biological datasets, GAIRA should operate at **sample level first**.

Cohort averages can be useful for summary, but should not be the primary analysis object.

### 15.1 Required behavior
For each sample, compute:
- BSV vector
- confidence
- supporting evidence

Then cohort summaries should be built from sample distributions.

This allows:
- visualization of within-cohort variability
- identification of outliers and substructure
- more honest comparison across cohorts

### 15.2 Why this matters
Analyzing only average spectra hides biological heterogeneity and can create misleadingly clean stories.

---

## 16. Cross-dataset coherence strategy

Cross-dataset coherence should first be pursued in BSV space, not by forcing all spectra into a shared latent encoder.

The sequence should be:
1. define grounded axes
2. validate them on calibration data
3. map all datasets into the same biochemical state space
4. assess whether similar perturbations or biological states show similar directional shifts

If cross-dataset coherence emerges in BSV space, that is a stronger result than raw spectral similarity.

Only after this should one consider learned nuisance-invariant layers or residual models.

---

## 17. DART-Met integration philosophy

DART-Met converts static spectroscopy into dynamic response measurement.

GAIRA is the representation layer that should make those dynamic responses interpretable.

Instead of only plotting intensity versus wavenumber over time, GAIRA-Dynamic should make it possible to represent:
- BSV as a function of time
- BSV as a function of voltage or waveform
- biochemical trajectory velocity in state space
- reversible and irreversible state shifts
- differential trajectory behavior between healthy and disease samples

This allows DART-Met to be framed not as raw spectral fluctuation, but as dynamic biochemical state interrogation.

---

## 18. What literature is for and what it is not for

GAIRA uses literature in two very different roles.

### 18.1 Grounding literature
Used in GAIRA-Base.
Purpose:
- support biochemical peak and band assignments
- capture ambiguity and context
- supplement spectral reference datasets

### 18.2 Disease or condition literature
Used in GAIRA-Interpret.
Purpose:
- explain why shifts in BSV space may matter biologically
- connect biochemical state changes to disease hypotheses

Disease literature must not define the BSV axes.
That would leak downstream interpretation back into the core representation layer.

---

## 19. What should be built first

The build order should be strict.

### Phase 1 — GAIRA-Base
- define grounding schema
- extract spectral primitives
- define candidate axes
- build transparent scoring
- generate BSV vectors

### Phase 2 — GAIRA-Validate
- build calibration benchmark harness
- score monotonicity, specificity, mixture behavior, stability
- prune and refine axis definitions

### Phase 3 — GAIRA-Cohort
- map target datasets at sample level
- compare cohort distributions
- evaluate cross-dataset coherence in BSV space

### Phase 4 — GAIRA-Interpret
- add biological context
- add disease literature linkage
- create interpretable reports

### Phase 5 — GAIRA-Dynamic
- compute real-time biochemical trajectories for DART-Met
- analyze waveform-dependent state shifts

Do not invert this order.

---

## 20. Role of learning models later

Machine learning remains useful, but its role should be constrained.

Possible later roles include:
- nuisance-invariant representation learning conditioned on BSV consistency
- residual correction models layered on top of grounded scores
- cohort- or disease-level structure discovery in BSV space
- dynamic trajectory modeling for DART-Met

Machine learning should not replace the foundational biochemical operator before the grounding and calibration layers are working.

---

## 21. Criteria for accepting a new axis, rule, or evidence source

A new axis or rule should only be retained if it improves one or more of the following without substantially harming interpretability:
- calibration fidelity
- robustness
- cross-dataset coherence
- downstream usefulness

New evidence sources should be accepted only if:
- provenance is clear
- context is known enough to interpret
- they improve rather than muddy the biochemical state representation

More data is not automatically better data.

---

## 22. Main failure modes to watch for

### 22.1 Circularity
Disease literature leaking into axis construction.

### 22.2 Dataset overfitting
A state representation that works only because it implicitly encodes one dataset’s measurement quirks.

### 22.3 Over-broad axes
Axes so broad that everything activates everything.

### 22.4 Over-fragmented axes
Too many tiny axes with poor stability and redundancy.

### 22.5 Premature motif formalization
Turning half-supported recurring patterns into hard-coded first-class objects too early.

### 22.6 Overconfidence
Claiming molecule- or disease-level certainty from mixture spectra.

---

## 23. The paper-level framing

A strong future GAIRA paper should not be framed as a generic retrieval tool or a narrow classifier.

The stronger framing is:

> A grounded biochemical state representation framework for Raman and SERS that is validated on compositional perturbation datasets, transferable across biological datasets, and extensible to dynamic perturbation trajectories.

That framing is methodologically stronger, more general, and more aligned with the actual technical novelty.

---

## 24. The one-sentence memory aid

If future work becomes confused, return to this sentence:

> GAIRA exists to turn Raman and SERS spectra into a validated biochemical state space before any disease or dynamic interpretation layer is applied.

---

## 25. Immediate operating principle

Right now the priority is not more retrieval, more graphs, or a larger encoder.

The priority is to make the BSV layer real.

Everything else depends on that.

---

## 26. GAIRA as a delta-state measurement system

Phase 2.3 established experimentally that GAIRA's informative output, under real-data calibration, is the **change in BSV relative to a reference condition**, not the absolute BSV at a single condition.

The reason is structural rather than empirical. A BSV is an evidence-weighted state estimate over a set of atlas-backed axes. Its absolute magnitudes are conditioned on:
- the matrix (serum, aqueous, tissue, etc.)
- the substrate and acquisition regime
- normalization behavior (max-in-window is baseline-sensitive)
- whatever competing analytes happen to occupy the same atlas bands

An absolute BSV therefore answers the question *"what evidence is present in this spectrum"*, conditioned on all of the above. It does **not** directly answer the question *"how is this spectrum different from a reference"*, which is the question calibration and biology actually ask.

A ΔBSV — computed between a sample and a defined baseline acquired under matched conditions — cancels out the conditioning and exposes the state change:

- Matrix-saturation effects cancel (a baseline that has Tyr=1.0 and a treated condition that has Tyr=1.0 yield Δ=0, which is the correct interpretation).
- Normalization artifacts cancel when baseline and treated are processed identically.
- Atlas-gap effects cancel on the blocked axis (Δ=0) but remain visible on collateral axes.
- Threshold transitions (0 → committed) become categorical rather than continuous-under-noise.

### 26.1 Canonical statement

> GAIRA measures biochemical **state change**, not static biochemical identity. Its primary scientific output is the ΔBSV between paired conditions acquired under matched instrumental and matrix conditions. The absolute BSV is a supporting context object, not the primary measurement.

### 26.2 Implications

- Calibration design must pair conditions (baseline + perturbation) under matched acquisition. Unpaired single-condition interpretation is not the correct use of the system.
- Cross-dataset ΔBSV comparisons require that each dataset's ΔBSV be computed within that dataset's own baseline. Cross-dataset *absolute* BSV comparison is not valid.
- A "zero delta" is an informative output: it tells the consumer the state did not change on the measured axis. It is distinct from a "no signal" output (which no longer has a well-defined interpretation).

---

## 27. Primary output hierarchy

The output layer of GAIRA has a defined hierarchy. Downstream consumers, figures, and tables must privilege higher-tier outputs.

### 27.1 Primary outputs

All of the following are computed relative to a defined baseline condition:

- **Δ axis magnitude** — the evidence-weighted change in a committed axis's numerical state between paired conditions.
- **Δ commit frequency** — the change in the fraction of replicates in which a given axis crosses its commit threshold. Binary-like axes (see §28) are best expressed in this metric.
- **Δ radar geometry** — the change in multi-axis polygonal representation, summarized by Δ radar area when a scalar is needed and by per-axis Δ magnitudes when shape is the question.

### 27.2 Secondary outputs

- **Absolute BSV** — the per-axis magnitudes and commit flags for an individual condition.
- **Absolute modifiers** — the continuous modifier values (centroids, ratios) for an individual condition.
- **Raw radar area** — the scalar polygon area of a single-condition BSV.

Secondary outputs exist to support primary outputs. They are legitimate context, but they are not the measurement.

### 27.3 Derived interpretation

- **Dose-response class** — one of the classes defined in §28.
- **Threshold / LOD boundary** — the concentration at which Δ commit frequency or Δ magnitude first becomes reproducible.
- **Saturation point** — the concentration at which further increases produce no additional Δ in the primary axis.
- **Analyte recoverability class** — the A / B / C / D / E classification established in Phase 2.2 and updated in Phase 2.3, carried per-analyte per-matrix.

These are derivations over the primary outputs. They are how GAIRA communicates its calibration results to downstream scientific and clinical consumers.

### 27.4 Example hierarchy in practice

A calibration report on an analyte must lead with Δ commit frequency or Δ magnitude across the concentration range, qualified by pooled-SD. It may then display absolute BSVs as supporting panels, typically as baseline-vs-dose overlays, but these are not the primary scientific claim.

---

## 28. Calibration behavior classes

GAIRA permits and correctly represents several distinct analyte response behaviors. A single "dose-response shape" expectation is wrong; different chemistries and different matrices produce different valid shapes. The classes below are stable across Phase 2.1 – Phase 2.3 evidence and should be the lexicon for future calibration discussion.

### 28.1 Monotonic quantitative emergence (Ergo-type)

A previously-uncommitted axis crosses its commit threshold at a turn-on concentration, rises monotonically through a dynamic range, and saturates.

Diagnostic features:
- baseline: axis at magnitude zero, commit frequency zero
- LOD: first concentration with reproducible commit
- dynamic range: concentrations between LOD and saturation
- saturation: magnitude pegged at 1.0 on the primary axis

Appropriate primary metric: Δ magnitude (within dynamic range) + Δ commit frequency (at and near LOD).

### 28.2 Thresholded / binary emergence (Ade-type)

The axis commits at any concentration above a low detection floor, across many orders of magnitude, with 100% commit frequency. The magnitude itself does not carry usable concentration information because normalization dynamics make it non-monotonic.

Diagnostic features:
- baseline: axis at zero (often because of blank-veto)
- turn-on: at or below the lowest tested non-zero concentration
- magnitude: non-monotonic over concentration, often peaking in the middle of the range
- commit frequency: near 1.0 everywhere above the detection floor

Appropriate primary metric: Δ commit frequency. Magnitude is a context-only signal.

### 28.3 Saturation behavior

At sufficiently high concentrations, the primary axis pegs at 1.0 and further concentration changes do not move it. This is not a failure; it is a bound on the representation.

Above saturation, concentration information is lost on the primary axis. Recovering it requires:
- a companion axis that has not yet saturated (e.g., TDR remains informative above the Ergo-antioxST saturation point),
- or a continuous modifier that tracks peak position or amplitude,
- or a Phase 6.4+ per-band reference normalization that removes the global-max ceiling.

Calibration designers should state the saturation point explicitly and not extrapolate beyond it.

### 28.4 Matrix-dependent behavior

The same analyte can fall into different behavior classes in different matrices. Adenine is a textbook example:
- aqueous matrix → class A binary emergence across six orders of magnitude
- serum matrix → conservative no-commit at the same concentrations, because the serum baseline already saturates the atlas bands the analyte would populate

Matrix is therefore a first-class calibration variable. A recoverability claim without a matrix qualifier is incomplete.

### 28.5 Blocked by known atlas limitation

An analyte may be systematically under-committed because its dominant SERS peak is attributed to a non-primary atlas band. The canonical example is uric acid's 640 cm⁻¹ peak mapping to `band_620_700` (Tyr primary) rather than to a purinic-dedicated primary.

These cases are not pipeline failures; they are atlas-refinement items. They must be reported as class E (atlas-blocked), with a concrete Phase 6.4 fix path, and must not be force-interpreted as "no signal".

### 28.6 Class reference table

| class | shape | primary metric | example |
|---|---|---|---|
| A+ monotonic quantitative | sigmoidal with saturation | Δ magnitude + Δ commit | Ergo → antioxidant_sulfur_thione |
| A binary / thresholded | flip at very low conc | Δ commit frequency | Adenine (aqueous) → purine_nucleobase |
| A weak but valid | small significant Δ with no false activations | Δ magnitude (Δ/SD ≥ ~3) | Methio → thiol_disulfide_redox |
| saturated-baseline | axis pinned at 1.0 in baseline | no useful Δ on this axis | Tyr in UA-rich serum |
| overlap-masked | primary band contested by co-committed axis | ambiguous Δ | Gluc vs serum lipid at band_1080_1140 |
| atlas-blocked (E) | always zero on expected axis | Δ=0 pathway; report as blocked | UA → purinic_metabolite |

---

## 29. Phase 2.3 — first real calibration pass

Phase 2.3 is the first phase in which a real-data calibration produced an unconditional PASS. It is therefore promoted to canonical status and becomes a permanent reference point for subsequent phases.

### 29.1 What Phase 2.3 validated

- **Ergothioneine → antioxidant_sulfur_thione**: monotonic-saturating sigmoidal dose-response across 11 concentrations with five replicates each. LOD at 0.4 µM, dynamic range 0.4 – 1.4 µM, saturation at ≥ 1.4 µM. This is the first quantitatively calibrated GAIRA axis.
- **Adenine → purine_nucleobase (aqueous matrix)**: binary detection across six orders of magnitude (74 pM to 10 µM), with 100% commit frequency at every non-blank concentration and a correctly vetoed blank substrate.
- **Methionine → thiol_disulfide_redox** (carried from Phase 2.2): weak but statistically clean axis response, Δ / pooled_SD ≈ 4.9, R_C1 centroid-aware refinement holding under real data.

Together these form the three canonical calibration anchors on real data.

### 29.2 What Phase 2.3 established about GAIRA as a system

- The scorer can produce **quantitative dose-response behavior**, not merely discrimination-grade activation.
- The system correctly **distinguishes two response classes** (monotonic emergence vs binary detection) within the same pipeline, with no architectural change.
- **Matrix-dependent recoverability** is a real phenomenon and must be treated as a first-class variable in calibration design.
- **Modifiers are chemistry-identity validators** in the current atlas, not continuous quantitative signals. This bounds what they can be used for.

### 29.3 What Phase 2.3 leaves open

- Atlas-level refinement items (primarily the UA / purinic class) remain open for Phase 6.4.
- Modifier-layer refinement to provide continuous-amplitude modifiers is Phase 6.5 scope.
- Per-band reference normalization, to remove the max-in-window constraint on magnitude at high concentrations, is Phase 6.4+ scope.

### 29.4 Canonical claim

> Phase 2.3 establishes GAIRA as a **biochemical state-space sensing system** validated on real dose-response data, with quantitative calibration on at least one axis (antioxidant_sulfur_thione) and binary detection across six orders of magnitude on at least one axis (purine_nucleobase, aqueous).

This claim is the first real-data statement GAIRA is entitled to make. It should be treated as durable, subject only to explicit Phase 6.4 / Phase 6.5 updates when those land.

---

## 30. ΔBSV as canon

The canonical rule:

> All future validation, calibration, and cohort analyses in GAIRA MUST use ΔBSV as the primary evaluation object. Absolute BSV may be reported as supporting context, but it must not be the primary scientific claim.

### 30.1 Implications for each downstream scope

- **Phase 2.4 (isotopic + mixture)**: directionality claims are ΔBSV claims. The isotopic contrast (analyte vs analyte-isotope under identical acquisition) is a textbook ΔBSV use case.
- **Phase 2.5 (cross-calibration synthesis)**: the synthesis across analytes must be framed in ΔBSV per-analyte per-matrix. An "absolute BSV" comparison across analytes is not meaningful.
- **Phase 3 (cohort studies)**: cohort-level discrimination must be framed in terms of ΔBSV relative to a defined reference condition within the cohort design (e.g., healthy control, time-zero, pre-treatment). Absolute BSV per cohort is supporting.
- **Phase 6.4 / 6.5 atlas and modifier refinement**: regression tests for refinement work must be framed as ΔBSV comparisons against prior-phase results.

### 30.2 What this does NOT mean

- It does not mean absolute BSVs are invalid. They are valid supporting context.
- It does not mean every figure must display only deltas. Baseline-vs-condition overlays are legitimate and often necessary.
- It does not mean single-condition interpretation is forbidden; it means single-condition interpretation must be labeled as such and not framed as a calibration claim.

### 30.3 Operational test

A downstream analysis is compliant with this rule if, on inspection, the first scientific claim it makes on an analyte or cohort is a delta claim against a defined baseline. If the first claim is an absolute claim, the analysis should be rewritten.

### 30.4 Canonical statement for memory

> GAIRA reports change, not identity. The primary object is ΔBSV. The baseline is always declared.

