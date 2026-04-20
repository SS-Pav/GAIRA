# GAIRA-Base

## 0. Purpose of this document

This document defines the implementation-facing specification for GAIRA-Base.

GAIRA-Base is the disease-agnostic biochemical state engine inside GAIRA.

Its job is to take a Raman or SERS spectrum and return a grounded biochemical support vector (BSV) along with interpretable evidence describing why each axis is activated or suppressed.

This document should be used directly in future Claude prompts when building or modifying GAIRA-Base.

---

## 1. Mission

GAIRA-Base exists to answer the following question:

> Given a Raman or SERS spectrum, what biochemical state does the spectral evidence support, using only grounding spectral references and grounding literature evidence?

The output is not a disease label.
The output is not a latent embedding.
The output is a transparent biochemical state representation.

---

## 2. Scope

### In scope
- preprocessing of spectra
- extraction of spectral primitives
- grounding evidence ingestion and normalization
- candidate axis scoring
- BSV generation
- axis confidence estimation
- support and conflict reporting
- radar geometry generation
- calibration-ready outputs

### Out of scope
- disease classification
- cohort-level statistical inference beyond direct BSV summaries
- literature retrieval for disease interpretation
- black-box global embedding training
- target-dataset-specific disease rules
- dynamic control logic for DART-Met

Those belong to later layers.

---

## 3. Required outputs for every spectrum

For every processed spectrum, GAIRA-Base should output at minimum:
- spectrum identifier
- preprocessing metadata
- spectral primitive table
- BSV vector
- per-axis confidence
- top supporting evidence entries per axis
- top conflicting evidence entries per axis
- global interpretability score
- optional radar geometry payload

This output should be deterministic for a fixed preprocessing and scoring configuration.

---

## 4. Core pipeline

The GAIRA-Base pipeline should follow this order:

1. ingest spectrum
2. standardize preprocessing
3. extract spectral primitives
4. read and normalize grounding evidence
5. score each candidate biochemical axis
6. estimate confidence and conflicts
7. generate BSV output
8. expose artifacts for calibration and visualization

The system should remain modular so that each stage can be audited independently.

---

## 5. Preprocessing requirements

Preprocessing must be deterministic, documented, and configurable.

### 5.1 Required preprocessing stages
- spectral cropping to the chosen Raman window or fingerprint region
- baseline correction
- smoothing if justified
- normalization

### 5.2 Design requirement
GAIRA-Base should preserve both:
- the fully preprocessed spectrum used for scoring
- enough metadata to reproduce that preprocessing exactly

### 5.3 Why this matters
Without deterministic preprocessing, later disagreements about axis behavior become impossible to debug.

---

## 6. Spectral primitive extraction

Spectral primitives are the base structural units in GAIRA-Base.

They should be extracted before biochemical meaning is assigned.

### 6.1 Primitive examples
- peak location
- peak height or prominence
- peak width
- local band area
- local envelope shape
- shoulder presence
- peak pair or band co-occurrence
- regional energy concentration
- relative ordering among major peaks
- absence of an expected companion feature

### 6.2 Primitive design principles
- primitives should be general enough to be reused across axes
- primitives should be measurable directly from spectra
- primitives should not themselves encode disease meaning

### 6.3 Output requirement
The primitive extractor should produce a machine-readable table for every spectrum.

That table should be inspectable and debuggable.

---

## 7. Grounding evidence layer

GAIRA-Base must use a rebuilt grounding evidence table as one of its central inputs.

This evidence layer should incorporate:
- spectral grounding datasets
- literature grounding evidence
- useful fields from the existing GAIRA evidence tables

### 7.1 Grounding data classes
These include:
- reference spectra from RamanBioLib or equivalent
- amino acid reference datasets
- nucleic acid and nucleobase reference datasets
- other well-defined biochemical reference spectra
- validation spectra where attribution is sufficiently clear

### 7.2 Grounding literature classes
These include only literature relevant to biochemical grounding:
- peak assignment papers
- Raman or SERS biochemical reference studies
- papers discussing assignment ambiguity or context dependence

### 7.3 Non-goal
Disease or condition literature should not be used to define GAIRA-Base axes.

---

## 8. Recommended grounding table structure

The grounding table should be richer than a peak assignment spreadsheet.

Each row should represent a grounded evidence object.

### 8.1 Recommended fields
- evidence_id
- source_type
- source_name
- provenance citation or identifier
- molecule
- molecular family
- biochemical class
- candidate axis
- optional bond or chemistry annotation
- measurement modality context
- sample or matrix context
- peak center
- band lower bound
- band upper bound
- expected prominence
- expected co-bands
- optional exclusions or conflicting bands
- primitive type represented
- ambiguity notes
- confidence tier
- free-text notes

### 8.2 Design requirement
The table should be able to express both:
- narrow local features
- broader band-pattern evidence

### 8.3 Use existing work
The current structured evidence tables should be reused and normalized into this stronger schema, not thrown away.

---

## 9. Candidate axis strategy

GAIRA-Base should begin with an overcomplete set of candidate biochemical axes rather than a prematurely collapsed final list.

### 9.1 Why
This supports chemically constrained discovery.
It allows the system to learn which axes are stable, redundant, or too noisy.

### 9.2 Suggested initial candidate pool
The exact list can evolve, but a good starting pool includes fine-grained axes such as:
- nucleic acid phosphate-associated
- purine or nucleobase-associated
- protein backbone or peptide-associated
- aromatic amino acid-associated
- aliphatic lipid-associated
- unsaturated or oxidized lipid-associated
- carbohydrate or glycan-associated
- sulfur, thiol, or redox-metabolite-associated
- small organic acid or metabolite-associated if justified by coverage

### 9.3 Required later step
After calibration, these can be:
- merged
- pruned
- stabilized into the operational v1 axis set

---

## 10. Axis scoring philosophy

Each axis score should quantify biochemical support from spectral evidence.

Axis scoring must be explicit and inspectable.

### 10.1 Conceptual axis-scoring flow
For each candidate axis:
1. identify matching primitives in the observed spectrum
2. retrieve compatible evidence objects from the grounding table
3. accumulate positive support
4. register missing expected support where appropriate
5. penalize conflicts or mutually inconsistent evidence
6. normalize to an interpretable scale
7. estimate confidence

### 10.2 Positive evidence examples
- strong primitive match near expected band
- correct co-band structure
- correct regional support
- internal consistency across multiple evidence rows

### 10.3 Conflict examples
- activation driven by one weak band while required companion bands are absent
- stronger evidence for a competing axis in the same region
- unsupported broad inflation from noisy background structure

### 10.4 Output requirement
Each axis must expose:
- score
- confidence
- support list
- conflict list

---

## 11. Confidence

GAIRA-Base should never output only a score.
It must also estimate whether that score is trustworthy.

### 11.1 Confidence should reflect
- amount of supporting evidence
- agreement among evidence objects
- specificity of matched primitives
- signal quality or interpretability of the spectrum
- degree of conflict with other evidence

### 11.2 Why confidence matters
Some spectra may be too weak, too noisy, or too ambiguous for stable biochemical decomposition.
GAIRA-Base should be able to say so.

---

## 12. Global interpretability score

GAIRA-Base should estimate whether a given spectrum is suitable for meaningful biochemical interpretation.

This is separate from per-axis confidence.

It can reflect:
- spectral quality
- number of usable primitives
- conflict burden across axes
- instability of scoring under nearby matching choices

This helps prevent over-interpretation of poor spectra.

---

## 13. Radar geometry

The radar plot is a visualization of the BSV vector, not the definition of the representation.

### 13.1 Principle
The radar is useful only if the BSV vector itself is valid.

### 13.2 Desired behavior
- controlled perturbations should deform the radar in expected directions
- mixtures should produce coherent intermediate shapes
- sample-level radars should reveal variability
- cohort summaries should be derived from sample-level BSVs, not just mean spectra

### 13.3 Implementation note
The geometry payload should be easy to compute from the BSV vector and confidence estimates.

---

## 14. Motifs

Motifs should not be hard-coded as first-class scoring objects in early GAIRA-Base.

### 14.1 Recommendation
Use spectral primitives as the direct evidence layer first.

### 14.2 When motifs become useful
Motifs may become useful later when:
- repeated primitive constellations are observed
- they improve calibration behavior
- they serve disease or condition interpretation

### 14.3 Practical rule
In early GAIRA-Base:
- primitives are primary
- motifs are optional derived summaries, not mandatory scoring anchors

---

## 15. Calibration readiness

GAIRA-Base is not complete unless it can be tested rigorously using calibration data.

### 15.1 Required calibration behaviors
The system should support evaluation on:
- spike-ins
- degradation experiments
- concentration series
- mixtures
- replicate sets

### 15.2 Desired evaluation metrics
- monotonicity
- directional correctness
- specificity
- mixture coherence
- replicate stability
- confidence calibration

These metrics can live in GAIRA-Validate, but GAIRA-Base must expose the necessary outputs.

---

## 16. Sample-level outputs for target datasets

GAIRA-Base should operate on every spectrum or sample individually.

### 16.1 Why
This preserves biological variability and avoids overreliance on averaged spectra.

### 16.2 Required output behavior
For target datasets, GAIRA-Base should generate BSVs per sample, after which later layers can perform:
- cohort aggregation
- density estimation
- distribution comparisons
- subgroup analysis

---

## 17. Cross-dataset strategy

GAIRA-Base should not assume that raw spectral similarity will transfer across datasets.

Instead, its role is to provide a common biochemical state representation that is more comparable across datasets than raw spectral space.

A later learned layer may refine this, but GAIRA-Base must stand on its own first.

---

## 18. Dynamic extensibility toward DART-Met

GAIRA-Base should be designed so that it can later be called frame-by-frame on dynamic spectra.

That means the core scoring pipeline should eventually support repeated inference over:
- time
- voltage
- waveform phase
- perturbation state

The future GAIRA-Dynamic layer can then consume these sequential BSV outputs.

---

## 19. Implementation principles

### 19.1 Determinism first
Identical inputs and settings should yield identical outputs.

### 19.2 Interpretability first
If a scoring rule cannot be explained, it should not define the representation.

### 19.3 Modular design
Preprocessing, primitive extraction, grounding ingestion, scoring, and visualization should be separable.

### 19.4 Preserve provenance
Every evidence object should remain traceable to its source.

### 19.5 Avoid hidden leakage
No disease labels or downstream biological conclusions should leak into core axis construction.

---

## 20. Immediate build sequence for GAIRA-Base

1. audit current grounding datasets and evidence tables
2. define the upgraded grounding-table schema
3. build the spectral primitive extractor
4. normalize grounding evidence into the new schema
5. define overcomplete candidate axes
6. implement explicit axis-scoring logic
7. generate BSV outputs and radar payloads
8. test on grounding spectra
9. test on calibration datasets
10. prune or merge candidate axes into the stabilized operational set

This is the correct order.

---

## 21. Main failure modes to guard against

### 21.1 Peak-list thinking only
Relying on single-peak assignments rather than structural evidence patterns.

### 21.2 Overlapping axes without disambiguation
Allowing the same evidence to inflate many axes simultaneously.

### 21.3 Carrying over noisy legacy fields unchanged
Reusing existing tables without cleaning or restructuring them.

### 21.4 Premature motif formalization
Turning suggestive multi-peak patterns into rigid logic too early.

### 21.5 Using target-dataset behavior to redefine biochemical grounding
This creates circularity and weakens transferability.

---

## 22. Definition of done for GAIRA-Base

GAIRA-Base can be considered scientifically meaningful only when all of the following are true:
- grounded reference spectra map sensibly into BSV space
- candidate axes are supported by real evidence and not just naming convenience
- calibration perturbations produce expected directional behavior
- support and conflict evidence can be inspected for each axis
- poor-quality spectra can be flagged as low-confidence
- sample-level BSV outputs are available for later cohort analysis

Until then, GAIRA-Base remains under construction.

---

## 23. Prompt-use instruction for future Claude sessions

When using Claude to build or modify GAIRA-Base, instruct it explicitly:

- use this document as the governing implementation spec
- do not introduce disease logic into GAIRA-Base
- preserve determinism and provenance
- prefer transparent scoring over opaque latent modeling
- treat spectral primitives as first-class and motifs as optional derived structures
- keep outputs calibration-ready

This will reduce architectural drift during development.
