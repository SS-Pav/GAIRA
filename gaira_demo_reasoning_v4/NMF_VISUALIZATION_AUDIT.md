# NMF Visualization Audit (Part 1)

## Why PCA was demoted

GAIRA inference **is** the frozen NMF decomposition (query → 24 non-negative component
activations, held-fixed dictionary). A PCA of reference spectra is a *different* linear
basis that the engine never uses, so leading the Reference Atlas with PCA (a) implied
the wrong representation and (b) made weak family separation look like an engine
failure when it is just a property of PCA on overlapping mixtures. PCA is retained only
as a clearly-labelled **exploratory, secondary** view ("not used for inference").

## NMF-native views implemented (now primary)

1. **NMF component atlas** (Section C, already present, now the primary explorer): all
   24 components with basis spectrum, dominant bands, top reference analytes + family,
   linked MSS motifs, many-to-many component→theme weights, stability, purity,
   confidence tier, and the c3 educational case.
2. **Component similarity map** (Section B): classical (Torgerson) MDS of the 24
   components on the **cosine distance between basis spectra**, coloured by dominant
   theme, every node directly labelled. Deterministic and distance-preserving.
3. **Component hierarchy / dendrogram** (Section B): average-linkage hierarchical
   clustering of the same distances, annotated by dominant theme.
4. **Component → MSS → theme network** (Section D): the existing Sankey, with edge
   widths from the actual registry weights; explicitly many-to-many (no one-to-one).

## Component distance definition

- **Primary distance**: `1 − cosine(basis_i, basis_j)` between the two frozen NMF basis
  spectra (24×24). Symmetric, zero diagonal, range [0, 0.99], verified deterministic.
- A single, clearly-defined distance is used (not an arbitrary average of incompatible
  measures). Shared reference-loading / shared-MSS similarity are surfaced separately in
  the component explorer and the Sankey rather than blended into one number.
- 2-D embedding by **classical MDS** (eigendecomposition of the double-centred squared
  distance) — chosen over UMAP because there are only 24 points and a deterministic,
  distance-preserving method is preferable; sign-fixed for reproducibility.

## Limitations

- The MDS map is a 2-D projection of a 24×24 distance — read clusters and neighbours,
  not exact coordinates.
- Basis-spectrum cosine distance captures spectral-shape similarity; it does not by
  itself encode perturbation or reference-loading relationships (those are shown in the
  component explorer and the Sankey).
- The dendrogram's linkage (average) is one reasonable choice; different linkages give
  slightly different groupings, so it is presented as similarity structure, not a
  definitive taxonomy.
