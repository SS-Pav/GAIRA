# BSV v2 — Component Revision Notes

## Components Reviewed

### protein_backbone → KEPT (no rename needed)
This component maps to collagen, proline, glycine + protein/peptide themes. The "backbone" in the name refers to protein backbone vibrations (amide bands) which ARE the primary evidence source. Name is scientifically accurate.

### nucleic_acid_backbone → KEPT (no rename needed)
Maps to phosphodiester and phosphate functional groups — these ARE the nucleic acid backbone. Name is accurate. The 0.8x weight is appropriate since it's chemistry-only (no specific base identity).

### redox_metabolite → KEPT (no rename needed)
Maps to glutathione (thiol antioxidant), ergothioneine (histidine-derived antioxidant), uric acid, carotenoids, dopamine. "Redox metabolite" accurately describes this heterogeneous group of small-molecule redox-active compounds. Alternative "redox_thiol_metabolite" was considered but rejected as too narrow (not all components are thiols).

## Display Names Updated
Component names are now displayed with improved line-breaking labels in the radar plot for readability:
- "Membrane\nLipid" instead of "Membrane Lipid"
- "Aromatic\nAmino Acid" instead of flat text
- "Glycan /\nCarbohydrate" for the saccharide component

## No Components Removed or Added
The original 8 components remain scientifically appropriate. No changes to the component map itself.
