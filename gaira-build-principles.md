# Build Principles

## Purpose
This document defines the immediate operating principles for building GAIRA-Base.

## Principles
1. Determinism first
   - Identical inputs and configurations must yield identical outputs.

2. Disease-agnostic core
   - GAIRA-Base must not encode disease logic, cohort rules, or target-dataset-specific labels.

3. Primitives before motifs
   - Spectral primitives are first-class.
   - Motifs are optional later derived summaries, not initial scoring anchors.

4. Calibration is the legitimacy test
   - GAIRA-Base is only meaningful if it behaves correctly on spike-ins, degradation, mixtures, concentration series, and replicates.

5. Provenance must be preserved
   - Every evidence object must remain traceable to its source dataset or literature reference.

6. Transparent scoring over opaque latent logic
   - The initial biochemical state engine must use explicit, inspectable scoring rules.

7. Overcomplete first, prune later
   - Candidate biochemical axes may begin broad and somewhat redundant, but must later be merged or removed based on calibration behavior.

8. Human-auditable knowledge layer
   - Markdown vault pages are allowed as supporting evidence summaries, but machine scoring must depend on structured schemas and config.

## Immediate priority
Make the BSV layer real before adding disease interpretation, dynamic demos, or new learned encoders.