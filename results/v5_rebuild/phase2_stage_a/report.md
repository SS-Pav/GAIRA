# Phase 2 Stage A — notebook summary

Reproduce:
```bash
python results/v5_rebuild/phase2_stage_a/code/run_input_audit.py   # §4 audit → manifest
python results/v5_rebuild/phase2_stage_a/code/run_stage_a.py       # analyses → scorecard/decision
python -m pytest tests/test_v5_representation.py -q                # 16 tests
```

**Input:** 479 spectra (214 Raman + 265 Ag-SERS), 87 analytes, 51 matched. Adenine
concentration series excluded (controlled perturbation). All 9 audit invariants pass.

**Headline result:**
| preprocessing | joint ARI(analyte) | joint ARI(nuisance) | modality leak (bal-acc) | cross-modal top-1 | perm p |
| --- | --- | --- | --- | --- | --- |
| A1 L2 | 0.02 | 0.26 (source) | 0.94 | 0.08 | 0.021 |
| **A2 SNV** | **0.15** | **0.02** | **0.83** | **0.16** | **<0.001** |
| A3 deriv | 0.15 | 0.02 | 0.93 | 0.04 | 0.262 |

Raman-only ARI(analyte) = 0.49–0.52 (chemistry recovered within modality).
Peak overlap matched ≈ mismatched (cross-modal signal is not from shared peak positions).

**Decision: Outcome B — modality-stratified representation defensible.**
Shared space rejected (modality bal-acc 0.83 > 0.75; top-1 0.16 < 0.30); direct
representation adequate within a modality; weak-but-significant residual cross-modal
signal → recommend Stage B chemical features. Full write-up:
`GAIRA_V5_PHASE2_STAGE_A_DIRECT_REPRESENTATION_REPORT.md`.

**STOP after Stage A.** No Stage B/C, observation model, ontology, BSV, MSS, or
perturbation evaluation was started.
