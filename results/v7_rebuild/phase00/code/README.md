# Phase 00 — code

Deterministic implementation of the GAIRA V7 benchmark lock and canonical data foundation.

## Run

```bash
export GAIRA_DATA_ROOT=/path/to/GAIRA_DATA/raw     # optional — degraded mode without it
python results/v7_rebuild/phase00/code/run_phase00.py        # pipeline → tables + manifests
python results/v7_rebuild/phase00/code/validate_phase00.py   # PASS/WARN/FAIL suite
python results/v7_rebuild/phase00/code/make_figures.py       # 8 figures, SVG + PNG
pytest tests/test_v7_phase00.py
```

Data-root precedence: `--data-root` > `$GAIRA_DATA_ROOT` > `$GAIRA_DEFAULT_DATA_ROOT` >
degraded mode. No lab-specific path is committed anywhere in this tree.

## Modules

| Module | Responsibility |
|---|---|
| `v7_paths.py` | repo anchors, data-root resolution, canonical hashing, git + environment capture |
| `v7_corpus.py` | loads the Raman grounding corpus reproducibly; verifies it against the frozen V5 card; both replicate-group keys |
| `v7_canonical.py` | canonical molecule identity — NFKC, declared merges, protected pairs, flagged near-misses, leakage report |
| `v7_partition.py` | the frozen chemical partition (V6.3 fine/broad), class-conflict detection, class census |
| `v7_quality.py` | the frozen quality score `q` (`v7_q_v2`) and analyte-balanced weights |
| `v7_splits.py` | frozen analyte-grouped CV folds and the three leakage checks |
| `v7_harness.py` | the **frozen evaluation harness** — metrics, size-matched random nulls, bootstrap CIs, McNemar, ECE |
| `v7_benchmark.py` | three-level benchmark lock and the frozen dependency graph |
| `run_phase00.py` | orchestrator: writes every table, manifest and `PHASE_STATE.json` |
| `validate_phase00.py` | independent validation — re-reads the written artefacts rather than trusting in-memory values |
| `make_figures.py` | 8 publication figures from the tables only |

## Guarantees

- **Deterministic.** Fixed seeds throughout; all 16 tables are byte-identical across
  consecutive full runs. Only `built_utc` in the manifest differs between runs, by design.
- **Read-only on frozen assets.** Nothing here writes to `assets/`, `results/v5_rebuild/`,
  `results/v6_rebuild/` or `src/gaira/`. Enforced by
  `tests/test_v7_phase00.py::test_phase00_writes_only_inside_its_own_tree`.
- **Provenance.** Every input and output carries a SHA-256 in `phase_00_manifest_v1.json`,
  alongside seeds, config, git SHA and the resolved environment.
- **Benchmark lock level 3.** The NMF basis is refitted from raw and compared element-wise to
  the frozen atlas: `max |H_rebuilt − H_frozen| = 0.0`.

## What must not be added here

- Phase 01+ work — balanced references, LSMs, CSMs, themes, BSV.
- Any write path outside `results/v7_rebuild/phase00/`.
- Absolute local paths, or a default that assumes a mounted volume.
- Raw spectra, or any committed model file.
