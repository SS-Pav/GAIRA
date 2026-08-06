# GAIRA V7 — Git and Versioning Plan

---

## 1. Branch

| Field | Value |
|---|---|
| Branch | `gaira-v7-rebuild` |
| Base | `gaira-v5-rebuild-plan` @ `ddbb3945d670eee58f5ad99f868fb3c36b2a2c06` |
| Remote | `origin` → `https://github.com/SS-Pav/GAIRA.git` |
| Integration target | `main` (when V7 is adopted, or when the negative result is final) |

**No force push. No history rewrite. No reset of shared commits.**

Long-lived phase work may use short-lived sub-branches
(`gaira-v7-phase03-integration`) merged back into `gaira-v7-rebuild`, but the phase branch is
where the committed record lives.

---

## 2. Commit sequence

Commits follow the phase layers. The expected sequence:

```
docs(v7): establish rebuild context and phased architecture          ← this commit
feat(v7-phase0): lock benchmark and canonical identities
analysis(v7-phase1): compare balanced reference constructions
feat(v7-phase2): build class-specific local spectral motifs
analysis(v7-phase2): validate LSM stability and class coverage
feat(v7-phase3): derive consensus spectral motifs
analysis(v7-phase3): evaluate CSM count and motif communities
feat(v7-phase4): derive soft biochemical themes
feat(v7-phase5): define absolute biochemical state vector
feat(v7-phase6): integrate canonical V7 inference engine
analysis(v7-phase7): validate full in-domain Raman corpus
feat(v7-phase8): add chemistry-aware representation learning
docs(v7): finalize V7 foundation reports and reproducibility
```

### Prefixes

| Prefix | Use |
|---|---|
| `docs(v7)` | context, plans, architecture, reports with no new computation |
| `feat(v7-phaseN)` | new code or new frozen artefacts |
| `analysis(v7-phaseN)` | evaluations, sweeps, comparison tables, figures |
| `test(v7)` | test additions |
| `fix(v7-phaseN)` | corrections to V7 code or artefacts |

---

## 3. Commit rules

1. **One scientific change per commit.** A commit that changes the LSM stability threshold
   *and* the CSM count is two commits. Bisecting a scientific regression is only possible if
   commits are scientifically atomic.
2. **Every phase report is committed with the code that generated it.** A report whose
   generating code is in a later commit cannot be reproduced from its own commit.
3. **Every generated asset ships with its manifest.** An artefact with no manifest has no
   provenance and is not evidence.
4. **No frozen-asset modification without an explicit version bump.** Never edit a frozen
   atlas in place — create a new version directory.
5. **No raw spectra in Git.** `data/`, `GAIRA_DATA/`, `/Volumes/`, `*.mat` are gitignored.
6. **No local absolute paths** in code, config, docs, or manifests.
7. **No `SSD_Rad` references as defaults.** Use `GAIRA_DATA_ROOT`.
8. **Clean working tree for phase commits.** A manifest with `code.dirty: true` invalidates
   the phase — the code that produced it cannot be recovered.
9. **No force push.**

---

## 4. Data-root policy

Resolution order, inherited from `tools/reproduce_gaira_foundation.py`:

```
--data-root  >  $GAIRA_DATA_ROOT  >  optional documented default  >  error
```

The default is `None` in committed code, so no lab-specific path is ever committed. Frozen
inference must run with `GAIRA_DATA_ROOT` unset — the atlas bundle is self-contained. This is
a Phase-06 gate and is checked by the scaffold test.

---

## 5. What is tracked

| Tracked | Not tracked |
|---|---|
| all V7 documents (`.md`) | raw spectra |
| frozen atlas bundles (`.npz`, `.json`, `.yaml`, `.csv`) | large sweep intermediates |
| phase manifests | per-run NMF outputs |
| tables (`.csv`) | `__pycache__`, `.venv` |
| figures (`.svg`, `.png`) | `*.pdf` (repo policy) |
| code | secrets, `.env` |
| tests | machine-specific files |

### Two repo-policy interactions worth knowing

**`*.pdf` is gitignored.** The root `.gitignore` tracks the Markdown source instead. V7
planning figures therefore ship as **SVG (vector) + PNG (preview)** rather than PDF. This is a
deliberate conformance choice, not an omission — SVG is a vector format and satisfies the
vector requirement.

**`checkpoints/` is gitignored globally.** A scoped `GAIRA_v7_rebuild/.gitignore` re-includes
`GAIRA_v7_rebuild/results/checkpoints/` so V7 atlas bundles can be tracked, while bulk
intermediates remain excluded by extension. The negation lives entirely inside
`GAIRA_v7_rebuild/`, so the root `.gitignore` is untouched.

---

## 6. Atlas versioning

`v7.MAJOR.MINOR.PATCH`, per `../architecture/ARTIFACT_AND_MANIFEST_SPEC.md`:

| Change | Bump | Fingerprint |
|---|---|---|
| any behaviour-determining layer (preprocessing spec, CSM basis, LSM dictionary, `S`, theme registry, BSV reference, OOD support) | MAJOR | changes |
| registry metadata, provenance, README, visualisation transform | MINOR | unchanged |
| non-behavioural typo | PATCH | unchanged |

Every atlas version gets its own directory under `results/checkpoints/`. Migration notes
accompany every fingerprint change, stating which layer changed and what the downstream
effect is.

---

## 7. Protected paths

These must not appear in any V7 commit's `git diff --name-only`:

```
assets/foundation/**
results/v5_rebuild/**
results/v6_rebuild/**
src/gaira/engine/**
src/gaira/preprocessing/**
streamlit_apps/**
gaira_foundation_explorer*/**
gaira_semantic_explorer*/**
tools/reproduce_gaira_foundation.py
```

Permitted V7 paths:

```
GAIRA_v7_rebuild/**
tests/test_v7_rebuild_scaffold.py
```

and, if genuinely necessary, a single minimal navigation link in the top-level `README.md` —
nothing more.

**Enforcement.** `tests/test_v7_rebuild_scaffold.py` verifies that the frozen atlas
fingerprint is unchanged and that no V7 document contains hard-coded absolute paths. A stricter
protected-path check belongs in CI when V7 implementation begins.

---

## 8. Pre-commit checklist

Before any V7 commit:

- [ ] `git status` shows only V7 paths
- [ ] `git diff --name-only` touches no protected path
- [ ] Atlas fingerprint `09ed804a40836f4a05a91ba10900cded` unchanged
- [ ] No absolute local paths in any changed file
- [ ] No raw spectra staged
- [ ] Every new artefact has a manifest
- [ ] Every new report names the code that produced it
- [ ] `pytest tests/test_v7_rebuild_scaffold.py` passes
- [ ] Commit message follows the prefix convention
- [ ] Working tree clean for phase commits (`code.dirty: false`)

---

## 9. Merge to `main`

V7 merges to `main` only after **either**:

- Phase 07 delivers a passing evaluation and the replacement recommendation is accepted; **or**
- Phase 07 delivers a documented negative result and the V5 atlas is formally retained.

In both cases the full evidence trail — every phase report, table, and figure — merges with
it. A negative result is committed with the same care as a positive one; the V6.3 revalidation
is the precedent, and it is the reason V7 exists at all.
