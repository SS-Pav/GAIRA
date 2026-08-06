# `GAIRA_v7_rebuild/reports/`

Phase reports and synthesis documents. **Currently empty — no phase has run.**

## What belongs here

- `PHASE_00_REPORT.md` … `PHASE_09_REPORT.md`
- the unified V7 Raman validation report (Phase 07)
- the CSM reference manual (Phase 03)
- the replacement recommendation (Phase 07)

## What must not be stored here

- **Planning documents.** Those live in `../plan/`. This directory holds *results*.
- **Architecture specifications.** Those live in `../architecture/`.
- **Raw tables and figures.** Those live in `../results/`; reports reference them.
- **PDFs.** Repo policy gitignores `*.pdf`; the Markdown source is tracked.

## Every phase report must contain

| Section | Content |
|---|---|
| Inputs | artefact IDs and hashes |
| Configuration | every parameter, every seed |
| Method | what was run, and the code entry point |
| Decisions | each with the document + section that pre-registered its rule |
| Results | tables and figures, with pointers into `../results/` |
| Gate results | each gate, pass/fail, with evidence |
| Limitations | what this phase does not establish |
| Reproduction | the exact command to re-run it |

## Reporting discipline

**Report what happened, including when it is inconvenient.** Three specific obligations:

1. **Negative results are reported at full length.** If Phase 01's control arm wins, or if the
   theme layer adds nothing over the CSM layer, that is the finding and it is written up as
   one. The V6.3 revalidation is the precedent — it established that ontology cleanup was not
   the fix, which is precisely why V7 exists.
2. **Semantic degradations get equal prominence with semantic rescues.** A phase report that
   lists what V7 fixed without listing what it broke is not a report.
3. **Point estimates are not results.** Every headline number carries a bootstrap CI, and
   every comparison carries a significance test and an effect size.
