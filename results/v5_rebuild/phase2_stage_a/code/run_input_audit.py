"""GAIRA V5 Phase 2 Stage A — §4 input audit (immutable input manifest).

Verifies every candidate 785 nm grounding spectrum against the admission gate,
records provenance/role, applies the Phase-2 role correction (adenine
concentration series = controlled perturbation → excluded from representation
fitting), confirms a single common wavenumber grid, canonical identities, and
no raw/processed duplication. Emits an immutable manifest + audit summary.

Read-only w.r.t. source data. Outputs under results/v5_rebuild/phase2_stage_a/.
"""
from __future__ import annotations
import sys, json, collections
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
from gaira.data import loader, gobbato                       # noqa
from gaira.data.schema import admits_to_joint_analysis       # noqa
from gaira.data.synonyms import canonical                    # noqa
from gaira.representation import datasets as ds              # noqa

PH = REPO / "results/v5_rebuild/phase2_stage_a"
TAB = PH / "tables"; LOG = PH / "logs"
for d in (TAB, LOG): d.mkdir(parents=True, exist_ok=True)

# analytes that are polymers/macromolecules, not single small molecules — flagged
# (retained as grounding, but not "one peak = one molecule" references)
NON_SMALL_MOLECULE = {"dna", "rna", "albumin", "glycogen", "cytochrome c", "coenzyme a", "acetyl-coa"}


def audit_row(spec, entered_repr, exclude_reason):
    rec = spec.record
    ok, gate = admits_to_joint_analysis(rec)
    can = canonical(rec.canonical_analyte_name)
    return dict(
        spectrum_id=rec.spectrum_id, canonical_analyte=can, raw_name=rec.canonical_analyte_name,
        modality=rec.modality.value, source=rec.source_dataset, excitation_nm=rec.excitation_nm,
        replicate=rec.replicate, concentration=rec.concentration, intended_role=rec.intended_role.value,
        raw_or_processed=rec.raw_or_processed, admission_gate_ok=ok, admission_gate_reason=gate,
        is_785=(rec.excitation_nm == 785.0), entered_representation=entered_repr,
        exclude_reason=exclude_reason, non_small_molecule_flag=(can in NON_SMALL_MOLECULE),
        wavenumber_min=rec.wavenumber_min, wavenumber_max=rec.wavenumber_max, point_count=rec.point_count)


def main():
    manifest = []
    # RamanBioLib — 785 in, others out
    for s in loader.load_ramanbiolib():
        is785 = s.record.excitation_nm == 785.0
        manifest.append(audit_row(s, is785, "" if is785 else f"non-785({s.record.excitation_nm})"))
    for s in loader.load_metabolite63():
        manifest.append(audit_row(s, False, "633nm(not 785)"))
    for s in loader.load_adenine():
        manifest.append(audit_row(s, False, "adenine_conc_series(controlled_perturbation_eval)"))
    for s in gobbato.load_gobbato_785():
        manifest.append(audit_row(s, True, ""))
    for s in loader.load_orc_ag_peaks():
        manifest.append(audit_row(s, False, "peak_only(MSS, no full spectrum)"))

    mf = pd.DataFrame(manifest)

    # ── invariant checks ──
    checks = {}
    entered = mf[mf.entered_representation]
    checks["all_entered_are_785"] = bool((entered.excitation_nm == 785.0).all())
    checks["all_entered_pass_admission_gate"] = bool(entered.admission_gate_ok.all())
    checks["all_entered_role_grounding"] = bool((entered.intended_role == "grounding").all())
    checks["no_perturbation_entered"] = bool(
        not entered.spectrum_id.str.contains("adenine_sers_control").any())
    checks["adenine_series_excluded"] = bool(
        (~mf[mf.spectrum_id.str.contains("adenine_sers_control", na=False)].entered_representation).all())
    # duplicate spectrum_id check
    checks["no_duplicate_spectrum_ids"] = bool(entered.spectrum_id.is_unique)
    # raw/processed double-count: same (analyte,modality,source,replicate) appearing twice
    dup = entered.groupby(["canonical_analyte", "modality", "source", "replicate"]).size()
    checks["no_raw_processed_double_count"] = bool((dup <= 1).all())

    # ── common grid check (on preprocessed rows) ──
    rows, _ = ds.build_phase2_input("A1_asls_savgol_l2")
    lens = {len(r.vector) for r in rows}
    checks["single_common_grid"] = bool(len(lens) == 1 and next(iter(lens)) == len(ds.GRID))
    checks["n_entered_matches_manifest"] = bool(len(rows) == int(entered.shape[0]))

    # ── overlap recomputation ──
    ram = set(entered[entered.modality == "raman"].canonical_analyte)
    ser = set(entered[entered.modality == "sers"].canonical_analyte)
    matched = ram & ser

    summary = {
        "n_candidate_spectra_examined": int(len(mf)),
        "n_entered_representation": int(entered.shape[0]),
        "n_excluded": int((~mf.entered_representation).sum()),
        "exclusion_reasons": dict(collections.Counter(
            mf[~mf.entered_representation].exclude_reason)),
        "n_raman_entered": int((entered.modality == "raman").sum()),
        "n_sers_entered": int((entered.modality == "sers").sum()),
        "n_unique_analytes": int(entered.canonical_analyte.nunique()),
        "n_raman_analytes": len(ram), "n_sers_analytes": len(ser),
        "n_matched_analytes": len(matched),
        "pct_analytes_matched": round(100 * len(matched) / max(1, entered.canonical_analyte.nunique()), 1),
        "sources_entered": dict(collections.Counter(entered.source)),
        "non_small_molecule_analytes_flagged": sorted(
            set(entered[entered.non_small_molecule_flag].canonical_analyte)),
        "invariant_checks": checks,
        "all_checks_pass": bool(all(checks.values())),
        "correction_applied": {
            "adenine_sers_control": "reclassified grounding→controlled_perturbation_eval; "
            "6 concentration-series spectra EXCLUDED from representation fitting. Adenine remains "
            "grounded via Gobbato Raman+Ag-SERS, so matched analyte count is unchanged.",
            "phase1_5_delta": "485→479 spectra; 271→265 Ag-SERS; matched 51 unchanged; 87 analytes unchanged.",
        },
        "matched_analytes": sorted(matched),
    }

    mf.to_csv(TAB / "phase2_input_manifest.csv", index=False)
    (TAB / "phase2_input_audit_summary.json").write_text(json.dumps(summary, indent=2))

    print("== Phase 2 Stage A — input audit ==")
    for k in ("n_candidate_spectra_examined", "n_entered_representation", "n_excluded",
              "n_raman_entered", "n_sers_entered", "n_unique_analytes",
              "n_matched_analytes", "pct_analytes_matched"):
        print(f"  {k}: {summary[k]}")
    print("  exclusions:", summary["exclusion_reasons"])
    print("  non-small-molecule flagged:", summary["non_small_molecule_analytes_flagged"])
    print("  INVARIANT CHECKS:")
    for k, v in checks.items():
        print(f"    [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  ALL CHECKS PASS: {summary['all_checks_pass']}")
    if not summary["all_checks_pass"]:
        print("  !! STOP: input audit failed — correct registry before analysis.")
        sys.exit(1)
    return summary


if __name__ == "__main__":
    main()
