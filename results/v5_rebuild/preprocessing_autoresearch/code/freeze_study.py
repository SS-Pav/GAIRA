"""Stage B0 step 1 — FREEZE the study design before any search is run.

Writes the nested split manifests and the acceptance thresholds. Both are frozen
artifacts: the outer test folds are used exactly once, at the very end, and the
acceptance thresholds are fixed before any outer-test number is seen.
"""
from __future__ import annotations
import sys, json, hashlib
from pathlib import Path

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
from gaira.preprocessing_autoresearch import corpus as CO
from gaira.preprocessing_autoresearch import evaluator as EV
from gaira.preprocessing_autoresearch.pareto import REJECT

OUT = REPO / "results/v5_rebuild/preprocessing_autoresearch"
CFG = OUT / "configs"; CFG.mkdir(parents=True, exist_ok=True)

SEED = 0
N_OUTER, N_INNER = 5, 4

# ── acceptance thresholds for outcome P1 — FROZEN BEFORE THE OUTER TEST ──
ACCEPTANCE = {
    "outer_mrr_improvement_abs": 0.08,
    "outer_top1_improvement_abs": 0.05,
    "mrr_bootstrap_ci_excludes_zero": True,
    "peak_effect_relative_improvement": 0.50,
    "sers_replicate_min_frac_of_L2": 0.90,
    "within_modality_retrieval_min_frac": 0.90,
    "peak_retention_min": 0.90,
    "peak_invention_max": 0.02,
    "duplicate_collapse_increase_allowed": 0.0,
    "min_fraction_outer_folds_stable": 0.80,
    "reference_baseline": "BASE_asls_sg_l2",
}


def main():
    raw, meta = CO.load_raw_frozen()
    matched = CO.matched_analytes(meta)
    splits = EV.make_nested_splits(matched, n_outer=N_OUTER, n_inner=N_INNER, seed=SEED)
    chk = EV.verify_nested_no_leakage(splits)
    assert chk["ok"], f"SPLIT LEAKAGE: {chk['problems']}"

    # spectrum-level id lists per outer fold (both modalities of a test analyte are test-only)
    for f in splits["folds"]:
        te = set(f["test_analytes"])
        f["test_spectrum_ids"] = meta[meta.analyte.isin(te)].spectrum_id.tolist()
        f["devel_spectrum_ids"] = meta[~meta.analyte.isin(te)].spectrum_id.tolist()

    (CFG / "nested_splits.json").write_text(json.dumps(splits, indent=2))
    (CFG / "acceptance_thresholds.json").write_text(json.dumps(ACCEPTANCE, indent=2))
    (CFG / "rejection_rules.json").write_text(json.dumps(REJECT, indent=2))
    fp = hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()[:32]
    (CFG / "study_manifest.json").write_text(json.dumps({
        "seed": SEED, "n_outer": N_OUTER, "n_inner": N_INNER,
        "n_spectra": len(meta), "n_analytes": int(meta.analyte.nunique()),
        "n_matched_analytes": len(matched),
        "n_raman": int((meta.modality == "raman").sum()),
        "n_sers": int((meta.modality == "sers").sum()),
        "splits_fingerprint": fp,
        "outer_test_used": False,
        "grid": "520-1750 cm-1 @ 2 cm-1 (fixed, not optimised)",
    }, indent=2))

    print("== Stage B0 study frozen ==")
    print(f"  spectra {len(meta)} | analytes {meta.analyte.nunique()} | matched {len(matched)}")
    print(f"  nested splits: {N_OUTER} outer x {N_INNER} inner  (leakage check PASS)")
    for f in splits["folds"]:
        print(f"    outer {f['outer_fold']}: {len(f['test_analytes'])} test analytes, "
              f"{len(f['devel_analytes'])} devel")
    print(f"  splits fingerprint {fp}")
    print(f"  acceptance thresholds + rejection rules frozen -> {CFG}")


if __name__ == "__main__":
    main()
