#!/usr/bin/env python3
"""GAIRA V7 Phase 00 — validation suite.

Independent of run_phase00.py: it re-reads the written artefacts and re-derives every
claim from them, rather than trusting values the pipeline held in memory. Emits a
PASS / FAIL / WARN row per validation item.

    python results/v7_rebuild/phase00/code/validate_phase00.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v7_paths as P                                                   # noqa: E402
import v7_quality as Q                                                 # noqa: E402
import v7_splits as SP                                                 # noqa: E402

R: list[dict] = []


def chk(item: str, category: str, status: str, expected, got, note: str = "") -> None:
    R.append({"item": item, "category": category, "status": status,
              "expected": expected, "got": got, "note": note})


def ok(item, cat, cond, expected, got, note=""):
    chk(item, cat, "PASS" if cond else "FAIL", expected, got, note)


def warn_if(item, cat, cond, expected, got, note=""):
    chk(item, cat, "PASS" if cond else "WARN", expected, got, note)


def main() -> int:
    T, M = P.TABLES, P.MANIFESTS

    # ── A. artefact presence ──────────────────────────────────────────────────
    required_tables = [
        "canonical_analytes_v1.csv", "alias_table_v1.csv", "alias_near_miss_audit_v1.csv",
        "chemical_partition_v1.csv", "class_conflicts_v1.csv", "class_census_v1.csv",
        "replicate_groups_v1.csv", "replicate_group_key_comparison_v1.csv",
        "spectrum_quality_v1.csv", "cv_folds_v1.csv", "cv_fold_summary_v1.csv",
        "benchmark_lock_v1.csv", "frozen_dependency_graph_v1.csv",
        "phase00_baseline_metrics.csv", "phase00_baseline_gain_v1.csv",
        "phase00_corpus_checks.csv",
    ]
    for t in required_tables:
        ok(f"artifact_present.{t}", "artifacts", (T / t).is_file(), True, (T / t).is_file())
    for m in ["dataset_card_v7.json", "cv_splits_v1.json", "alias_leakage_report_v1.json",
              "partition_rationale_v1.json", "quality_summary_v1.json",
              "phase00_component_baseline_v1.json", "phase_00_manifest_v1.json"]:
        ok(f"artifact_present.{m}", "artifacts", (M / m).is_file(), True, (M / m).is_file())
    ok("artifact_present.PHASE_STATE.json", "artifacts",
       (P.PHASE00 / "PHASE_STATE.json").is_file(), True, (P.PHASE00 / "PHASE_STATE.json").is_file())

    canon = pd.read_csv(T / "canonical_analytes_v1.csv")
    alias = pd.read_csv(T / "alias_table_v1.csv")
    part = pd.read_csv(T / "chemical_partition_v1.csv")
    census = pd.read_csv(T / "class_census_v1.csv")
    folds = pd.read_csv(T / "cv_folds_v1.csv")
    bench = pd.read_csv(T / "benchmark_lock_v1.csv")
    base = pd.read_csv(T / "phase00_baseline_metrics.csv")
    card = json.loads((M / "dataset_card_v7.json").read_text())
    splits = json.loads((M / "cv_splits_v1.json").read_text())
    leak = json.loads((M / "alias_leakage_report_v1.json").read_text())
    manifest = json.loads((M / "phase_00_manifest_v1.json").read_text())
    state = json.loads((P.PHASE00 / "PHASE_STATE.json").read_text())

    # ── B. frozen atlas integrity ─────────────────────────────────────────────
    H = np.asarray(np.load(P.FOUNDATION / "manifold_components.npz")["components"], float)
    fp = P.sha256_array(H)
    ok("atlas.fingerprint_recomputed", "benchmark_lock",
       fp == P.CANONICAL_ATLAS_FINGERPRINT, P.CANONICAL_ATLAS_FINGERPRINT, fp)
    ok("atlas.shape", "benchmark_lock", H.shape == (24, 676), "(24, 676)", str(H.shape))
    ok("atlas.nonnegative", "benchmark_lock", bool((H >= 0).all()), True, bool((H >= 0).all()))

    fman = json.loads((P.FOUNDATION / "MANIFEST.json").read_text())
    bad = [n for n, rec in fman["files"].items()
           if P.sha256_file(P.FOUNDATION / n) != rec["sha256"]]
    ok("atlas.all_file_hashes_match", "benchmark_lock", not bad, [], bad)

    lock = bench[bench.status != "PASS"]
    ok("benchmark_lock.all_checks_pass", "benchmark_lock", len(lock) == 0, 0, len(lock),
       "" if len(lock) == 0 else "; ".join(lock.check.tolist()))
    reb = bench[bench.check == "rebuild.max_abs_difference"]
    if len(reb):
        ok("benchmark_lock.rebuild_bit_exact", "benchmark_lock",
           float(reb.iloc[0].got) == 0.0, 0.0, float(reb.iloc[0].got),
           "NMF refitted from raw reproduces the frozen basis exactly")
    else:
        chk("benchmark_lock.rebuild_bit_exact", "benchmark_lock", "WARN",
            0.0, "not run", "degraded mode — raw root unavailable")

    # ── C. corpus ─────────────────────────────────────────────────────────────
    ok("corpus.n_spectra", "corpus", card["n_spectra"] == 375, 375, card["n_spectra"])
    ok("corpus.n_analytes", "corpus", card["n_analytes"] == 167, 167, card["n_analytes"])
    ok("corpus.n_bins", "corpus", card["n_bins"] == 676, 676, card["n_bins"])
    ok("corpus.window", "corpus", card["window_cm"] == [450.0, 1800.0], [450.0, 1800.0],
       card["window_cm"])
    ok("corpus.preprocessing_unchanged", "corpus", card["preprocessing"] == dict(P.PREPROC),
       dict(P.PREPROC), card["preprocessing"])
    ok("corpus.raman_only", "corpus", "Ag-SERS" in card["excluded_domains"],
       "Ag-SERS excluded", card["excluded_domains"][:3])

    # ── D. canonical identity ─────────────────────────────────────────────────
    ok("canonical.ids_unique", "identity", canon.canonical_id.is_unique, True,
       canon.canonical_id.is_unique)
    ok("canonical.every_surface_form_mapped", "identity",
       alias.surface_form.nunique() == card["n_analytes"], card["n_analytes"],
       alias.surface_form.nunique())
    ok("canonical.one_id_per_surface_form", "identity",
       alias.groupby("surface_form").canonical_id.nunique().max() == 1, 1,
       int(alias.groupby("surface_form").canonical_id.nunique().max()))
    ok("canonical.count", "identity", len(canon) == leak["n_canonical_ids"],
       leak["n_canonical_ids"], len(canon))
    ok("canonical.spectra_conserved", "identity",
       int(canon.n_spectra.sum()) == card["n_spectra"], card["n_spectra"],
       int(canon.n_spectra.sum()),
       "merging must not create or destroy spectra")
    ok("canonical.enantiomers_protected", "identity",
       (canon.canonical_id == "(+)-arabinose").any() and (canon.canonical_id == "(-)-arabinose").any(),
       "both present", "checked",
       "(+)- and (-)-arabinose must remain distinct canonical IDs")
    ok("canonical.anomer_protected", "identity",
       (canon.canonical_id == "β-d-glucose").any(), "present", "checked",
       "β-D-glucose must not be merged into (+)-glucose")
    near = pd.read_csv(T / "alias_near_miss_audit_v1.csv")
    unres = near[near.decision == "NOT_MERGED_UNRESOLVED"]
    warn_if("canonical.no_unresolved_near_misses", "identity", len(unres) == 0, 0, len(unres),
            "; ".join(f"{r.form_a} vs {r.form_b}" for _, r in unres.iterrows()))

    # ── E. partition ──────────────────────────────────────────────────────────
    ok("partition.covers_every_id", "partition",
       part.fine_class.astype(bool).all(), True, bool(part.fine_class.astype(bool).all()))
    ok("partition.no_unknown_class", "partition",
       not (part.fine_class == "unknown").any(), False,
       bool((part.fine_class == "unknown").any()))
    ok("partition.every_class_has_rationale", "partition",
       census.rationale.astype(str).str.len().gt(0).all(), True,
       bool(census.rationale.astype(str).str.len().gt(0).all()))
    ok("partition.one_class_per_id", "partition",
       part.groupby("canonical_id").fine_class.nunique().max() == 1, 1,
       int(part.groupby("canonical_id").fine_class.nunique().max()))
    ok("partition.analytes_conserved", "partition",
       int(census.n_canonical_analytes.sum()) == len(canon), len(canon),
       int(census.n_canonical_analytes.sum()))
    conf = census[census.source_confounded]
    warn_if("partition.no_source_confounded_class", "partition", len(conf) == 0, 0, len(conf),
            "; ".join(f"{r.fine_class} ({r.dominant_source_fraction:.0%} {r.dominant_source})"
                      for _, r in conf.iterrows()))
    tiny = census[census.k_c_ceiling < 1]
    warn_if("partition.every_class_supports_a_motif", "partition", len(tiny) == 0, 0, len(tiny),
            "; ".join(tiny.fine_class.tolist()))

    # ── F. splits ─────────────────────────────────────────────────────────────
    lc = splits["leakage_checks"]
    ok("splits.no_canonical_id_across_folds", "splits",
       lc["canonical_id_across_folds"] is False, False, lc["canonical_id_across_folds"])
    ok("splits.no_alias_collision", "splits",
       lc["alias_collision"] is False, False, lc["alias_collision"])
    ok("splits.no_replicate_across_folds", "splits",
       lc["replicate_across_folds"] is False, False, lc["replicate_across_folds"])
    ok("splits.all_checks_false", "splits", splits["all_checks_false"] is True, True,
       splits["all_checks_false"])
    ok("splits.grouping_is_canonical_id", "splits", splits["grouping"] == "canonical_id",
       "canonical_id", splits["grouping"])
    ok("splits.every_id_assigned", "splits", len(folds) == len(canon), len(canon), len(folds))
    ok("splits.n_folds", "splits", folds.fold.nunique() == SP.N_FOLDS, SP.N_FOLDS,
       int(folds.fold.nunique()))
    # determinism: re-cut the folds from the written tables and compare
    refolds = SP.make_folds(canon, part)
    same = bool((refolds.sort_values("canonical_id").fold.values ==
                 folds.sort_values("canonical_id").fold.values).all())
    ok("splits.deterministic_recut", "splits", same, "identical",
       "identical" if same else "DIFFERENT",
       "re-running make_folds on the written tables reproduces the fold assignment")
    sizes = folds.groupby("fold").size()
    warn_if("splits.balanced", "splits", (sizes.max() - sizes.min()) <= 8,
            "max-min <= 8", int(sizes.max() - sizes.min()))

    # ── G. quality ────────────────────────────────────────────────────────────
    if (T / "spectrum_quality_v1.csv").is_file():
        qual = pd.read_csv(T / "spectrum_quality_v1.csv")
        qsum = json.loads((M / "quality_summary_v1.json").read_text())
        ok("quality.rows_match_spectra", "quality", len(qual) == card["n_spectra"],
           card["n_spectra"], len(qual))
        ok("quality.spectrum_id_unique", "quality", qual.spectrum_id.is_unique, True,
           bool(qual.spectrum_id.is_unique))
        ok("quality.in_unit_interval", "quality",
           bool(((qual.quality_score >= 0) & (qual.quality_score <= 1)).all()), True,
           bool(((qual.quality_score >= 0) & (qual.quality_score <= 1)).all()))
        ok("quality.weights_sum_to_one_per_molecule", "quality",
           qsum["weights_sum_to_one"] is True, True, qsum["weights_sum_to_one"])
        ok("quality.q_version_frozen", "quality", qsum["q_version"] == Q.Q_VERSION,
           Q.Q_VERSION, qsum["q_version"])
        ok("quality.discriminative", "quality", qsum["q_max_over_min"] > 1.5, "> 1.5",
           qsum["q_max_over_min"],
           "a degenerate q would make Strategy B identical to the control by construction")
        warn_if("quality.no_spectra_below_floor", "quality", qsum["n_below_qc_floor"] == 0,
                0, qsum["n_below_qc_floor"])
        warn_if("quality.no_nan_bins", "quality", qsum["n_spectra_with_nan_bins"] == 0, 0,
                qsum["n_spectra_with_nan_bins"],
                f"{qsum['total_nan_bins']} NaN bins total (grid-edge effect)")

    # ── H. baseline ───────────────────────────────────────────────────────────
    mss_fine = base[(base.level == "mss") & (base.labels == "v7_fine_16")]
    ok("baseline.mss_fine_reproduces_v63", "baseline",
       len(mss_fine) and abs(float(mss_fine.retrieval_p1.iat[0]) - 0.6707) < 5e-4,
       0.6707, float(mss_fine.retrieval_p1.iat[0]) if len(mss_fine) else None,
       "V6.3 published MSS fine retrieval@1")
    coord_fine = base[(base.level == "coord") & (base.labels == "v7_fine_16")]
    ok("baseline.coord_fine_reproduces_v63", "baseline",
       len(coord_fine) and abs(float(coord_fine.retrieval_p1.iat[0]) - 0.6467) < 5e-4,
       0.6467, float(coord_fine.retrieval_p1.iat[0]) if len(coord_fine) else None)
    broad = base[(base.level == "coord") & (base.labels == "v7_broad_6")]
    ok("baseline.coord_broad_reproduces_v63", "baseline",
       len(broad) and abs(float(broad.retrieval_p1.iat[0]) - 0.8204) < 5e-4,
       0.8204, float(broad.retrieval_p1.iat[0]) if len(broad) else None)
    rnd = base[base.labels == "size_matched_random"]
    ok("baseline.random_control_near_chance", "baseline",
       bool((rnd.retrieval_p1 < 0.15).all()), "< 0.15",
       round(float(rnd.retrieval_p1.max()), 4),
       "size-matched random ontologies must score near chance")
    ok("baseline.random_control_count", "baseline",
       len(rnd) == 4, 4, len(rnd), "one random control per level")

    # ── I. provenance & determinism ───────────────────────────────────────────
    ok("manifest.schema", "provenance",
       manifest["schema"] == "gaira_v7_phase_manifest_v1", "gaira_v7_phase_manifest_v1",
       manifest["schema"])
    ok("manifest.every_output_hashed", "provenance",
       all(o.get("sha256") for o in manifest["outputs"]), True,
       all(o.get("sha256") for o in manifest["outputs"]))
    ok("manifest.every_input_hashed", "provenance",
       all(i.get("sha256") for i in manifest["inputs"]), True,
       all(i.get("sha256") for i in manifest["inputs"]))
    ok("manifest.seeds_recorded", "provenance", bool(manifest.get("seeds")), True,
       bool(manifest.get("seeds")))
    ok("manifest.environment_recorded", "provenance",
       bool(manifest.get("environment", {}).get("python")), True,
       bool(manifest.get("environment", {}).get("python")))
    ok("manifest.decisions_preregistered", "provenance",
       all(d.get("rule_preregistered_in") for d in manifest["decisions"]), True,
       all(d.get("rule_preregistered_in") for d in manifest["decisions"]))
    warn_if("manifest.code_clean", "provenance", manifest["code"]["dirty"] is False, False,
            manifest["code"]["dirty"],
            "a manifest produced from a dirty tree cannot be reproduced from its commit")

    # output hashes still match the files on disk
    stale = []
    for o in manifest["outputs"]:
        p = P.REPO / o["path"]
        if p.is_file() and P.sha256_file(p) != o["sha256"]:
            stale.append(o["artifact_id"])
    ok("manifest.output_hashes_current", "provenance", not stale, [], stale)

    # ── J. isolation: nothing outside the V7 tree was written ─────────────────
    ok("isolation.frozen_assets_untouched", "isolation", not bad, [], bad,
       "assets/foundation file hashes still match the frozen manifest")
    ok("isolation.write_root", "isolation",
       all(o["path"].startswith("results/v7_rebuild/phase00/") for o in manifest["outputs"]),
       "results/v7_rebuild/phase00/",
       sorted({o["path"].split("/")[0] for o in manifest["outputs"]}))

    # ── K. phase state ────────────────────────────────────────────────────────
    ok("phase_state.status", "state", state["status"] == "COMPLETE", "COMPLETE", state["status"])
    ok("phase_state.fingerprint", "state",
       state["atlas_fingerprint"] == P.CANONICAL_ATLAS_FINGERPRINT,
       P.CANONICAL_ATLAS_FINGERPRINT, state["atlas_fingerprint"])
    ok("phase_state.all_gates_pass", "state", all(g["passed"] for g in state["gates"]), True,
       all(g["passed"] for g in state["gates"]))
    ok("phase_state.next_phase_not_started", "state", "NOT STARTED" in state["next_phase"],
       "Phase 01 NOT STARTED", state["next_phase"])

    # ── emit ──────────────────────────────────────────────────────────────────
    df = pd.DataFrame(R)[["category", "item", "status", "expected", "got", "note"]]
    P.VALIDATION.mkdir(parents=True, exist_ok=True)
    out = P.VALIDATION / "phase00_validation_v1.csv"
    df.to_csv(out, index=False, lineterminator="\n")

    summ = (df.groupby(["category", "status"]).size().unstack(fill_value=0)
            .reindex(columns=["PASS", "WARN", "FAIL"], fill_value=0).reset_index())
    summ.to_csv(P.VALIDATION / "phase00_validation_summary_v1.csv", index=False,
                lineterminator="\n")

    n_pass = int((df.status == "PASS").sum())
    n_warn = int((df.status == "WARN").sum())
    n_fail = int((df.status == "FAIL").sum())
    print(summ.to_string(index=False))
    print(f"\nTOTAL  PASS {n_pass}   WARN {n_warn}   FAIL {n_fail}   (of {len(df)})")
    if n_warn:
        print("\nWARN items:")
        for _, r in df[df.status == "WARN"].iterrows():
            print(f"  - {r['item']}: {r['note'] or r['got']}")
    if n_fail:
        print("\nFAIL items:")
        for _, r in df[df.status == "FAIL"].iterrows():
            print(f"  - {r['item']}: expected {r['expected']} got {r['got']}")
    print(f"\nwritten: {out.relative_to(P.REPO)}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
